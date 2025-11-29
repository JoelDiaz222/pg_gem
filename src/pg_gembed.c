#include "pg_gembed.h"
#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "catalog/namespace.h"
#include "vector.h"

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(embed_text);

Datum embed_text(PG_FUNCTION_ARGS)
{
    text *method_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    text *input_text = PG_GETARG_TEXT_P(2);

    char *method_str = text_to_cstring(method_text);
    char *model_str = text_to_cstring(model_text);

    int method_id = validate_embedding_method(method_str);
    if (method_id < 0)
        elog(ERROR, "Invalid embedding method: %s", method_str);

    int model_id = validate_embedding_model(method_id, model_str, INPUT_TYPE_TEXT);
    if (model_id < 0)
        elog(ERROR, "Model not allowed: %s", model_str);

    StringSlice c_input;
    c_input.ptr = VARDATA_ANY(input_text);
    c_input.len = VARSIZE_ANY_EXHDR(input_text);

    InputData input_data = {
        .input_type = INPUT_TYPE_TEXT,
        .binary_data = NULL,
        .n_binary = 0,
        .text_data = &c_input,
        .n_text = 1
    };

    EmbeddingBatch batch;
    int err = generate_embeddings(method_id, model_id, &input_data, &batch);

    if (err < 0) {
        free_embedding_batch(&batch);
        elog(ERROR, "Embedding generation failed (code=%d)", err);
    }

    if (batch.n_vectors != 1)
    {
        free_embedding_batch(&batch);
        elog(ERROR, "Expected 1 embedding, got %zu", batch.n_vectors);
    }

    Vector *v = (Vector *)palloc(VECTOR_SIZE(batch.dim));
    SET_VARSIZE(v, VECTOR_SIZE(batch.dim));
    v->dim = batch.dim;
    v->unused = 0;
    memcpy(v->x, batch.data, sizeof(float) * batch.dim);

    free_embedding_batch(&batch);

    PG_RETURN_POINTER(v);
}

PG_FUNCTION_INFO_V1(embed_texts);

Datum embed_texts(PG_FUNCTION_ARGS)
{
    text *method_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    ArrayType *input_array = PG_GETARG_ARRAYTYPE_P(2);
    Datum *text_elems;
    bool *nulls;
    int nitems;

    char *method_str = text_to_cstring(method_text);
    char *model_str = text_to_cstring(model_text);

    int method_id = validate_embedding_method(method_str);
    if (method_id < 0)
        elog(ERROR, "Invalid embedding method: %s", method_str);

    int model_id = validate_embedding_model(method_id, model_str, INPUT_TYPE_TEXT);
    if (model_id < 0)
        elog(ERROR, "Model not allowed: %s", model_str);

    deconstruct_array(
        input_array,
        TEXTOID,
        -1,
        false,
        'i',
        &text_elems,
        &nulls,
        &nitems
    );

    if (nitems == 0)
        PG_RETURN_NULL();

    StringSlice *c_inputs = palloc(sizeof(StringSlice) * nitems);
    for (int i = 0; i < nitems; i++)
    {
        text *t = DatumGetTextP(text_elems[i]);
        c_inputs[i].ptr = VARDATA_ANY(t);
        c_inputs[i].len = VARSIZE_ANY_EXHDR(t);
    }

    InputData input_data = {
        .input_type = INPUT_TYPE_TEXT,
        .binary_data = NULL,
        .n_binary = 0,
        .text_data = c_inputs,
        .n_text = nitems
    };

    EmbeddingBatch batch;
    int err = generate_embeddings(method_id, model_id, &input_data, &batch);

    pfree(c_inputs);

    if (err < 0) {
        free_embedding_batch(&batch);
        elog(ERROR, "Embedding generation failed (code=%d)", err);
    }

    Datum *vectors = palloc(sizeof(Datum) * batch.n_vectors);
    for (size_t i = 0; i < batch.n_vectors; i++)
    {
        Vector *v = (Vector *)palloc(VECTOR_SIZE(batch.dim));
        SET_VARSIZE(v, VECTOR_SIZE(batch.dim));
        v->dim = batch.dim;
        v->unused = 0;
        memcpy(v->x, batch.data + i * batch.dim, sizeof(float) * batch.dim);
        vectors[i] = PointerGetDatum(v);
    }

    Oid vector_type_oid = TypenameGetTypid("vector");
    ArrayType *result = construct_array(
        vectors,
        batch.n_vectors,
        vector_type_oid,
        -1,
        false,
        'd'
    );

    free_embedding_batch(&batch);

    for (size_t i = 0; i < batch.n_vectors; i++)
        pfree(DatumGetPointer(vectors[i]));
    pfree(vectors);

    PG_RETURN_ARRAYTYPE_P(result);
}

PG_FUNCTION_INFO_V1(embed_texts_with_ids);

Datum
embed_texts_with_ids(PG_FUNCTION_ARGS)
{
    text *method_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    ArrayType *ids_array = PG_GETARG_ARRAYTYPE_P(2);
    ArrayType *texts_array = PG_GETARG_ARRAYTYPE_P(3);

    Datum *id_elems;
    bool *id_nulls;
    int n_ids;

    Datum *text_elems;
    bool *text_nulls;
    int n_texts;

    FuncCallContext *funcctx;
    typedef struct
    {
        int *ids;
        Vector **vectors;
        int nitems;
        int current;
    } user_fctx;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext oldcontext;

        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

        char *method_str = text_to_cstring(method_text);
        char *model_str = text_to_cstring(model_text);

        int method_id = validate_embedding_method(method_str);
        if (method_id < 0)
            elog(ERROR, "Invalid embedding method: %s", method_str);

        int model_id = validate_embedding_model(method_id, model_str, INPUT_TYPE_TEXT);
        if (model_id < 0)
            elog(ERROR, "Model not allowed: %s", model_str);

        deconstruct_array(ids_array, INT4OID, 4, true, 'i',
                          &id_elems, &id_nulls, &n_ids);
        deconstruct_array(texts_array, TEXTOID, -1, false, 'i',
                          &text_elems, &text_nulls, &n_texts);

        if (n_ids != n_texts)
            elog(ERROR, "Identifiers and texts arrays must have same length");

        StringSlice *c_inputs = palloc(sizeof(StringSlice) * n_texts);
        int *c_ids = palloc(sizeof(int) * n_ids);

        for (int i = 0; i < n_texts; i++)
        {
            if (id_nulls[i] || text_nulls[i])
                elog(ERROR, "NULL values not allowed");

            c_ids[i] = DatumGetInt32(id_elems[i]);
            text *t = DatumGetTextP(text_elems[i]);
            c_inputs[i].ptr = VARDATA_ANY(t);
            c_inputs[i].len = VARSIZE_ANY_EXHDR(t);
        }

        InputData input_data = {
            .input_type = INPUT_TYPE_TEXT,
            .binary_data = NULL,
            .n_binary = 0,
            .text_data = c_inputs,
            .n_text = n_texts
        };

        EmbeddingBatch batch;
        int err = generate_embeddings(method_id, model_id, &input_data, &batch);

        pfree(c_inputs);

        if (err != 0)
            elog(ERROR, "embedding generation failed (code=%d)", err);

        Vector **vectors = palloc(sizeof(Vector *) * batch.n_vectors);
        for (size_t i = 0; i < batch.n_vectors; i++)
        {
            Vector *v = (Vector *)palloc(VECTOR_SIZE(batch.dim));
            SET_VARSIZE(v, VECTOR_SIZE(batch.dim));
            v->dim = batch.dim;
            v->unused = 0;
            memcpy(v->x, batch.data + i * batch.dim, sizeof(float) * batch.dim);
            vectors[i] = v;
        }

        size_t n_vectors = batch.n_vectors;

        free_embedding_batch(&batch);

        user_fctx *fctx = palloc(sizeof(user_fctx));
        fctx->ids = c_ids;
        fctx->vectors = vectors;
        fctx->nitems = n_vectors;
        fctx->current = 0;

        funcctx->user_fctx = fctx;

        TupleDesc tupdesc = CreateTemplateTupleDesc(2);
        TupleDescInitEntry(tupdesc, (AttrNumber)1, "sentence_id", INT4OID, -1, 0);
        TupleDescInitEntry(tupdesc, (AttrNumber)2, "embedding", TypenameGetTypid("vector"), -1, 0);
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);

        MemoryContextSwitchTo(oldcontext);
    }

    funcctx = SRF_PERCALL_SETUP();
    user_fctx *fctx = (user_fctx *)funcctx->user_fctx;

    if (fctx->current < fctx->nitems)
    {
        Datum values[2];
        bool nulls[2] = {false, false};
        HeapTuple tuple;

        values[0] = Int32GetDatum(fctx->ids[fctx->current]);
        values[1] = PointerGetDatum(fctx->vectors[fctx->current]);

        tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
        fctx->current++;

        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    else
    {
        SRF_RETURN_DONE(funcctx);
    }
}

PG_FUNCTION_INFO_V1(embed_multimodal);

Datum embed_multimodal(PG_FUNCTION_ARGS)
{
    text *method_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    bytea *image_bytea = PG_ARGISNULL(2) ? NULL : PG_GETARG_BYTEA_P(2);
    ArrayType *text_array = PG_ARGISNULL(3) ? NULL : PG_GETARG_ARRAYTYPE_P(3);

    char *method_str = text_to_cstring(method_text);
    char *model_str = text_to_cstring(model_text);

    int method_id = validate_embedding_method(method_str);
    if (method_id < 0)
        elog(ERROR, "Invalid embedding method: %s", method_str);

    int model_id = validate_embedding_model(method_id, model_str, INPUT_TYPE_MULTIMODAL);
    if (model_id < 0)
        elog(ERROR, "Model not allowed for multimodal: %s", model_str);

    ByteSlice image_slice = {NULL, 0};
    if (image_bytea != NULL)
    {
        image_slice.ptr = (unsigned char *)VARDATA_ANY(image_bytea);
        image_slice.len = VARSIZE_ANY_EXHDR(image_bytea);
    }

    StringSlice *c_inputs = NULL;
    int n_texts = 0;
    if (text_array != NULL)
    {
        Datum *text_elems;
        bool *nulls;
        deconstruct_array(text_array, TEXTOID, -1, false, 'i', &text_elems, &nulls, &n_texts);

        if (n_texts > 0)
        {
            c_inputs = palloc(sizeof(StringSlice) * n_texts);
            for (int i = 0; i < n_texts; i++)
            {
                text *t = DatumGetTextP(text_elems[i]);
                c_inputs[i].ptr = VARDATA_ANY(t);
                c_inputs[i].len = VARSIZE_ANY_EXHDR(t);
            }
        }
    }

    if (image_slice.ptr == NULL && n_texts == 0)
        elog(ERROR, "At least one of image or texts must be provided");

    InputData input_data = {
        .input_type = INPUT_TYPE_MULTIMODAL,
        .binary_data = image_slice.ptr ? &image_slice : NULL,
        .n_binary = image_slice.ptr ? 1 : 0,
        .text_data = c_inputs,
        .n_text = n_texts
    };

    EmbeddingBatch batch;
    int err = generate_embeddings(method_id, model_id, &input_data, &batch);

    if (c_inputs)
        pfree(c_inputs);

    if (err < 0)
    {
        free_embedding_batch(&batch);
        elog(ERROR, "Multimodal embedding generation failed (code=%d)", err);
    }

    Datum *vectors = palloc(sizeof(Datum) * batch.n_vectors);
    for (size_t i = 0; i < batch.n_vectors; i++)
    {
        Vector *v = (Vector *)palloc(VECTOR_SIZE(batch.dim));
        SET_VARSIZE(v, VECTOR_SIZE(batch.dim));
        v->dim = batch.dim;
        v->unused = 0;
        memcpy(v->x, batch.data + i * batch.dim, sizeof(float) * batch.dim);
        vectors[i] = PointerGetDatum(v);
    }

    Oid vector_type_oid = TypenameGetTypid("vector");
    ArrayType *result = construct_array(
        vectors,
        batch.n_vectors,
        vector_type_oid,
        -1,
        false,
        'd'
    );

    free_embedding_batch(&batch);

    for (size_t i = 0; i < batch.n_vectors; i++)
        pfree(DatumGetPointer(vectors[i]));
    pfree(vectors);

    PG_RETURN_ARRAYTYPE_P(result);
}
