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
    text *embedder_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    text *input_text = PG_GETARG_TEXT_P(2);

    char *embedder_str = text_to_cstring(embedder_text);
    char *model_str = text_to_cstring(model_text);

    int embedder_id = validate_embedder(embedder_str);
    if (embedder_id < 0)
        elog(ERROR, "Invalid embedder: %s", embedder_str);

    int model_id = validate_embedding_model(embedder_id, model_str, INPUT_TYPE_TEXT);
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
    int err = generate_embeddings(embedder_id, model_id, &input_data, &batch);

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
    text *embedder_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    ArrayType *input_array = PG_GETARG_ARRAYTYPE_P(2);
    Datum *text_elems;
    bool *nulls;
    int nitems;

    char *embedder_str = text_to_cstring(embedder_text);
    char *model_str = text_to_cstring(model_text);

    int embedder_id = validate_embedder(embedder_str);
    if (embedder_id < 0)
        elog(ERROR, "Invalid embedder: %s", embedder_str);

    int model_id = validate_embedding_model(embedder_id, model_str, INPUT_TYPE_TEXT);
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
    int err = generate_embeddings(embedder_id, model_id, &input_data, &batch);

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
    text *embedder_text = PG_GETARG_TEXT_P(0);
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

        char *embedder_str = text_to_cstring(embedder_text);
        char *model_str = text_to_cstring(model_text);

        int embedder_id = validate_embedder(embedder_str);
        if (embedder_id < 0)
            elog(ERROR, "Invalid embedder: %s", embedder_str);

        int model_id = validate_embedding_model(embedder_id, model_str, INPUT_TYPE_TEXT);
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
        int err = generate_embeddings(embedder_id, model_id, &input_data, &batch);

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

PG_FUNCTION_INFO_V1(embed_image);

Datum embed_image(PG_FUNCTION_ARGS)
{
    text *embedder_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    bytea *input_bytea = PG_GETARG_BYTEA_P(2);

    char *embedder_str = text_to_cstring(embedder_text);
    char *model_str = text_to_cstring(model_text);

    int embedder_id = validate_embedder(embedder_str);
    if (embedder_id < 0)
        elog(ERROR, "Invalid embedder: %s", embedder_str);

    int model_id = validate_embedding_model(embedder_id, model_str, INPUT_TYPE_IMAGE);
    if (model_id < 0)
        elog(ERROR, "Model not allowed: %s", model_str);

    ByteSlice c_input;
    c_input.ptr = (unsigned char *)VARDATA_ANY(input_bytea);
    c_input.len = VARSIZE_ANY_EXHDR(input_bytea);

    InputData input_data = {
        .input_type = INPUT_TYPE_IMAGE,
        .binary_data = &c_input,
        .n_binary = 1,
        .text_data = NULL,
        .n_text = 0
    };

    EmbeddingBatch batch;
    int err = generate_embeddings(embedder_id, model_id, &input_data, &batch);

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

PG_FUNCTION_INFO_V1(embed_images);

Datum embed_images(PG_FUNCTION_ARGS)
{
    text *embedder_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    ArrayType *input_array = PG_GETARG_ARRAYTYPE_P(2);
    Datum *bytea_elems;
    bool *nulls;
    int nitems;

    char *embedder_str = text_to_cstring(embedder_text);
    char *model_str = text_to_cstring(model_text);

    int embedder_id = validate_embedder(embedder_str);
    if (embedder_id < 0)
        elog(ERROR, "Invalid embedder: %s", embedder_str);

    int model_id = validate_embedding_model(embedder_id, model_str, INPUT_TYPE_IMAGE);
    if (model_id < 0)
        elog(ERROR, "Model not allowed: %s", model_str);

    deconstruct_array(
        input_array,
        BYTEAOID,
        -1,
        false,
        'i',
        &bytea_elems,
        &nulls,
        &nitems
    );

    if (nitems == 0)
        PG_RETURN_NULL();

    ByteSlice *c_inputs = palloc(sizeof(ByteSlice) * nitems);
    for (int i = 0; i < nitems; i++)
    {
        bytea *b = DatumGetByteaP(bytea_elems[i]);
        c_inputs[i].ptr = (unsigned char *)VARDATA_ANY(b);
        c_inputs[i].len = VARSIZE_ANY_EXHDR(b);
    }

    InputData input_data = {
        .input_type = INPUT_TYPE_IMAGE,
        .binary_data = c_inputs,
        .n_binary = nitems,
        .text_data = NULL,
        .n_text = 0
    };

    EmbeddingBatch batch;
    int err = generate_embeddings(embedder_id, model_id, &input_data, &batch);

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

PG_FUNCTION_INFO_V1(embed_images_with_ids);

Datum
embed_images_with_ids(PG_FUNCTION_ARGS)
{
    text *embedder_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    ArrayType *ids_array = PG_GETARG_ARRAYTYPE_P(2);
    ArrayType *images_array = PG_GETARG_ARRAYTYPE_P(3);

    Datum *id_elems;
    bool *id_nulls;
    int n_ids;

    Datum *image_elems;
    bool *image_nulls;
    int n_images;

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

        char *embedder_str = text_to_cstring(embedder_text);
        char *model_str = text_to_cstring(model_text);

        int embedder_id = validate_embedder(embedder_str);
        if (embedder_id < 0)
            elog(ERROR, "Invalid embedder: %s", embedder_str);

        int model_id = validate_embedding_model(embedder_id, model_str, INPUT_TYPE_IMAGE);
        if (model_id < 0)
            elog(ERROR, "Model not allowed: %s", model_str);

        deconstruct_array(ids_array, INT4OID, 4, true, 'i',
                          &id_elems, &id_nulls, &n_ids);
        deconstruct_array(images_array, BYTEAOID, -1, false, 'i',
                          &image_elems, &image_nulls, &n_images);

        if (n_ids != n_images)
            elog(ERROR, "Identifiers and images arrays must have same length");

        ByteSlice *c_inputs = palloc(sizeof(ByteSlice) * n_images);
        int *c_ids = palloc(sizeof(int) * n_ids);

        for (int i = 0; i < n_images; i++)
        {
            if (id_nulls[i] || image_nulls[i])
                elog(ERROR, "NULL values not allowed");

            c_ids[i] = DatumGetInt32(id_elems[i]);
            bytea *b = DatumGetByteaP(image_elems[i]);
            c_inputs[i].ptr = (unsigned char *)VARDATA_ANY(b);
            c_inputs[i].len = VARSIZE_ANY_EXHDR(b);
        }

        InputData input_data = {
            .input_type = INPUT_TYPE_IMAGE,
            .binary_data = c_inputs,
            .n_binary = n_images,
            .text_data = NULL,
            .n_text = 0
        };

        EmbeddingBatch batch;
        int err = generate_embeddings(embedder_id, model_id, &input_data, &batch);

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
    text *embedder_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    ArrayType *images_array = PG_ARGISNULL(2) ? NULL : PG_GETARG_ARRAYTYPE_P(2);
    ArrayType *text_array = PG_ARGISNULL(3) ? NULL : PG_GETARG_ARRAYTYPE_P(3);

    char *embedder_str = text_to_cstring(embedder_text);
    char *model_str = text_to_cstring(model_text);

    int embedder_id = validate_embedder(embedder_str);
    if (embedder_id < 0)
        elog(ERROR, "Invalid embedder: %s", embedder_str);

    int model_id = validate_embedding_model(embedder_id, model_str, INPUT_TYPE_MULTIMODAL);
    if (model_id < 0)
        elog(ERROR, "Model not allowed for multimodal embedding: %s", model_str);

    ByteSlice *c_images = NULL;
    int n_images = 0;
    if (images_array != NULL)
    {
        Datum *bytea_elems;
        bool *nulls;
        deconstruct_array(images_array, BYTEAOID, -1, false, 'i', &bytea_elems, &nulls, &n_images);

        if (n_images > 0)
        {
            c_images = palloc(sizeof(ByteSlice) * n_images);
            for (int i = 0; i < n_images; i++)
            {
                bytea *b = DatumGetByteaP(bytea_elems[i]);
                c_images[i].ptr = (unsigned char *)VARDATA_ANY(b);
                c_images[i].len = VARSIZE_ANY_EXHDR(b);
            }
        }
    }

    StringSlice *c_texts = NULL;
    int n_texts = 0;
    if (text_array != NULL)
    {
        Datum *text_elems;
        bool *nulls;
        deconstruct_array(text_array, TEXTOID, -1, false, 'i', &text_elems, &nulls, &n_texts);

        if (n_texts > 0)
        {
            c_texts = palloc(sizeof(StringSlice) * n_texts);
            for (int i = 0; i < n_texts; i++)
            {
                text *t = DatumGetTextP(text_elems[i]);
                c_texts[i].ptr = VARDATA_ANY(t);
                c_texts[i].len = VARSIZE_ANY_EXHDR(t);
            }
        }
    }

    if (n_images == 0 && n_texts == 0)
        elog(ERROR, "At least one of images or texts must be provided");

    InputData input_data = {
        .input_type = INPUT_TYPE_MULTIMODAL,
        .binary_data = c_images,
        .n_binary = n_images,
        .text_data = c_texts,
        .n_text = n_texts
    };

    EmbeddingBatch batch;
    int err = generate_embeddings(embedder_id, model_id, &input_data, &batch);

    if (c_images)
        pfree(c_images);
    if (c_texts)
        pfree(c_texts);

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
