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

PG_FUNCTION_INFO_V1(generate_embeddings);

Datum generate_embeddings(PG_FUNCTION_ARGS)
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
        elog(ERROR, "Invalid embedding method: %s (use 'fastembed' or 'grpc')", method_str);

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

    EmbeddingBatch batch;
    int err = generate_embeddings_from_texts(method_id, model_id, c_inputs, nitems, &batch);

    pfree(c_inputs);

    if (err < 0) {
        free_embedding_batch(&batch);
        elog(ERROR, "embedding generation failed (code=%d)", err);
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

PG_FUNCTION_INFO_V1(generate_embeddings_with_ids);

Datum
generate_embeddings_with_ids(PG_FUNCTION_ARGS)
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
            elog(ERROR, "Invalid embedding method: %s (use 'fastembed' or 'grpc')", method_str);

        int model_id = validate_embedding_model(method_id, model_str, INPUT_TYPE_TEXT);
        if (model_id < 0)
            elog(ERROR, "Model not allowed: %s", model_str);

        deconstruct_array(ids_array, INT4OID, 4, true, 'i',
                          &id_elems, &id_nulls, &n_ids);
        deconstruct_array(texts_array, TEXTOID, -1, false, 'i',
                          &text_elems, &text_nulls, &n_texts);

        if (n_ids != n_texts)
            elog(ERROR, "ids and texts arrays must have same length");

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

        EmbeddingBatch batch;
        int err = generate_embeddings_from_texts(method_id, model_id, c_inputs, n_texts, &batch);

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
