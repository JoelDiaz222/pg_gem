#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "catalog/namespace.h"
#include "vector.h"

PG_MODULE_MAGIC;

typedef struct
{
    float **data;
    size_t n_vectors;
    size_t dim;
} EmbeddingBatch;

extern int generate_embeddings_from_texts(
    const char **inputs,
    size_t n_inputs,
    EmbeddingBatch *out_batch
);

extern void free_embedding_batch(EmbeddingBatch *batch);

PG_FUNCTION_INFO_V1(generate_embeddings);

Datum generate_embeddings(PG_FUNCTION_ARGS)
{
    ArrayType *input_array = PG_GETARG_ARRAYTYPE_P(0);
    Datum *text_elems;
    bool *nulls;
    int nitems;

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

    const char **c_inputs = palloc(sizeof(char *) * nitems);
    for (int i = 0; i < nitems; i++)
        c_inputs[i] = TextDatumGetCString(text_elems[i]);

    EmbeddingBatch batch;
    if (generate_embeddings_from_texts(c_inputs, nitems, &batch) != 0)
        elog(ERROR, "embedding generation failed");

    Datum *vectors = palloc(sizeof(Datum) * batch.n_vectors);
    for (size_t i = 0; i < batch.n_vectors; i++)
    {
        Vector *v = (Vector *)palloc(VECTOR_SIZE(batch.dim));
        SET_VARSIZE(v, VECTOR_SIZE(batch.dim));
        v->dim = batch.dim;
        v->unused = 0;
        memcpy(v->x, batch.data[i], sizeof(float) * batch.dim);
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

    PG_RETURN_ARRAYTYPE_P(result);
}

PG_FUNCTION_INFO_V1(generate_embeddings_with_ids);

Datum
generate_embeddings_with_ids(PG_FUNCTION_ARGS)
{
    ArrayType *ids_array = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *texts_array = PG_GETARG_ARRAYTYPE_P(1);

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

        // Deconstruct input arrays
        deconstruct_array(ids_array, INT4OID, 4, true, 'i',
                          &id_elems, &id_nulls, &n_ids);
        deconstruct_array(texts_array, TEXTOID, -1, false, 'i',
                          &text_elems, &text_nulls, &n_texts);

        if (n_ids != n_texts)
            elog(ERROR, "ids and texts arrays must have same length");

        const char **c_inputs = palloc(sizeof(char *) * n_texts);
        int *c_ids = palloc(sizeof(int) * n_ids);

        for (int i = 0; i < n_texts; i++)
        {
            if (id_nulls[i] || text_nulls[i])
                elog(ERROR, "NULL values not allowed");

            c_ids[i] = DatumGetInt32(id_elems[i]);
            c_inputs[i] = TextDatumGetCString(text_elems[i]);
        }

        // Generate embeddings
        EmbeddingBatch batch;
        if (generate_embeddings_from_texts(c_inputs, n_texts, &batch) != 0)
            elog(ERROR, "embedding generation failed");

        // Prepare results
        Vector **vectors = palloc(sizeof(Vector *) * batch.n_vectors);
        for (size_t i = 0; i < batch.n_vectors; i++)
        {
            Vector *v = (Vector *)palloc(VECTOR_SIZE(batch.dim));
            SET_VARSIZE(v, VECTOR_SIZE(batch.dim));
            v->dim = batch.dim;
            v->unused = 0;
            memcpy(v->x, batch.data[i], sizeof(float) * batch.dim);
            vectors[i] = v;
        }

        free_embedding_batch(&batch);

        user_fctx *fctx = palloc(sizeof(user_fctx));
        fctx->ids = c_ids;
        fctx->vectors = vectors;
        fctx->nitems = batch.n_vectors;
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
