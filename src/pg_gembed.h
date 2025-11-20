#ifndef PG_GEMBED_H
#define PG_GEMBED_H

#include "stddef.h"

#define INPUT_TYPE_TEXT 0

/* Structure for storing generated embeddings */
typedef struct
{
    float *data;
    size_t n_vectors;
    size_t dim;
} EmbeddingBatch;

/* Structure for passing text data */
typedef struct
{
    const char *ptr;
    size_t len;
} StringSlice;

/* Validates the embedding method name and returns method ID */
extern int validate_embedding_method(const char *method);

/* Validates the model name for a given method and returns model ID */
extern int validate_embedding_model(int method_id, const char *model, int input_type);

/* Generates embeddings from text inputs using the specified method and model */
extern int generate_embeddings_from_texts(
    int method_id,
    int model_id,
    const StringSlice *inputs,
    size_t n_inputs,
    EmbeddingBatch *out_batch
);

/* Frees memory allocated for an embedding batch */
extern void free_embedding_batch(EmbeddingBatch *batch);

#endif /* PG_GEMBED_H */
