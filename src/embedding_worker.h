/* -------------------------------------------------------------------------
 *
 * embedding_worker.h
 * Declarations for the embedding background worker
 *
 * This header defines types, structures, and function prototypes
 * used for generating embeddings for text columns.
 *
 * -------------------------------------------------------------------------
 */
#ifndef EMBEDDING_WORKER_H
#define EMBEDDING_WORKER_H

#include "postgres.h"

/* Main entry point for the background worker */
extern void embedding_worker_main(Datum main_arg);

/* Module initialization function */
extern void _PG_init(void);

/* Structure representing an embedding job */
typedef struct EmbeddingJob
{
    int job_id;
    char *source_schema;
    char *source_table;
    char *source_column;
    char *source_id_column;
    char *target_schema;
    char *target_table;
    char *target_column;
    char *embedder;
    char *model;
} EmbeddingJob;

#endif /* EMBEDDING_WORKER_H */
