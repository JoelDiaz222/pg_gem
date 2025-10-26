/* -------------------------------------------------------------------------
 *
 * embedding_worker.c
 * Background worker implementation for automatic embedding generation.
 *
 * This worker monitors specified tables and generates embeddings for
 * text columns, storing results in designated embedding columns.
 *
 * -------------------------------------------------------------------------
 */
/* Header files of this project */
#include "embedding_worker.h"
#include "pg_gem.h"

/* These are always necessary for a bgworker */
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "postmaster/interrupt.h"
#include "storage/latch.h"

/* These headers are used by this particular worker's code */
#include "access/xact.h"
#include "executor/spi.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "pgstat.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/snapmgr.h"

PGDLLEXPORT void embedding_worker_main(Datum main_arg);

/*
 * Load all active jobs from the jobs table
 */
static List *
load_embedding_jobs(void)
{
    int ret;
    StringInfoData buf;
    List *jobs = NIL;

    elog(DEBUG1, "entering function %s", __func__);

    initStringInfo(&buf);
    appendStringInfo(&buf,
        "SELECT job_id, source_schema, source_table, source_column, "
        "       source_id_column, target_schema, target_table, "
        "       target_column, method, model "
        "FROM gem_jobs.embedding_jobs "
        "WHERE enabled = true");

    ret = SPI_execute(buf.data, true, 0);
    if (ret != SPI_OK_SELECT)
        elog(ERROR, "failed to load jobs");

    elog(LOG, "Found %lld active embedding jobs.", SPI_processed);

    for (uint64 i = 0; i < SPI_processed; i++)
    {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        EmbeddingJob *job = (EmbeddingJob *)palloc(sizeof(EmbeddingJob));
        bool isnull;

        job->job_id = DatumGetInt32(SPI_getbinval(tuple, tupdesc, 1, &isnull));
        job->source_schema = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 2, &isnull));
        job->source_table = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 3, &isnull));
        job->source_column = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 4, &isnull));
        job->source_id_column = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 5, &isnull));
        job->target_schema = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 6, &isnull));
        job->target_table = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 7, &isnull));
        job->target_column = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 8, &isnull));
        job->method = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 9, &isnull));
        job->model = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 10, &isnull));

        elog(DEBUG1, "Loaded job ID %d (%s.%s -> %s.%s)",
             job->job_id, job->source_schema, job->source_table,
             job->target_schema, job->target_table);

        jobs = lappend(jobs, job);
    }

    return jobs;
}

/*
 * Process a specific embedding job
 */
static void
process_embedding_job(EmbeddingJob *job)
{
    int ret;
    StringInfoData buf;
    int last_processed_id = 0;
    bool isnull;
    int n_rows;
    int *ids = NULL;
    StringSlice *texts = NULL;
    int max_id;
    int i;
    int method_id;
    int model_id;
    EmbeddingBatch batch;
    int err;

    elog(LOG, "Starting to process job ID: %d (%s.%s.%s -> %s.%s.%s)",
         job->job_id, job->source_schema, job->source_table, job->source_column,
         job->target_schema, job->target_table, job->target_column);

    /* Get last processed ID */
    initStringInfo(&buf);
    appendStringInfo(&buf,
        "SELECT last_processed_id FROM gem_jobs.embedding_jobs WHERE job_id = %d",
        job->job_id);

    ret = SPI_execute(buf.data, true, 0);
    if (ret == SPI_OK_SELECT && SPI_processed > 0)
    {
        Datum datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
        if (!isnull)
            last_processed_id = DatumGetInt32(datum);
    }
    elog(DEBUG1, "Job %d: last_processed_id is %d", job->job_id, last_processed_id);

    /* Find rows needing embeddings */
    resetStringInfo(&buf);
    appendStringInfo(&buf,
        "SELECT s.%s, s.%s "
        "FROM %s.%s s "
        "LEFT JOIN %s.%s t ON s.%s = t.%s "
        "WHERE s.%s > %d AND (t.%s IS NULL OR t.%s IS NULL) "
        "ORDER BY s.%s "
        "LIMIT %d",
        quote_identifier(job->source_id_column),
        quote_identifier(job->source_column),
        quote_identifier(job->source_schema),
        quote_identifier(job->source_table),
        quote_identifier(job->target_schema),
        quote_identifier(job->target_table),
        quote_identifier(job->source_id_column),
        quote_identifier(job->source_id_column),
        quote_identifier(job->source_id_column),
        last_processed_id,
        quote_identifier(job->source_id_column),
        quote_identifier(job->target_column),
        quote_identifier(job->source_id_column),
        embedding_worker_batch_size);

    ret = SPI_execute(buf.data, true, 0);
    if (ret != SPI_OK_SELECT)
    {
        elog(WARNING, "failed to query source table for job %d: %s",
             job->job_id, SPI_result_code_string(ret));
        return;
    }

    if (SPI_processed == 0)
    {
        elog(LOG, "Job %d: No new rows to process.", job->job_id);
        return;
    }

    elog(LOG, "Job %d: Found %d new rows to process.", job->job_id, (int)SPI_processed);

    /* Prepare data for embedding generation */
    n_rows = SPI_processed;
    ids = (int *)palloc(sizeof(int) * n_rows);
    texts = (StringSlice *)palloc(sizeof(StringSlice) * n_rows);
    max_id = last_processed_id;

    for (i = 0; i < n_rows; i++)
    {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        text *t;
        Datum datum;

        /* Get ID with null check */
        datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
        if (isnull)
        {
            elog(WARNING, "Job %d: NULL id column at row %d, skipping", job->job_id, i);
            pfree(ids);
            pfree(texts);
            return;
        }
        ids[i] = DatumGetInt32(datum);
        if (ids[i] > max_id)
            max_id = ids[i];

        /* Get text with null check */
        datum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
        if (isnull)
        {
            elog(WARNING, "Job %d: NULL text column at row %d (id=%d), skipping",
                 job->job_id, i, ids[i]);
            pfree(ids);
            pfree(texts);
            return;
        }

        t = DatumGetTextPP(datum);  /* Use DatumGetTextPP for proper detoasting */
        texts[i].ptr = VARDATA_ANY(t);
        texts[i].len = VARSIZE_ANY_EXHDR(t);

        if (texts[i].len == 0)
        {
            elog(WARNING, "Job %d: Empty text at row %d (id=%d)",
                 job->job_id, i, ids[i]);
        }
    }

    /* Validate method and model */
    method_id = validate_embedding_method(job->method);
    if (method_id < 0)
    {
        elog(WARNING, "invalid method '%s' for job %d", job->method, job->job_id);
        pfree(ids);
        pfree(texts);
        return;
    }

    model_id = validate_embedding_model(method_id, job->model);
    if (model_id < 0)
    {
        elog(WARNING, "invalid model '%s' for job %d", job->model, job->job_id);
        pfree(ids);
        pfree(texts);
        return;
    }

    elog(DEBUG1, "Job %d: Generating embeddings for %d texts using %s with model %s.",
         job->job_id, n_rows, job->method, job->model);

    err = generate_embeddings_from_texts(method_id, model_id, texts, n_rows, &batch);

    pfree(texts);
    texts = NULL;

    if (err != 0)
    {
        elog(WARNING, "embedding generation failed for job %d (code=%d)", job->job_id, err);
        pfree(ids);
        return;
    }

    /* Validate batch results */
    if (batch.n_vectors == 0 || batch.dim == 0 || batch.data == NULL)
    {
        elog(WARNING, "Job %d: Invalid batch result (n_vectors=%zu, dim=%zu, data=%p)",
             job->job_id, batch.n_vectors, batch.dim, batch.data);
        pfree(ids);
        return;
    }

    elog(DEBUG1, "Job %d: Successfully generated %zu embeddings with dimension %zu.",
         job->job_id, batch.n_vectors, batch.dim);

    /* Store embeddings in target table */
    for (i = 0; i < (int)batch.n_vectors && i < n_rows; i++)
    {
        StringInfoData upsert_buf;
        StringInfoData vec_str;
        size_t j;

        initStringInfo(&upsert_buf);
        initStringInfo(&vec_str);

        /* Build vector literal - pre-allocate enough space */
        appendStringInfoChar(&vec_str, '[');
        for (j = 0; j < batch.dim; j++)
        {
            if (j > 0)
                appendStringInfoChar(&vec_str, ',');
            appendStringInfo(&vec_str, "%.9g", batch.data[i * batch.dim + j]);
        }
        appendStringInfoChar(&vec_str, ']');

        /* Use UPDATE...WHERE or INSERT pattern that works without unique constraint */
        appendStringInfo(&upsert_buf,
            "UPDATE %s.%s SET %s = %s::vector WHERE %s = %d",
            quote_identifier(job->target_schema),
            quote_identifier(job->target_table),
            quote_identifier(job->target_column),
            quote_literal_cstr(vec_str.data),
            quote_identifier(job->source_id_column),
            ids[i]);

        elog(DEBUG2, "Job %d: Updating embedding for source ID %d.", job->job_id, ids[i]);

        ret = SPI_execute(upsert_buf.data, false, 0);

        /* If no rows updated, insert instead */
        if (ret == SPI_OK_UPDATE && SPI_processed == 0)
        {
            resetStringInfo(&upsert_buf);

            appendStringInfo(&upsert_buf,
                "INSERT INTO %s.%s (%s, %s) VALUES (%d, %s::vector)",
                quote_identifier(job->target_schema),
                quote_identifier(job->target_table),
                quote_identifier(job->source_id_column),
                quote_identifier(job->target_column),
                ids[i],
                quote_literal_cstr(vec_str.data));

            elog(DEBUG2, "Job %d: Inserting new embedding for source ID %d.", job->job_id, ids[i]);

            ret = SPI_execute(upsert_buf.data, false, 0);
            if (ret != SPI_OK_INSERT)
            {
                elog(WARNING, "failed to insert embedding for id %d in job %d: %s",
                     ids[i], job->job_id, SPI_result_code_string(ret));
            }
        }
        else if (ret != SPI_OK_UPDATE)
        {
            elog(WARNING, "failed to update embedding for id %d in job %d: %s",
                 ids[i], job->job_id, SPI_result_code_string(ret));
        }

        /* StringInfo data is automatically freed when memory context resets */
    }

    elog(DEBUG1, "Job %d: Finished inserting/updating %zu embeddings.", job->job_id, batch.n_vectors);

    free_embedding_batch(&batch);
    pfree(ids);

    /* Update last processed ID */
    resetStringInfo(&buf);
    appendStringInfo(&buf,
        "UPDATE gem_jobs.embedding_jobs "
        "SET last_processed_id = %d, last_run_at = CURRENT_TIMESTAMP "
        "WHERE job_id = %d",
        max_id, job->job_id);

    ret = SPI_execute(buf.data, false, 0);
    if (ret != SPI_OK_UPDATE)
        elog(WARNING, "failed to update last_processed_id for job %d", job->job_id);

    elog(DEBUG1, "Job %d: Updated last_processed_id to %d.", job->job_id, max_id);

    elog(LOG, "embedding_worker: processed %d rows for job %d", n_rows, job->job_id);
}

/*
 * Main loop for the embedding worker
 */
void
embedding_worker_main(Datum _)
{
    List *jobs;
    ListCell *lc;

    /* Establish signal handlers */
    pqsignal(SIGHUP, SignalHandlerForConfigReload);
    pqsignal(SIGTERM, SignalHandlerForShutdownRequest);

    BackgroundWorkerUnblockSignals();
    /* Connect to database */
    BackgroundWorkerInitializeConnection("joeldiaz", NULL, 0);

    elog(LOG, "embedding_worker started with pid %d", MyProcPid);

    /* Main loop */
    for (;;)
    {
        elog(DEBUG1, "Worker main loop started. Naptime: %d seconds.", embedding_worker_naptime);

        if (embedding_worker_wait_event_main == 0)
            embedding_worker_wait_event_main = WaitEventExtensionNew("EmbeddingWorkerMain");

        (void) WaitLatch(MyLatch,
                        WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                        embedding_worker_naptime * 1000L,
                        embedding_worker_wait_event_main);
        ResetLatch(MyLatch);

        CHECK_FOR_INTERRUPTS();

        if (ConfigReloadPending)
        {
            elog(LOG, "Configuration reload requested (SIGHUP).");
            ConfigReloadPending = false;
            ProcessConfigFile(PGC_SIGHUP);
        }

        /* Start transaction and process jobs - with error handling */
        elog(LOG, "Worker waking up to check for jobs.");

        PG_TRY();
        {
            SetCurrentStatementStartTimestamp();
            StartTransactionCommand();
            SPI_connect();
            PushActiveSnapshot(GetTransactionSnapshot());
            pgstat_report_activity(STATE_RUNNING, "processing embedding jobs");

            jobs = load_embedding_jobs();

            if (list_length(jobs) == 0)
            {
                elog(LOG, "No active jobs found. Going back to sleep.");
            }
            else
            {
                foreach(lc, jobs)
                {
                    EmbeddingJob *job = (EmbeddingJob *)lfirst(lc);

                    /* Process each job with individual error handling */
                    PG_TRY();
                    {
                        process_embedding_job(job);
                    }
                    PG_CATCH();
                    {
                        ErrorData *edata;

                        /* Save error info */
                        edata = CopyErrorData();
                        FlushErrorState();

                        /* Log the error but continue with other jobs */
                        elog(WARNING, "Error processing job %d: %s",
                             job->job_id, edata->message);

                        FreeErrorData(edata);
                    }
                    PG_END_TRY();

                    /* Add a small interrupt check between jobs */
                    CHECK_FOR_INTERRUPTS();
                }
            }

            SPI_finish();
            PopActiveSnapshot();
            CommitTransactionCommand();
            pgstat_report_stat(true);
            pgstat_report_activity(STATE_IDLE, NULL);
            elog(LOG, "Finished job processing cycle. Sleeping for %d seconds.", embedding_worker_naptime);
        }
        PG_CATCH();
        {
            ErrorData *edata;

            /* Save error info */
            edata = CopyErrorData();
            FlushErrorState();

            /* Log the error */
            elog(WARNING, "Error in worker main loop: %s", edata->message);

            FreeErrorData(edata);

            /* Abort the transaction if still in progress */
            PG_TRY();
            {
                AbortCurrentTransaction();
            }
            PG_CATCH();
            {
                /* Ignore errors during abort */
                FlushErrorState();
            }
            PG_END_TRY();

            /* Clean up SPI */
            SPI_finish();

            pgstat_report_activity(STATE_IDLE, NULL);
        }
        PG_END_TRY();
    }
}

/*
 * Module initialization
 */
void
_PG_init(void)
{
    BackgroundWorker worker;

    DefineCustomIntVariable("pg_gem.embedding_worker_naptime",
                           "Duration between each check (in seconds).",
                           NULL,
                           &embedding_worker_naptime,
                           10,
                           1,
                           INT_MAX,
                           PGC_SIGHUP,
                           0,
                           NULL, NULL, NULL);

    DefineCustomIntVariable("pg_gem.embedding_worker_batch_size",
                           "Number of rows to process per batch.",
                           NULL,
                           &embedding_worker_batch_size,
                           256,
                           1,
                           10000,
                           PGC_SIGHUP,
                           0,
                           NULL, NULL, NULL);

    if (!process_shared_preload_libraries_in_progress)
    {
        elog(DEBUG1, "Skipping background worker registration; not in shared_preload_libraries context.");
        return;
    }

    MarkGUCPrefixReserved("pg_gem");

    memset(&worker, 0, sizeof(worker));
    worker.bgw_flags = BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
    worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
    worker.bgw_restart_time = BGW_DEFAULT_RESTART_INTERVAL;
    sprintf(worker.bgw_library_name, "pg_gem");
    sprintf(worker.bgw_function_name, "embedding_worker_main");
    snprintf(worker.bgw_name, BGW_MAXLEN, "pg_gem embedding worker");
    snprintf(worker.bgw_type, BGW_MAXLEN, "pg_gem_embedding_worker");
    worker.bgw_notify_pid = 0;

    RegisterBackgroundWorker(&worker);
    elog(LOG, "pg_gem background worker registered.");
}
