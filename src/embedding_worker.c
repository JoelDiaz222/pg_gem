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
#include "pg_gembed.h"

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

/* GUC variable declarations */
static int embedding_worker_naptime = 1000;
static int embedding_worker_batch_size = 100;

/* Wait event identifier cached from shared memory */
static uint32 embedding_worker_wait_event_main = 0;

/* -------------------------------------------------------------------------
 * Helper Functions
 * -------------------------------------------------------------------------
 */

/*
 * Parse a single job row from the jobs table
 */
static EmbeddingJob *
parse_job_tuple(HeapTuple tuple, TupleDesc tupdesc)
{
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
    job->embedder = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 9, &isnull));
    job->model = TextDatumGetCString(SPI_getbinval(tuple, tupdesc, 10, &isnull));

    elog(DEBUG1, "Loaded job ID %d (%s.%s -> %s.%s)",
         job->job_id, job->source_schema, job->source_table,
         job->target_schema, job->target_table);

    return job;
}

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
        "       target_column, embedder, model "
        "FROM gembed.embedding_jobs "
        "WHERE enabled = true");

    ret = SPI_execute(buf.data, true, 0);
    if (ret != SPI_OK_SELECT)
        elog(ERROR, "failed to load jobs");

    elog(LOG, "Found %lld active embedding jobs.", SPI_processed);

    for (uint64 i = 0; i < SPI_processed; i++)
    {
        EmbeddingJob *job = parse_job_tuple(SPI_tuptable->vals[i],
                                            SPI_tuptable->tupdesc);
        jobs = lappend(jobs, job);
    }

    return jobs;
}

/*
 * Get the last processed ID for a job
 */
static int
get_last_processed_id(int job_id)
{
    int ret;
    StringInfoData buf;
    int last_processed_id = 0;
    bool isnull;

    initStringInfo(&buf);
    appendStringInfo(&buf,
        "SELECT last_processed_id FROM gembed.embedding_jobs WHERE job_id = %d",
        job_id);

    ret = SPI_execute(buf.data, true, 0);
    if (ret == SPI_OK_SELECT && SPI_processed > 0)
    {
        Datum datum = SPI_getbinval(SPI_tuptable->vals[0],
                                    SPI_tuptable->tupdesc, 1, &isnull);
        if (!isnull)
            last_processed_id = DatumGetInt32(datum);
    }

    elog(DEBUG1, "Job %d: last_processed_id is %d", job_id, last_processed_id);
    return last_processed_id;
}

/*
 * Build query to find rows needing embeddings
 */
static void
build_pending_rows_query(StringInfo buf, EmbeddingJob *job, int last_processed_id)
{
    appendStringInfo(buf,
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
}

/*
 * Extract IDs and texts from query results
 * Returns the maximum ID found, or -1 on error
 */
static int
extract_ids_and_texts(int job_id, int n_rows, int **ids_out,
                      StringSlice **texts_out, int last_processed_id)
{
    int *ids = (int *)palloc(sizeof(int) * n_rows);
    StringSlice *texts = (StringSlice *)palloc(sizeof(StringSlice) * n_rows);
    int max_id = last_processed_id;
    int i;

    for (i = 0; i < n_rows; i++)
    {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        text *t;
        Datum datum;
        bool isnull;

        /* Get ID with null check */
        datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
        if (isnull)
        {
            elog(WARNING, "Job %d: NULL id column at row %d, skipping", job_id, i);
            pfree(ids);
            pfree(texts);
            return -1;
        }
        ids[i] = DatumGetInt32(datum);
        if (ids[i] > max_id)
            max_id = ids[i];

        /* Get text with null check */
        datum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
        if (isnull)
        {
            elog(WARNING, "Job %d: NULL text column at row %d (id=%d), skipping",
                 job_id, i, ids[i]);
            pfree(ids);
            pfree(texts);
            return -1;
        }

        t = DatumGetTextPP(datum);
        texts[i].ptr = VARDATA_ANY(t);
        texts[i].len = VARSIZE_ANY_EXHDR(t);

        if (texts[i].len == 0)
        {
            elog(WARNING, "Job %d: Empty text at row %d (id=%d)",
                 job_id, i, ids[i]);
        }
    }

    *ids_out = ids;
    *texts_out = texts;
    return max_id;
}

/*
 * Validate embedder and model for a job
 * Returns true if valid, false otherwise
 */
static bool
validate_job_embedder_and_model(EmbeddingJob *job, int *embedder_id_out,
                               int *model_id_out)
{
    int embedder_id, model_id;
    embedder_id = validate_embedder(job->embedder);
    if (embedder_id < 0)
    {
        elog(WARNING, "Invalid embedder '%s' for job %d", job->embedder, job->job_id);
        return false;
    }

    model_id = validate_embedding_model(embedder_id, job->model, INPUT_TYPE_TEXT);
    if (model_id < 0)
    {
        elog(WARNING, "Invalid model '%s' for job %d", job->model, job->job_id);
        return false;
    }

    *embedder_id_out = embedder_id;
    *model_id_out = model_id;
    return true;
}

/*
 * Build a vector literal string from embedding data
 */
static void
build_vector_literal(StringInfo vec_str, const EmbeddingBatch *batch, int idx)
{
    size_t j;

    appendStringInfoChar(vec_str, '[');
    for (j = 0; j < batch->dim; j++)
    {
        if (j > 0)
            appendStringInfoChar(vec_str, ',');
        appendStringInfo(vec_str, "%.9g", batch->data[idx * batch->dim + j]);
    }
    appendStringInfoChar(vec_str, ']');
}

/*
 * Update or insert a single embedding
 */
static void
upsert_embedding(EmbeddingJob *job, int id, const char *vec_literal)
{
    StringInfoData buf;
    int ret;

    initStringInfo(&buf);

    /* Try UPDATE first */
    appendStringInfo(&buf,
        "UPDATE %s.%s SET %s = %s::vector WHERE %s = %d",
        quote_identifier(job->target_schema),
        quote_identifier(job->target_table),
        quote_identifier(job->target_column),
        quote_literal_cstr(vec_literal),
        quote_identifier(job->source_id_column),
        id);

    elog(DEBUG2, "Job %d: Updating embedding for source ID %d.", job->job_id, id);

    ret = SPI_execute(buf.data, false, 0);

    /* If no rows updated, insert instead */
    if (ret == SPI_OK_UPDATE && SPI_processed == 0)
    {
        resetStringInfo(&buf);

        appendStringInfo(&buf,
            "INSERT INTO %s.%s (%s, %s) VALUES (%d, %s::vector)",
            quote_identifier(job->target_schema),
            quote_identifier(job->target_table),
            quote_identifier(job->source_id_column),
            quote_identifier(job->target_column),
            id,
            quote_literal_cstr(vec_literal));

        elog(DEBUG2, "Job %d: Inserting new embedding for source ID %d.",
             job->job_id, id);

        ret = SPI_execute(buf.data, false, 0);
        if (ret != SPI_OK_INSERT)
        {
            elog(WARNING, "failed to insert embedding for id %d in job %d: %s",
                 id, job->job_id, SPI_result_code_string(ret));
        }
    }
    else if (ret != SPI_OK_UPDATE)
    {
        elog(WARNING, "failed to update embedding for id %d in job %d: %s",
             id, job->job_id, SPI_result_code_string(ret));
    }
}

/*
 * Store all embeddings from a batch
 */
static void
store_embeddings(EmbeddingJob *job, const EmbeddingBatch *batch,
                 const int *ids, int n_rows)
{
    int i;

    for (i = 0; i < (int)batch->n_vectors && i < n_rows; i++)
    {
        StringInfoData vec_str;

        initStringInfo(&vec_str);
        build_vector_literal(&vec_str, batch, i);
        upsert_embedding(job, ids[i], vec_str.data);
    }

    elog(DEBUG1, "Job %d: Finished inserting/updating %zu embeddings.",
         job->job_id, batch->n_vectors);
}

/*
 * Update the last processed ID for a job
 */
static void
update_last_processed_id(int job_id, int max_id)
{
    StringInfoData buf;
    int ret;

    initStringInfo(&buf);
    appendStringInfo(&buf,
        "UPDATE gembed.embedding_jobs "
        "SET last_processed_id = %d, last_run_at = CURRENT_TIMESTAMP "
        "WHERE job_id = %d",
        max_id, job_id);

    ret = SPI_execute(buf.data, false, 0);
    if (ret != SPI_OK_UPDATE)
        elog(WARNING, "failed to update last_processed_id for job %d", job_id);

    elog(DEBUG1, "Job %d: Updated last_processed_id to %d.", job_id, max_id);
}

/* -------------------------------------------------------------------------
 * Main Job Processing
 * -------------------------------------------------------------------------
 */

/*
 * Process a specific embedding job
 */
static void
process_embedding_job(EmbeddingJob *job)
{
    int ret;
    StringInfoData buf;
    int last_processed_id;
    int n_rows;
    int *ids = NULL;
    StringSlice *texts = NULL;
    int max_id;
    int embedder_id, model_id;
    EmbeddingBatch batch;
    int err;

    elog(LOG, "Starting to process job ID: %d (%s.%s.%s -> %s.%s.%s)",
         job->job_id, job->source_schema, job->source_table, job->source_column,
         job->target_schema, job->target_table, job->target_column);

    /* Get last processed ID */
    last_processed_id = get_last_processed_id(job->job_id);

    /* Find rows needing embeddings */
    initStringInfo(&buf);
    build_pending_rows_query(&buf, job, last_processed_id);

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

    elog(LOG, "Job %d: Found %d new rows to process.",
         job->job_id, (int)SPI_processed);

    /* Extract data from results */
    n_rows = SPI_processed;
    max_id = extract_ids_and_texts(job->job_id, n_rows, &ids, &texts,
                                   last_processed_id);
    if (max_id < 0)
        return;

    /* Validate embedder and model */
    if (!validate_job_embedder_and_model(job, &embedder_id, &model_id))
    {
        pfree(ids);
        pfree(texts);
        return;
    }

    elog(DEBUG1, "Job %d: Generating embeddings for %d texts using %s with model %s.",
         job->job_id, n_rows, job->embedder, job->model);

    /* Generate embeddings */
    InputData input_data = {
        .input_type = INPUT_TYPE_TEXT,
        .binary_data = NULL,
        .n_binary = 0,
        .text_data = texts,
        .n_text = n_rows
    };

    err = generate_embeddings(embedder_id, model_id, &input_data, &batch);
    pfree(texts);

    if (err != 0)
    {
        elog(WARNING, "embedding generation failed for job %d (code=%d)",
             job->job_id, err);
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

    /* Store embeddings */
    store_embeddings(job, &batch, ids, n_rows);
    free_embedding_batch(&batch);
    pfree(ids);

    /* Update last processed ID */
    update_last_processed_id(job->job_id, max_id);

    elog(LOG, "embedding_worker: processed %d rows for job %d", n_rows, job->job_id);
}

/* -------------------------------------------------------------------------
 * Worker Main Loop and Initialization
 * -------------------------------------------------------------------------
 */

/*
 * Setup signal handlers and connect to database
 */
static void
initialize_worker(void)
{
    pqsignal(SIGHUP, SignalHandlerForConfigReload);
    pqsignal(SIGTERM, SignalHandlerForShutdownRequest);
    BackgroundWorkerUnblockSignals();
    BackgroundWorkerInitializeConnection("joeldiaz", NULL, 0);
    elog(LOG, "embedding_worker started with pid %d", MyProcPid);
}

/*
 * Wait for next cycle or shutdown signal
 */
static void
worker_wait_for_next_cycle(void)
{
    if (embedding_worker_wait_event_main == 0)
        embedding_worker_wait_event_main = WaitEventExtensionNew("EmbeddingWorkerMain");

    (void) WaitLatch(MyLatch,
                    WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                    embedding_worker_naptime * 1000L,
                    embedding_worker_wait_event_main);
    ResetLatch(MyLatch);
}

/*
 * Handle configuration reload request
 */
static void
handle_config_reload(void)
{
    if (ConfigReloadPending)
    {
        elog(LOG, "Configuration reload requested (SIGHUP).");
        ConfigReloadPending = false;
        ProcessConfigFile(PGC_SIGHUP);
    }
}

/*
 * Process all jobs with individual error handling
 */
static void
process_all_jobs(List *jobs)
{
    ListCell *lc;

    foreach(lc, jobs)
    {
        EmbeddingJob *job = (EmbeddingJob *)lfirst(lc);

        PG_TRY();
        {
            process_embedding_job(job);
        }
        PG_CATCH();
        {
            ErrorData *edata = CopyErrorData();
            FlushErrorState();
            elog(WARNING, "Error processing job %d: %s",
                 job->job_id, edata->message);
            FreeErrorData(edata);
        }
        PG_END_TRY();

        CHECK_FOR_INTERRUPTS();
    }
}

/*
 * Execute one cycle of job processing
 */
static void
execute_job_cycle(void)
{
    List *jobs;

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
        process_all_jobs(jobs);
    }

    SPI_finish();
    PopActiveSnapshot();
    CommitTransactionCommand();
    pgstat_report_stat(true);
    pgstat_report_activity(STATE_IDLE, NULL);
    elog(LOG, "Finished job processing cycle. Sleeping for %d seconds.",
         embedding_worker_naptime);
}

/*
 * Handle errors in the main loop
 */
static void
handle_main_loop_error(void)
{
    ErrorData *edata = CopyErrorData();
    FlushErrorState();
    elog(WARNING, "Error in worker main loop: %s", edata->message);
    FreeErrorData(edata);

    /* Abort the transaction if still in progress */
    PG_TRY();
    {
        AbortCurrentTransaction();
    }
    PG_CATCH();
    {
        FlushErrorState();
    }
    PG_END_TRY();

    SPI_finish();
    pgstat_report_activity(STATE_IDLE, NULL);
}

/*
 * Main loop for the embedding worker
 */
void
embedding_worker_main(Datum _)
{
    initialize_worker();

    /* Main loop */
    for (;;)
    {
        elog(DEBUG1, "Worker main loop started. Naptime: %d seconds.",
             embedding_worker_naptime);

        worker_wait_for_next_cycle();
        CHECK_FOR_INTERRUPTS();
        handle_config_reload();

        elog(LOG, "Worker waking up to check for jobs.");

        PG_TRY();
        {
            execute_job_cycle();
        }
        PG_CATCH();
        {
            handle_main_loop_error();
        }
        PG_END_TRY();
    }
}

/* -------------------------------------------------------------------------
 * Module Initialization
 * -------------------------------------------------------------------------
 */

/*
 * Define GUC configuration variables
 */
static void
define_guc_variables(void)
{
    MarkGUCPrefixReserved("gembed");
    DefineCustomIntVariable("gembed.embedding_worker_naptime",
                           "Duration between each check (in seconds).",
                           NULL,
                           &embedding_worker_naptime,
                           10,
                           1,
                           INT_MAX,
                           PGC_SIGHUP,
                           0,
                           NULL, NULL, NULL);

    DefineCustomIntVariable("gembed.embedding_worker_batch_size",
                           "Number of rows to process per batch.",
                           NULL,
                           &embedding_worker_batch_size,
                           256,
                           1,
                           10000,
                           PGC_SIGHUP,
                           0,
                           NULL, NULL, NULL);
}

/*
 * Register the background worker
 */
static void
register_background_worker(void)
{
    BackgroundWorker worker;

    memset(&worker, 0, sizeof(worker));
    worker.bgw_flags = BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
    worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
    worker.bgw_restart_time = BGW_DEFAULT_RESTART_INTERVAL;
    sprintf(worker.bgw_library_name, "pg_gembed");
    sprintf(worker.bgw_function_name, "embedding_worker_main");
    snprintf(worker.bgw_name, BGW_MAXLEN, "pg_gembed embedding worker");
    snprintf(worker.bgw_type, BGW_MAXLEN, "pg_gembed_embedding_worker");
    worker.bgw_notify_pid = 0;

    RegisterBackgroundWorker(&worker);
    elog(LOG, "pg_gembed background worker registered.");
}

/*
 * Module initialization
 */
void
_PG_init(void)
{
    define_guc_variables();

    if (!process_shared_preload_libraries_in_progress)
    {
        elog(DEBUG1, "Skipping background worker registration; not in shared_preload_libraries context.");
        return;
    }

    register_background_worker();
}
