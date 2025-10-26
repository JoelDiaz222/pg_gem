-- Core embedding generation functions
CREATE FUNCTION generate_embeddings(
    method text,
    model text,
    texts text[]
)
RETURNS vector[]
AS
'MODULE_PATHNAME',
'generate_embeddings'
LANGUAGE C
STRICT
PARALLEL SAFE;

CREATE OR REPLACE FUNCTION generate_embeddings_with_ids(
    method text,
    model text,
    ids integer[],
    texts text[]
)
RETURNS TABLE (
    sentence_id integer,
    embedding   vector
)
AS
'MODULE_PATHNAME',
'generate_embeddings_with_ids'
LANGUAGE C
STRICT
PARALLEL SAFE;

-- Background worker schema and tables
CREATE SCHEMA IF NOT EXISTS gem_jobs;

CREATE TABLE gem_jobs.embedding_jobs (
    job_id            SERIAL    PRIMARY KEY,
    source_schema     TEXT      DEFAULT 'public',
    source_table      TEXT      NOT NULL,
    source_column     TEXT      NOT NULL,
    source_id_column  TEXT      NOT NULL,
    target_schema     TEXT      DEFAULT 'public',
    target_table      TEXT      NOT NULL,
    target_column     TEXT      NOT NULL,
    method            TEXT      NOT NULL,
    model             TEXT      NOT NULL,
    enabled           BOOLEAN   DEFAULT true,
    last_processed_id INTEGER   DEFAULT 0,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_run_at       TIMESTAMP
);

CREATE INDEX idx_jobs_enabled
    ON gem_jobs.embedding_jobs (enabled)
    WHERE enabled = true;

CREATE FUNCTION embedding_worker_launch()
RETURNS integer
AS
'MODULE_PATHNAME',
'embedding_worker_launch'
LANGUAGE C;

-- View for job status
CREATE VIEW gem_jobs.job_status AS
SELECT
    j.job_id,
    j.source_schema || '.' || j.source_table || '.' || j.source_column AS source,
    j.target_schema || '.' || j.target_table || '.' || j.target_column AS target,
    j.method,
    j.model,
    j.enabled,
    j.last_processed_id,
    j.last_run_at,
    j.created_at,
    CASE
        WHEN j.last_run_at IS NULL THEN 'never run'
        WHEN j.last_run_at < NOW() - INTERVAL '1 hour' THEN 'stale'
        WHEN j.enabled THEN 'active'
        ELSE 'disabled'
    END AS status
FROM gem_jobs.embedding_jobs j;
