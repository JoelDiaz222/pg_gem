CREATE FUNCTION generate_embeddings(
    method text,
    texts text[]
)
RETURNS vector[]
AS
    'MODULE_PATHNAME',
    'generate_embeddings'
LANGUAGE C STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION generate_embeddings_with_ids(
    method text,
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
LANGUAGE C STRICT PARALLEL SAFE;
