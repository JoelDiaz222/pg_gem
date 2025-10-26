# pg_gem

## Generate Embeddings directly in PostgreSQL

A PostgreSQL extension that brings ML-powered vector embedding generation directly into your database. Supports
local embedding generation using FastEmbed-rs, or through a local gRPC server.

## Features

- 🚀 **Self-contained**: Generate embeddings without external API calls
- ⚡ **Fast**: Rust-powered inference with thread-local model caching
- 🔒 **Private**: Your data is not sent to external inference providers
- 💰 **Cost-effective**: No per-token API fees, predictable infrastructure costs
- 🎯 **Simple**: Just SQL functions, no orchestration required
- 🔄 **Flexible**: Support for both local (FastEmbed) and remote (gRPC) inference

## Installation

### Prerequisites

- PostgreSQL 17+
- [pgvector](https://github.com/pgvector/pgvector) extension
- Rust toolchain (for building)

### Build from Source

```bash
git clone https://github.com/JoelDiaz222/pg_gem
cd pg_gem

make install
```

### Enable in PostgreSQL

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pg_gem;
```

## Usage

### Basic Embedding Generation

```sql
-- Generate embeddings using local FastEmbed model
SELECT generate_embeddings(
    'fastembed',
    'Qdrant/all-MiniLM-L6-v2-onnx',
    ARRAY ['Hello world', 'Embedding in PostgreSQL']
);
```

Returns an array of `vector` types compatible with pgvector.

### Bulk Insert with IDs

```sql
-- Generate and insert embeddings with associated IDs
INSERT INTO documents (id, embedding)
SELECT sentence_id, embedding
FROM generate_embeddings_with_ids(
    'fastembed',
    'Qdrant/all-MiniLM-L6-v2-onnx',
    ARRAY [1, 2, 3],
    ARRAY ['First document', 'Second document', 'Third document']
);
```

### Semantic Search Example

```sql
-- Create a table with embeddings
CREATE TABLE articles
(
    id        SERIAL PRIMARY KEY,
    title     TEXT,
    content   TEXT,
    embedding vector(384)
);

-- Generate embeddings during insert
INSERT INTO articles (title, content, embedding)
SELECT title,
       content,
       (generate_embeddings('fastembed', 'Qdrant/all-MiniLM-L6-v2-onnx', ARRAY [content]))[1]
FROM (VALUES ('Understanding Transformers', 'Transformers have revolutionized NLP by using attention mechanisms.'),
             ('Graph Neural Networks', 'GNNs operate on graph structures to capture relationships.'),
             ('Reinforcement Learning Basics',
              'An introduction to RL concepts like agents and environments.')) AS t(title, content);

-- Perform semantic search
SELECT id,
       title,
       content,
       embedding <=> (SELECT (generate_embeddings('fastembed', 'Qdrant/all-MiniLM-L6-v2-onnx',
                                                  ARRAY ['machine learning']))[1]) AS distance
FROM articles
ORDER BY distance
LIMIT 10;
```

## Architecture

```
┌────────────────────────────────────────────┐
│             PostgreSQL Query               │
│   (e.g. SELECT generate_embeddings(...))   │
└──────────────────────┬─────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────┐
│        PostgreSQL C Extension (FFI)       │
│  - Defined via CREATE FUNCTION ...        │
│  - Calls into extern "C" Rust functions   │
└──────────────────────┬────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│             Rust Core Library                │
│   ┌──────────────────────┐  ┌────────────┐   │
│   │   fastembed-rs       │  │ gRPC Client│   │
│   │ (local embedding)    │  │ (remote)   │   │
│   └──────────────────────┘  └────────────┘   │
└──────────────────────────────────────────────┘
```

**Key design decisions**:

- **Thread-local model caching**: Models loaded once per connection
- **Zero-copy FFI**: Direct memory transfer between Rust and PostgreSQL
- **Flat memory layout**: Contiguous vector storage for optimal cache performance

## License

Licensed under the [Apache License 2.0](./LICENSE).

## Acknowledgments

- [pgvector](https://github.com/pgvector/pgvector) for the vector datatype
- [FastEmbed-rs](https://github.com/Anush008/fastembed-rs) for the embedding library
    - [A fork](https://github.com/JoelDiaz222/fastembed-rs) has been done to support returning a contiguous buffer of
      embeddings
- [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) for gRPC protocol reference
    - [A fork](https://github.com/JoelDiaz222/text-embeddings-inference) has been done for generating batches of
      embeddings using gRPC
