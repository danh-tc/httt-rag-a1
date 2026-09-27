CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    title TEXT,
    body TEXT NOT NULL,
    -- One column per embedder in rag/config.py EMBEDDERS (rag.ingest also adds any missing one).
    embedding vector(768),           -- e5-base (default)
    embedding_e5_small vector(384),  -- e5-small (ablation)
    embedding_bge_m3 vector(1024),   -- bge-m3 (ablation)
    embedding_minilm vector(384)     -- all-MiniLM-L6-v2 (ablation)
);

CREATE INDEX documents_embedding_idx ON documents USING hnsw (embedding vector_cosine_ops);
CREATE INDEX documents_embedding_e5_small_idx ON documents USING hnsw (embedding_e5_small vector_cosine_ops);
CREATE INDEX documents_embedding_bge_m3_idx ON documents USING hnsw (embedding_bge_m3 vector_cosine_ops);
CREATE INDEX documents_embedding_minilm_idx ON documents USING hnsw (embedding_minilm vector_cosine_ops);
-- BM25 (Tantivy) over body; default tokenizer lowercases and splits on non-word chars,
-- which for Vietnamese means one token per syllable.
CREATE INDEX documents_bm25_idx ON documents USING bm25 (id, body) WITH (key_field = 'id');
