CREATE EXTENSION IF NOT EXISTS vector;

-- 'simple' config: Postgres has no Vietnamese stemmer, so this just tokenizes
-- on whitespace/punctuation without stemming. Good enough for BM25-ish ranking.
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    title TEXT,
    body TEXT NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', body)) STORED,
    embedding vector(768)
);

CREATE INDEX documents_tsv_idx ON documents USING GIN (tsv);
CREATE INDEX documents_embedding_idx ON documents USING hnsw (embedding vector_cosine_ops);
