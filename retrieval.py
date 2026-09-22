"""Tier-1 retrieval over the `documents` table: BM25 (tsvector), dense (pgvector), and RRF fusion."""

import re

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

DB_DSN = "postgresql://rag:rag@localhost:5432/rag"
MODEL_NAME = "intfloat/multilingual-e5-base"
RRF_K = 60  # standard RRF damping constant

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_conn():
    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)
    return conn


def _make_tsquery(query: str) -> str:
    # OR the query words together so ts_rank behaves like BM25 (rank by how many/rare
    # terms match) instead of AND-ing everything and returning nothing on long queries.
    words = re.findall(r"\w+", query.lower())
    return " | ".join(words) if words else query


def bm25_search(query: str, k: int, conn=None) -> list[tuple[str, float]]:
    """Return up to k (doc_id, ts_rank score) pairs, best first."""
    conn = conn or get_conn()
    tsquery = _make_tsquery(query)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, ts_rank(tsv, to_tsquery('simple', %s)) AS score
            FROM documents
            WHERE tsv @@ to_tsquery('simple', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (tsquery, tsquery, k),
        )
        return cur.fetchall()


def dense_search(query: str, k: int, conn=None) -> list[tuple[str, float]]:
    """Return up to k (doc_id, cosine similarity) pairs, best first."""
    conn = conn or get_conn()
    model = get_model()
    query_vec = model.encode(f"query: {query}", normalize_embeddings=True)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, 1 - (embedding <=> %s) AS score
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_vec, query_vec, k),
        )
        return cur.fetchall()


def rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]], k: int, rrf_k: int = RRF_K
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over several (doc_id, score) rankings; higher fused score is better."""
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]


def hybrid_search(query: str, k: int, candidate_n: int = 50, conn=None) -> list[tuple[str, float]]:
    """BM25 top-N + dense top-N, fused with RRF, truncated to k."""
    conn = conn or get_conn()
    bm25_results = bm25_search(query, candidate_n, conn=conn)
    dense_results = dense_search(query, candidate_n, conn=conn)
    return rrf_fuse([bm25_results, dense_results], k=k)


def fetch_texts(doc_ids: list[str], conn=None) -> dict[str, dict]:
    conn = conn or get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, title, body FROM documents WHERE id = ANY(%s)", (doc_ids,)
        )
        return {row[0]: {"title": row[1], "body": row[2]} for row in cur.fetchall()}


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Đại học nào lâu đời nhất ở Việt Nam?"
    conn = get_conn()

    print(f"Query: {query}\n")
    for name, results in [
        ("BM25", bm25_search(query, 5, conn=conn)),
        ("Dense", dense_search(query, 5, conn=conn)),
        ("Hybrid (RRF)", hybrid_search(query, 5, conn=conn)),
    ]:
        print(f"--- {name} ---")
        texts = fetch_texts([doc_id for doc_id, _ in results], conn=conn)
        for doc_id, score in results:
            snippet = texts[doc_id]["body"][:120].replace("\n", " ")
            print(f"  {doc_id} ({score:.4f}) {texts[doc_id]['title']!r} {snippet}...")
        print()
