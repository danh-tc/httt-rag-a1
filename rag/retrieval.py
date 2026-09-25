"""Tier-1 retrieval over the `documents` table: BM25 (pg_search), dense (pgvector), and RRF fusion.

Every retriever returns [(doc_id, score), ...] sorted best first.
"""

from psycopg2 import sql

from rag.config import CANDIDATE_N, DEFAULT_EMBEDDER, DEFAULT_FUSION, EMBEDDERS, FUSIONS, RRF_K
from rag.embedding import encode_query

Ranking = list[tuple[str, float]]


def bm25_search(conn, query: str, k: int) -> Ranking:
    """BM25 over `body` via the pg_search index."""
    with conn.cursor() as cur:
        # paradedb.match tokenizes the raw query and ORs the terms, so free text with
        # punctuation ("?", ":") is safe, unlike the Tantivy query-string syntax.
        cur.execute(
            """
            SELECT id, paradedb.score(id) AS score
            FROM documents
            WHERE id @@@ paradedb.match('body', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, k),
        )
        return cur.fetchall()


def dense_search(conn, query: str, k: int, embedder: str = DEFAULT_EMBEDDER) -> Ranking:
    """Cosine similarity over `embedder`'s vector column via its pgvector HNSW index."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT id, 1 - ({col} <=> %(vec)s) AS score
                FROM documents
                ORDER BY {col} <=> %(vec)s
                LIMIT %(k)s
                """
            ).format(col=sql.Identifier(EMBEDDERS[embedder]["column"])),
            {"vec": encode_query(query, embedder), "k": k},
        )
        return cur.fetchall()


def rrf_fuse(rankings: list[Ranking], k: int, rrf_k: int = RRF_K) -> Ranking:
    """Reciprocal Rank Fusion: score = sum over rankings of 1 / (rrf_k + rank)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]


def hybrid_search(
    conn,
    query: str,
    k: int,
    candidate_n: int = CANDIDATE_N,
    fusion: str = DEFAULT_FUSION,
    embedder: str = DEFAULT_EMBEDDER,
) -> Ranking:
    """BM25 top-N + dense top-N, fused with RRF, truncated to k.

    fusion="backend" runs two queries and fuses in Python; fusion="sql" does it all in one query.
    """
    if fusion == "sql":
        return _hybrid_search_sql(conn, query, k, candidate_n, embedder)
    if fusion != "backend":
        raise ValueError(f"unknown fusion {fusion!r}, expected one of {FUSIONS}")
    return rrf_fuse(
        [bm25_search(conn, query, candidate_n), dense_search(conn, query, candidate_n, embedder)], k=k
    )


def _hybrid_search_sql(conn, query: str, k: int, candidate_n: int, embedder: str, rrf_k: int = RRF_K) -> Ranking:
    """Same ranking as the backend path: ties break by first appearance (BM25 list, then dense-only)."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                WITH bm25 AS (
                    SELECT id, row_number() OVER (ORDER BY score DESC) AS r
                    FROM (
                        SELECT id, paradedb.score(id) AS score
                        FROM documents
                        WHERE id @@@ paradedb.match('body', %(q)s)
                        ORDER BY score DESC
                        LIMIT %(n)s
                    ) t
                ),
                dense AS (
                    SELECT id, row_number() OVER (ORDER BY dist) AS r
                    FROM (
                        SELECT id, {col} <=> %(vec)s AS dist
                        FROM documents
                        ORDER BY dist
                        LIMIT %(n)s
                    ) t
                ),
                hits AS (
                    SELECT id, r, r AS first_seen FROM bm25
                    UNION ALL
                    SELECT id, r, %(n)s + r FROM dense
                )
                SELECT id, sum(1.0::float8 / (%(rrf_k)s + r)) AS score
                FROM hits
                GROUP BY id
                ORDER BY score DESC, min(first_seen)
                LIMIT %(k)s
                """
            ).format(col=sql.Identifier(EMBEDDERS[embedder]["column"])),
            {"q": query, "vec": encode_query(query, embedder), "n": candidate_n, "rrf_k": rrf_k, "k": k},
        )
        return cur.fetchall()


def fetch_texts(conn, doc_ids: list[str]) -> dict[str, dict]:
    """Return {doc_id: {"title": ..., "body": ...}} for the given ids."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, body FROM documents WHERE id = ANY(%s)", (doc_ids,))
        return {doc_id: {"title": title, "body": body} for doc_id, title, body in cur.fetchall()}
