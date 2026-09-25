"""Postgres connection helper (ParadeDB image: pg_search for BM25, pgvector for dense)."""

import psycopg2
from pgvector.psycopg2 import register_vector

from rag.config import DB_DSN, HNSW_EF_SEARCH


def get_conn(dsn: str = DB_DSN, autocommit: bool = True):
    """Search/eval connections autocommit: otherwise every SELECT leaves the session "idle in
    transaction", holding a table lock that blocks ingest / schema changes while the API runs.
    Pass autocommit=False for writes that must be atomic (ingest)."""
    conn = psycopg2.connect(dsn)
    conn.autocommit = autocommit
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = %s", (HNSW_EF_SEARCH,))
    if not autocommit:
        conn.commit()  # keep the session setting even if the caller later rolls back
    return conn
