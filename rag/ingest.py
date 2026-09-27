"""Encode the VieQuADRetrieval corpus and (re)load it into Postgres.

    python -m rag.ingest                                   # reload all documents, encode with every embedder
    python -m rag.ingest --embedders e5-small minilm       # only (re)fill those vector columns

The BM25 index is maintained by Postgres; only the embeddings are computed here.
Encode time / model size per embedder goes to results/ingest_stats.json (for the trade-off table).
"""

import argparse
import json
import time

import torch
from psycopg2 import sql
from psycopg2.extras import execute_values

from rag.config import EMBEDDERS, RESULTS_DIR
from rag.dataset import load_corpus
from rag.db import get_conn
from rag.embedding import encode_passages, get_model

STATS_PATH = RESULTS_DIR / "ingest_stats.json"


def passage_text(doc: dict) -> str:
    title = (doc.get("title") or "").strip()
    return f"{title}. {doc['text']}" if title else doc["text"]


def load_documents(conn, corpus: dict):
    with conn, conn.cursor() as cur:
        cur.execute("TRUNCATE documents")
        execute_values(
            cur,
            "INSERT INTO documents (id, title, body) VALUES %s",
            [(doc_id, doc["title"] or None, doc["text"]) for doc_id, doc in corpus.items()],
        )


def fill_embeddings(conn, corpus: dict, embedder: str) -> dict:
    """Encode the corpus with `embedder` into its own vector column (created on first use)."""
    spec = EMBEDDERS[embedder]
    column = sql.Identifier(spec["column"])
    model = get_model(embedder)
    doc_ids = list(corpus)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    embeddings = encode_passages([passage_text(corpus[d]) for d in doc_ids], embedder, show_progress_bar=True)
    encode_s = time.perf_counter() - start

    with conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL("ALTER TABLE documents ADD COLUMN IF NOT EXISTS {} vector({})").format(column, sql.Literal(spec["dim"]))
        )
        execute_values(
            cur,
            sql.SQL("UPDATE documents d SET {} = v.emb FROM (VALUES %s) AS v(id, emb) WHERE d.id = v.id")
            .format(column)
            .as_string(cur),
            list(zip(doc_ids, embeddings)),
            template="(%s, %s::vector)",
        )
        # A parallel HNSW build asks for ~maintenance_work_mem of shared memory, more than Docker's
        # default 64 MB /dev/shm. 2.5k rows build fine on one worker.
        cur.execute("SET LOCAL max_parallel_maintenance_workers = 0")
        cur.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON documents USING hnsw ({} vector_cosine_ops)").format(
                sql.Identifier(f"documents_{spec['column']}_idx"), column
            )
        )
    return {
        "model": spec["model"],
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "dim": spec["dim"],
        "max_seq_length": model.max_seq_length,
        "encode_corpus_s": round(encode_s, 2),
        "n_docs": len(doc_ids),
        "device": str(model.device),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedders", nargs="+", choices=tuple(EMBEDDERS), default=None,
                        help="only (re)encode these, keeping the documents already in the DB")
    args = parser.parse_args()

    corpus = load_corpus()
    conn = get_conn(autocommit=False)  # TRUNCATE + INSERT, and each column fill, commit atomically
    if args.embedders is None:
        load_documents(conn, corpus)
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8")) if STATS_PATH.exists() else {}
    for embedder in args.embedders or EMBEDDERS:
        stats[embedder] = fill_embeddings(conn, corpus, embedder)
        print(f"{embedder}: {stats[embedder]}")
    conn.close()

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Ingested {len(corpus)} documents; stats in {STATS_PATH}")


if __name__ == "__main__":
    main()
