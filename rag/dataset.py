"""Load VieQuADRetrieval corpus/queries/qrels (BEIR-style parquet) as plain Python dicts.

Inspect a query and its labeled documents:
    python -m rag.dataset            # dataset stats
    python -m rag.dataset --list     # first 20 query ids
    python -m rag.dataset 3          # query 3 + its relevant documents
"""

import sys
from pathlib import Path

import duckdb

from rag.config import DATA_DIR

_SPLIT_FILE = "validation-00000-of-00001.parquet"


def _read(data_dir: Path, subset: str, columns: str) -> list[tuple]:
    path = data_dir / subset / _SPLIT_FILE
    return duckdb.sql(f"SELECT {columns} FROM read_parquet(?)", params=[str(path)]).fetchall()


def load_corpus(data_dir: Path = DATA_DIR) -> dict[str, dict]:
    """Return {doc_id: {"title": str, "text": str}}."""
    rows = _read(data_dir, "corpus", '"_id", title, text')
    return {doc_id: {"title": title, "text": text} for doc_id, title, text in rows}


def load_queries(data_dir: Path = DATA_DIR) -> dict[str, str]:
    """Return {query_id: text}."""
    return dict(_read(data_dir, "queries", '"_id", text'))


def load_qrels(data_dir: Path = DATA_DIR) -> dict[str, dict[str, int]]:
    """Return {query_id: {doc_id: relevance}}."""
    qrels: dict[str, dict[str, int]] = {}
    for query_id, doc_id, score in _read(data_dir, "qrels", '"query-id", "corpus-id", score'):
        qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


def _main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    queries = load_queries()

    if not argv:
        qrels = load_qrels()
        print(f"{len(load_corpus())} documents, {len(queries)} queries, {len(qrels)} judged queries")
        return 0

    if argv[0] == "--list":
        for qid in sorted(queries, key=int)[:20]:
            print(f"{qid}: {queries[qid]}")
        return 0

    qid = argv[0]
    if qid not in queries:
        print(f"Không tìm thấy query ID {qid!r}. Chạy 'python -m rag.dataset --list' để xem ID hợp lệ.")
        return 1

    corpus = load_corpus()
    print(f"Query ID {qid}: {queries[qid]}\n")
    for doc_id, score in sorted(load_qrels().get(qid, {}).items(), key=lambda item: -item[1]):
        doc = corpus[doc_id]
        print(f"Document ID {doc_id} | relevance {score} | {doc['title']}\n{doc['text']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
