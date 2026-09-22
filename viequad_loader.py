"""Load VieQuADRetrieval corpus/queries/qrels as plain Python dicts for the search pipeline."""

from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).parent / "data" / "VieQuADRetrieval"


def load_corpus(data_dir: Path = DATA_DIR) -> dict[str, dict]:
    """Return {doc_id: {"title": str, "text": str}}."""
    path = data_dir / "corpus" / "validation-00000-of-00001.parquet"
    rows = duckdb.sql(
        'SELECT "_id", title, text FROM read_parquet(?)', params=[str(path)]
    ).fetchall()
    return {doc_id: {"title": title, "text": text} for doc_id, title, text in rows}


def load_queries(data_dir: Path = DATA_DIR) -> dict[str, str]:
    """Return {query_id: text}."""
    path = data_dir / "queries" / "validation-00000-of-00001.parquet"
    rows = duckdb.sql(
        'SELECT "_id", text FROM read_parquet(?)', params=[str(path)]
    ).fetchall()
    return {query_id: text for query_id, text in rows}


def load_qrels(data_dir: Path = DATA_DIR) -> dict[str, dict[str, int]]:
    """Return {query_id: {doc_id: score}}."""
    path = data_dir / "qrels" / "validation-00000-of-00001.parquet"
    rows = duckdb.sql(
        'SELECT "query-id", "corpus-id", score FROM read_parquet(?)', params=[str(path)]
    ).fetchall()
    qrels: dict[str, dict[str, int]] = {}
    for query_id, doc_id, score in rows:
        qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


if __name__ == "__main__":
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels()
    print(f"{len(corpus)} documents, {len(queries)} queries, {len(qrels)} query judgments")
