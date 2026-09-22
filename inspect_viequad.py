"""Show a VieQuADRetrieval query and its labeled relevant documents.

Usage: python inspect_viequad.py --list
       python inspect_viequad.py 3
"""

import sys
from pathlib import Path

import duckdb

sys.stdout.reconfigure(encoding="utf-8")


root = Path(__file__).parent / "data" / "VieQuADRetrieval"
qid = sys.argv[1] if len(sys.argv) > 1 else "0"
queries = str(root / "queries" / "validation-00000-of-00001.parquet")
qrels = str(root / "qrels" / "validation-00000-of-00001.parquet")
corpus = str(root / "corpus" / "validation-00000-of-00001.parquet")

if qid == "--list":
    examples = duckdb.sql(
        'SELECT "_id", text FROM read_parquet(?) ORDER BY TRY_CAST("_id" AS INTEGER) LIMIT 20',
        params=[queries],
    ).fetchall()
    for query_id, question in examples:
        print(f"{query_id}: {question}")
    raise SystemExit(0)

rows = duckdb.sql(
    """
    SELECT q."_id", q.text, r."corpus-id", r.score, d.title, d.text
    FROM read_parquet(?) AS q
    JOIN read_parquet(?) AS r ON q."_id" = r."query-id"
    JOIN read_parquet(?) AS d ON r."corpus-id" = d."_id"
    WHERE q."_id" = ?
    ORDER BY r.score DESC, r."corpus-id"
    """,
    params=[queries, qrels, corpus, qid],
).fetchall()

if not rows:
    print(
        f"Không tìm thấy query ID {qid!r}. "
        "Chạy 'python inspect_viequad.py --list' để xem ID hợp lệ; "
        "ví dụ: 'python inspect_viequad.py 3'."
    )
    raise SystemExit(1)

print(f"Query ID {qid}: {rows[0][1]}\n")
for _, _, doc_id, score, title, content in rows:
    print(f"Document ID {doc_id} | relevance {score} | {title}\n{content}\n")
