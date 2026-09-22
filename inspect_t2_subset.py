"""Show a Vietnamese T2Ranking query and its graded passages.

Examples:
    python inspect_t2_subset.py --list
    python inspect_t2_subset.py 1214
    python inspect_t2_subset.py 1214 --max-per-grade 2
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).parent / "data" / "t2ranking_vi" / "subset_150"


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return {row["_id"]: row for row in map(json.loads, f)}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_id", nargs="?")
    parser.add_argument("--list", action="store_true", help="List the 150 query IDs and texts")
    parser.add_argument("--max-per-grade", type=int, default=2, help="How many passages to show for each grade")
    args = parser.parse_args()

    queries = read_jsonl(ROOT / "queries.jsonl")
    if args.list:
        for qid, query in queries.items():
            print(f"{qid}\t{query['text']}")
        return
    qid = args.query_id or next(iter(queries))
    if qid not in queries:
        parser.error(f"Query ID {qid!r} is not in the subset; use --list")

    docs = read_jsonl(ROOT / "corpus.jsonl")
    labels = defaultdict(list)
    with (ROOT / "qrels.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["query-id"] == qid:
                labels[int(row["score"])].append(row["corpus-id"])

    print(f"Query {qid}: {queries[qid]['text']}")
    for grade in (3, 2, 1, 0):
        pids = labels[grade]
        print(f"\nGrade {grade}: {len(pids)} passages")
        for pid in pids[: args.max_per_grade]:
            print(f"  [{pid}] {docs[pid]['text']}")


if __name__ == "__main__":
    main()
