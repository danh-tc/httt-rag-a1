"""Evaluate bm25 / dense / hybrid / hybrid_rerank on VieQuADRetrieval: Recall@k, MRR@k, nDCG@k."""

import argparse
import json
import math
import random
import time

from retrieval import get_conn
from search import MODES, search as run_search
from viequad_loader import load_qrels, load_queries


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    return len(set(retrieved) & relevant) / len(relevant)


def mrr_at_k(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevance: dict[str, int]) -> float:
    dcg = sum(
        relevance.get(doc_id, 0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved, start=1)
    )
    ideal_gains = sorted(relevance.values(), reverse=True)[: len(retrieved)]
    idcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(mode: str, queries: dict, qrels: dict, k: int, conn, qids: list[str]):
    per_query = {}
    for qid in qids:
        relevant = qrels.get(qid)
        if not relevant:
            continue
        results = run_search(queries[qid], mode, k=k, conn=conn)
        retrieved = [doc_id for doc_id, _ in results]
        per_query[qid] = {
            "recall": recall_at_k(retrieved, set(relevant)),
            "mrr": mrr_at_k(retrieved, set(relevant)),
            "ndcg": ndcg_at_k(retrieved, relevant),
            "retrieved": retrieved,
            "relevant": list(relevant),
        }

    n = len(per_query)
    agg = {
        "n": n,
        "recall": sum(v["recall"] for v in per_query.values()) / n,
        "mrr": sum(v["mrr"] for v in per_query.values()) / n,
        "ndcg": sum(v["ndcg"] for v in per_query.values()) / n,
    }
    return agg, per_query


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sample", type=int, default=None, help="evaluate a random subset of queries")
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--save", default="eval_results.json")
    args = parser.parse_args()

    queries = load_queries()
    qrels = load_qrels()
    conn = get_conn()

    qids = [qid for qid in queries if qrels.get(qid)]
    if args.sample:
        random.Random(42).shuffle(qids)
        qids = qids[: args.sample]

    print(f"Evaluating on {len(qids)} queries, k={args.k}, modes={args.modes}\n")

    try:
        with open(args.save, encoding="utf-8") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}

    for mode in args.modes:
        start = time.time()
        agg, per_query = evaluate(mode, queries, qrels, args.k, conn, qids)
        elapsed = time.time() - start
        all_results[mode] = {"agg": agg, "per_query": per_query}
        print(
            f"{mode:15s} n={agg['n']:4d}  Recall@{args.k}={agg['recall']:.4f}  "
            f"MRR@{args.k}={agg['mrr']:.4f}  nDCG@{args.k}={agg['ndcg']:.4f}  ({elapsed:.1f}s)"
        )
        # Save after every mode, not just at the end, so an interrupted run keeps earlier progress.
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved per-query breakdown to {args.save}")
