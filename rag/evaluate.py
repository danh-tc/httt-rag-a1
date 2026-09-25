"""Evaluate search modes on VieQuADRetrieval: Recall@k, MRR@k, nDCG@k, found-rate per doc kind, latency.

    python -m rag.evaluate                                   # all 4 modes, all 2048 queries
    python -m rag.evaluate --sample 20 --save results/eval_sample20.json
    python -m rag.evaluate --modes hybrid_rerank --candidate-n 20 --reranker mminilm   # tier-2 ablation
    python -m rag.evaluate --modes dense hybrid --embedder minilm                       # embedding ablation

    python -m rag.evaluate --rescore                         # recompute metrics from saved rankings, no search

Two sets of metrics are reported. The dataset's own qrels judge 2 docs per query (the context
paragraph and the answer span); the "_para" metrics count only the paragraph as relevant, since the
span is a few syllables with almost no word overlap and is not what a searcher actually wants.

Each (mode, config) is stored under its own key, e.g. "hybrid_rerank" for the defaults and
"hybrid_rerank[candidate_n=20,reranker=mminilm]" otherwise, so ablation runs never overwrite each other.
"""

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

from rag.config import DEFAULT_K, EMBEDDERS, FUSIONS, RERANKERS, RESULTS_DIR
from rag.dataset import load_corpus, load_qrels, load_queries
from rag.dataset_stats import KIND_LABELS, PARAGRAPH, doc_kinds
from rag.db import get_conn
from rag.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from rag.search import DEFAULT_CONFIG, MODES, SearchConfig, search


def run_key(mode: str, config: SearchConfig) -> str:
    """Name a run by the knobs that differ from the defaults and actually affect this mode."""
    default = DEFAULT_CONFIG.applies_to(mode)
    changed = {k: v for k, v in config.applies_to(mode).items() if default[k] != v}
    return f"{mode}[{','.join(f'{k}={v}' for k, v in changed.items())}]" if changed else mode


def evaluate(conn, mode: str, config: SearchConfig, queries: dict, qrels: dict, kinds: dict, qids: list[str], k: int):
    search(conn, queries[qids[0]], mode, k=k, config=config)  # warm-up: load models before timing

    per_query = {}
    for qid in qids:
        timings: dict = {}
        retrieved = [doc_id for doc_id, _ in search(conn, queries[qid], mode, k=k, config=config, timings=timings)]
        per_query[qid] = score(retrieved, qrels[qid], kinds) | timings | {"retrieved": retrieved, "relevant": list(qrels[qid])}
    return aggregate(per_query), per_query


METRICS = ["recall", "mrr", "ndcg", "recall_para", "mrr_para", "ndcg_para"] + [f"found_{kind}" for kind in KIND_LABELS]


def score(retrieved: list[str], relevance: dict[str, int], kinds: dict[str, str]) -> dict[str, float]:
    """Per-query metrics against the full qrels and against the paragraph only."""
    relevant = set(relevance)
    para = {d: g for d, g in relevance.items() if kinds[d] == PARAGRAPH}
    return {
        "recall": recall_at_k(retrieved, relevant),
        "mrr": mrr_at_k(retrieved, relevant),
        "ndcg": ndcg_at_k(retrieved, relevance),
        "recall_para": recall_at_k(retrieved, set(para)),
        "mrr_para": mrr_at_k(retrieved, set(para)),
        "ndcg_para": ndcg_at_k(retrieved, para),
        # Which half of the judgment was found: the context paragraph and/or the answer span.
        **{f"found_{kind}": float(any(kinds[d] == kind for d in relevant & set(retrieved))) for kind in KIND_LABELS},
    }


def aggregate(per_query: dict) -> dict:
    rows = per_query.values()
    agg = {"n": len(per_query)} | {m: statistics.fmean(v[m] for v in rows) for m in METRICS}
    for t in ("retrieval_ms", "rerank_ms"):
        agg[t] = round(statistics.fmean(v[t] for v in rows), 1)
    agg["ms_per_query"] = round(agg["retrieval_ms"] + agg["rerank_ms"], 1)
    return agg


def report(key: str, agg: dict, k: int):
    print(
        f"{key:40s} n={agg['n']:4d}  R@{k}={agg['recall']:.4f}  MRR@{k}={agg['mrr']:.4f}  nDCG@{k}={agg['ndcg']:.4f}  | "
        f"para-only R={agg['recall_para']:.4f} MRR={agg['mrr_para']:.4f} nDCG={agg['ndcg_para']:.4f}  | "
        f"span={agg['found_span']:.1%}  {agg.get('ms_per_query', '?')}ms/q"
    )


def rescore(path: Path, qrels: dict, kinds: dict):
    """Recompute every run's metrics from its saved rankings (e.g. after adding a metric)."""
    all_results = json.loads(path.read_text(encoding="utf-8"))
    for key, run in all_results.items():
        per_query = run["per_query"]
        if not per_query:
            continue
        for qid, row in per_query.items():
            row.update(score(row["retrieved"], qrels[qid], kinds))
        # Only the quality metrics; timings (absent from older runs) are kept as recorded.
        run["agg"] |= {m: statistics.fmean(row[m] for row in per_query.values()) for m in METRICS}
        run.setdefault("k", DEFAULT_K)  # early runs didn't record k; they all used the default
        report(key, run["agg"], run["k"])
    path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRescored {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--sample", type=int, default=None, help="evaluate a random subset of N queries (seed 42)")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--embedder", choices=tuple(EMBEDDERS), default=DEFAULT_CONFIG.embedder,
                        help="bi-encoder for the dense side (dense and hybrid modes)")
    parser.add_argument("--candidate-n", type=int, default=DEFAULT_CONFIG.candidate_n,
                        help="per-retriever top-N for RRF and the rerank pool (hybrid modes)")
    parser.add_argument("--fusion", choices=FUSIONS, default=DEFAULT_CONFIG.fusion, help="where RRF runs (hybrid modes)")
    parser.add_argument("--reranker", choices=tuple(RERANKERS), default=DEFAULT_CONFIG.reranker,
                        help="cross-encoder for hybrid_rerank")
    parser.add_argument("--save", type=Path, default=RESULTS_DIR / "eval_full.json")
    parser.add_argument("--rescore", action="store_true", help="recompute metrics in --save from saved rankings, then exit")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.rescore:
        qrels = load_qrels()
        rescore(args.save, qrels, doc_kinds(load_corpus(), qrels))
        return

    config = SearchConfig(embedder=args.embedder, candidate_n=args.candidate_n, fusion=args.fusion, reranker=args.reranker)
    queries, qrels = load_queries(), load_qrels()
    kinds = doc_kinds(load_corpus(), qrels)
    qids = [qid for qid in queries if qrels.get(qid)]
    if args.sample:
        random.Random(42).shuffle(qids)
        qids = qids[: args.sample]

    print(f"Evaluating on {len(qids)} queries, k={args.k}, modes={args.modes}, config={asdict(config)}\n")

    # Merge into an existing results file so modes/configs can be (re)run one at a time.
    all_results = json.loads(args.save.read_text(encoding="utf-8")) if args.save.exists() else {}
    args.save.parent.mkdir(parents=True, exist_ok=True)

    conn = get_conn()
    for mode in args.modes:
        key = run_key(mode, config)
        start = time.time()
        agg, per_query = evaluate(conn, mode, config, queries, qrels, kinds, qids, args.k)
        agg["seconds"] = round(time.time() - start, 1)
        all_results[key] = {"mode": mode, "k": args.k, "config": config.applies_to(mode), "agg": agg, "per_query": per_query}
        report(key, agg, args.k)
        # Save after every run so an interrupted sweep keeps earlier progress.
        args.save.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved per-query breakdown to {args.save}")


if __name__ == "__main__":
    main()
