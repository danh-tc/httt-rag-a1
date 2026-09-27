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

Runs are resumable: progress is checkpointed every CHECKPOINT_EVERY queries (atomically, so a power
cut can't corrupt the file). Re-running the same command skips finished runs and continues a partial
one from where it stopped. The whole ablation sweep is `python -m rag.ablation`.
"""

import argparse
import json
import os
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

CHECKPOINT_EVERY = 100
METRICS = ["recall", "mrr", "ndcg", "recall_para", "mrr_para", "ndcg_para"] + [f"found_{kind}" for kind in KIND_LABELS]


def run_key(mode: str, config: SearchConfig) -> str:
    """Name a run by the knobs that differ from the defaults and actually affect this mode."""
    default = DEFAULT_CONFIG.applies_to(mode)
    changed = {k: v for k, v in config.applies_to(mode).items() if default[k] != v}
    return f"{mode}[{','.join(f'{k}={v}' for k, v in changed.items())}]" if changed else mode


def load_data(sample: int | None = None):
    """(queries, qrels, kinds) and the judged query ids, optionally a seeded random subset."""
    queries, qrels = load_queries(), load_qrels()
    kinds = doc_kinds(load_corpus(), qrels)
    qids = [qid for qid in queries if qrels.get(qid)]
    if sample:
        random.Random(42).shuffle(qids)
        qids = qids[:sample]
    return (queries, qrels, kinds), qids


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_results(path: Path, results: dict):
    """Write via a temp file + rename, so an interrupted write never leaves a truncated JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def is_complete(run: dict | None, qids: list[str]) -> bool:
    # Runs saved before timings were recorded lack ms_per_query; treat them as stale and redo them.
    return (
        bool(run)
        and not run.get("partial")
        and "ms_per_query" in run.get("agg", {})
        and set(run["per_query"]) == set(qids)
    )


def run_one(conn, mode: str, config: SearchConfig, data: tuple, qids: list[str], k: int, results: dict, save_path: Path):
    """Evaluate one (mode, config) into results[key], resuming a partial run and checkpointing as it goes."""
    queries, qrels, kinds = data
    key = run_key(mode, config)
    previous = results.get(key)
    if is_complete(previous, qids):
        print(f"{key:40s} already done, skipping", flush=True)
        return
    resuming = bool(previous and previous.get("partial"))
    per_query = previous["per_query"] if resuming else {}
    seconds = previous.get("seconds_so_far", 0.0) if resuming else 0.0
    todo = [qid for qid in qids if qid not in per_query]
    print(f"{key:40s} {'resuming at' if resuming else 'starting'} {len(per_query)}/{len(qids)}", flush=True)

    search(conn, queries[todo[0]], mode, k=k, config=config)  # warm-up: load models before timing
    start = time.time()

    def record(partial: bool):
        run = {"mode": mode, "k": k, "config": config.applies_to(mode), "per_query": per_query}
        if partial:
            run |= {"partial": True, "seconds_so_far": seconds + time.time() - start}
        else:
            run["agg"] = aggregate(per_query) | {"seconds": round(seconds + time.time() - start, 1)}
        results[key] = run
        save_results(save_path, results)

    for i, qid in enumerate(todo, start=1):
        timings: dict = {}
        retrieved = [doc_id for doc_id, _ in search(conn, queries[qid], mode, k=k, config=config, timings=timings)]
        per_query[qid] = score(retrieved, qrels[qid], kinds) | timings | {"retrieved": retrieved, "relevant": list(qrels[qid])}
        if i % CHECKPOINT_EVERY == 0 and i < len(todo):
            record(partial=True)
            print(f"  {key}: {len(per_query)}/{len(qids)} ({time.time() - start:.0f}s)", flush=True)
    record(partial=False)
    report(key, results[key]["agg"], k)


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
        f"span={agg['found_span']:.1%}  {agg.get('ms_per_query', '?')}ms/q",
        flush=True,
    )


def rescore(path: Path, qrels: dict, kinds: dict):
    """Recompute every run's metrics from its saved rankings (e.g. after adding a metric)."""
    all_results = load_results(path)
    for key, run in all_results.items():
        per_query = run["per_query"]
        if not per_query or run.get("partial"):
            continue
        for qid, row in per_query.items():
            row.update(score(row["retrieved"], qrels[qid], kinds))
        # Only the quality metrics; timings (absent from older runs) are kept as recorded.
        run["agg"] |= {m: statistics.fmean(row[m] for row in per_query.values()) for m in METRICS}
        run.setdefault("k", DEFAULT_K)  # early runs didn't record k; they all used the default
        report(key, run["agg"], run["k"])
    save_results(path, all_results)
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
    data, qids = load_data(args.sample)
    print(f"Evaluating on {len(qids)} queries, k={args.k}, modes={args.modes}, config={asdict(config)}\n")

    # Merge into an existing results file so modes/configs can be (re)run one at a time.
    results = load_results(args.save)
    conn = get_conn()
    for mode in args.modes:
        run_one(conn, mode, config, data, qids, args.k, results, args.save)
    print(f"\nSaved per-query breakdown to {args.save}")


if __name__ == "__main__":
    main()
