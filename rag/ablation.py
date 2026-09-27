"""Run the whole ablation sweep in one process, resumably.

    python -m rag.ablation                      # all runs, 2048 queries -> results/eval_full.json
    python -m rag.ablation --skip n100          # leave out the slowest run (N = 100)
    python -m rag.ablation --list               # show the plan and what is already done
    python -m rag.ablation --sample 20 --save results/eval_sample20.json

Interrupted (Ctrl+C, crash, power cut)? Run the same command again: finished runs are skipped and a
partial run continues from its last checkpoint (every 100 queries). Each run keeps only the models it
uses on the GPU (8 GB cannot hold all six at once).
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

from rag import dataset_stats, embedding, rerank
from rag.config import RESULTS_DIR
from rag.db import get_conn
from rag.evaluate import is_complete, load_data, load_results, run_key, run_one
from rag.search import MODES, SearchConfig

# (id, ablation, modes, config overrides). Cheap runs first; the slowest (N = 100) last so it can be cut.
RUNS = [
    ("main", "1. Bảng chính + baseline", MODES, {}),
    ("e5-small", "2. Embedding e5-small", ("dense", "hybrid"), {"embedder": "e5-small"}),
    ("bge-m3", "2. Embedding bge-m3", ("dense", "hybrid"), {"embedder": "bge-m3"}),
    ("minilm", "2. Embedding MiniLM", ("dense", "hybrid"), {"embedder": "minilm"}),
    ("mminilm", "3. Reranker mMiniLM", ("hybrid_rerank",), {"reranker": "mminilm"}),
    ("sql", "4. RRF trong SQL", ("hybrid",), {"fusion": "sql"}),
    ("n20", "5. N = 20", ("hybrid", "hybrid_rerank"), {"candidate_n": 20}),
    ("light", "6. Cấu hình nhẹ", ("hybrid_rerank",), {"embedder": "e5-small", "candidate_n": 20}),
    ("n100", "5. N = 100", ("hybrid", "hybrid_rerank"), {"candidate_n": 100}),
]


def keep_only_models_for(mode: str, config: SearchConfig):
    """Free every model this run doesn't use. Keeping all 6 cached overflowed the 8 GB GPU: Windows
    then spilled to system RAM and the N = 100 rerank ran ~12x slower."""
    embedding.unload_except(set() if mode == "bm25" else {config.embedder})
    rerank.unload_except({config.reranker} if mode == "hybrid_rerank" else set())
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=None, help="random subset of N queries (seed 42)")
    parser.add_argument("--save", type=Path, default=RESULTS_DIR / "eval_full.json")
    parser.add_argument("--skip", nargs="*", default=[], choices=[r[0] for r in RUNS], help="run ids to leave out")
    parser.add_argument("--list", action="store_true", help="print the plan with done/partial/todo status, then exit")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    data, qids = load_data(args.sample)
    results = load_results(args.save)
    plan = [(rid, label, mode, SearchConfig(**overrides)) for rid, label, modes, overrides in RUNS if rid not in args.skip for mode in modes]

    print(f"{len(qids)} queries -> {args.save}\n")
    for rid, label, mode, config in plan:
        run = results.get(run_key(mode, config))
        status = "done" if is_complete(run, qids) else f"partial {len(run['per_query'])}" if run and run.get("partial") else "todo"
        print(f"  [{status:>12}] {label:28s} {run_key(mode, config)}")
    if args.list:
        return

    print()
    conn = get_conn()
    start = time.time()
    for rid, label, mode, config in plan:
        keep_only_models_for(mode, config)
        run_one(conn, mode, config, data, qids, args.k, results, args.save)
    print(f"\nAll runs finished in {(time.time() - start) / 60:.1f} min -> {args.save}")

    if args.save == RESULTS_DIR / "eval_full.json":
        dataset_stats.main()  # refresh found_by_kind.png from the full results


if __name__ == "__main__":
    main()
