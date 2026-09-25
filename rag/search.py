"""Single entry point over every search configuration.

    python -m rag.search "Đại học nào lâu đời nhất ở Việt Nam?"
"""

import sys
import time
from dataclasses import asdict, dataclass

from rag.config import (
    CANDIDATE_N,
    DEFAULT_EMBEDDER,
    DEFAULT_FUSION,
    DEFAULT_K,
    DEFAULT_RERANKER,
    EMBEDDERS,
    FUSIONS,
    RERANKERS,
)
from rag.db import get_conn
from rag.rerank import rerank
from rag.retrieval import (
    Ranking,
    bm25_search,
    dense_search,
    fetch_texts,
    hybrid_search,
)

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class SearchConfig:
    """The ablation knobs. bm25 ignores all of them; dense reads only the embedder."""

    embedder: str = DEFAULT_EMBEDDER  # key of config.EMBEDDERS (dense side of every non-lexical mode)
    candidate_n: int = CANDIDATE_N  # per-retriever top-N fed to RRF, and the rerank pool
    fusion: str = DEFAULT_FUSION  # where RRF runs: "backend" | "sql"
    reranker: str = DEFAULT_RERANKER  # key of config.RERANKERS

    def __post_init__(self):
        if self.embedder not in EMBEDDERS:
            raise ValueError(f"embedder must be one of {tuple(EMBEDDERS)}")
        if self.candidate_n < 1:
            raise ValueError("candidate_n must be >= 1")
        if self.fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}")
        if self.reranker not in RERANKERS:
            raise ValueError(f"reranker must be one of {tuple(RERANKERS)}")

    def applies_to(self, mode: str) -> dict:
        """The subset of knobs that actually changes `mode`'s output (for labelling results)."""
        knobs = asdict(self)
        if mode == "dense":
            knobs = {"embedder": self.embedder}
        elif mode == "hybrid":
            knobs.pop("reranker")
        elif mode != "hybrid_rerank":
            knobs = {}
        return knobs


DEFAULT_CONFIG = SearchConfig()


def search(
    conn,
    query: str,
    mode: str,
    k: int = DEFAULT_K,
    config: SearchConfig = DEFAULT_CONFIG,
    timings: dict | None = None,
) -> Ranking:
    """Run one search. If `timings` is given, it is filled with retrieval_ms / rerank_ms."""
    start = time.perf_counter()
    if mode == "hybrid_rerank":
        # The rerank pool can't be smaller than k, or the UI would get fewer than k results.
        pool = max(config.candidate_n, k)
        candidates = hybrid_search(conn, query, pool, candidate_n=pool, fusion=config.fusion, embedder=config.embedder)
        texts = fetch_texts(conn, [doc_id for doc_id, _ in candidates])
        retrieved = time.perf_counter()
        results = rerank(
            query, [(doc_id, texts[doc_id]["body"]) for doc_id, _ in candidates], k, reranker=config.reranker
        )
        if timings is not None:
            timings["retrieval_ms"] = (retrieved - start) * 1000
            timings["rerank_ms"] = (time.perf_counter() - retrieved) * 1000
        return results

    if mode == "bm25":
        results = bm25_search(conn, query, k)
    elif mode == "dense":
        results = dense_search(conn, query, k, config.embedder)
    elif mode == "hybrid":
        results = hybrid_search(
            conn, query, k, candidate_n=config.candidate_n, fusion=config.fusion, embedder=config.embedder
        )
    else:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    if timings is not None:
        timings["retrieval_ms"] = (time.perf_counter() - start) * 1000
        timings["rerank_ms"] = 0.0
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    query = " ".join(sys.argv[1:]) or "Có bao nhiêu chiến cơ trong phi đội Iraq khi tẩu thoát sang Iran?"
    conn = get_conn()

    print(f"Query: {query}\n")
    for mode in MODES:
        results = search(conn, query, mode, k=5)
        texts = fetch_texts(conn, [doc_id for doc_id, _ in results])
        print(f"--- {mode} ---")
        for doc_id, score in results:
            doc = texts[doc_id]
            print(f"  {doc_id} ({score:.4f}) {doc['title']!r} {doc['body'][:120].replace(chr(10), ' ')}...")
        print()
