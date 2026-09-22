"""Unified search entry point over the 4 configurations: bm25 / dense / hybrid / hybrid_rerank."""

from retrieval import bm25_search, dense_search, fetch_texts, get_conn, hybrid_search
from rerank import rerank

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


def search(
    query: str, mode: str, k: int = 10, candidate_n: int = 50, conn=None
) -> list[tuple[str, float]]:
    conn = conn or get_conn()
    if mode == "bm25":
        return bm25_search(query, k, conn=conn)
    if mode == "dense":
        return dense_search(query, k, conn=conn)
    if mode == "hybrid":
        return hybrid_search(query, k, candidate_n=candidate_n, conn=conn)
    if mode == "hybrid_rerank":
        candidates = hybrid_search(query, candidate_n, candidate_n=candidate_n, conn=conn)
        texts = fetch_texts([doc_id for doc_id, _ in candidates], conn=conn)
        pairs = [(doc_id, texts[doc_id]["body"]) for doc_id, _ in candidates]
        return rerank(query, pairs, k)
    raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Có bao nhiêu chiến cơ trong phi đội Iraq khi tẩu thoát sang Iran?"
    conn = get_conn()

    print(f"Query: {query}\n")
    for mode in MODES:
        results = search(query, mode, k=5, conn=conn)
        texts = fetch_texts([doc_id for doc_id, _ in results], conn=conn)
        print(f"--- {mode} ---")
        for doc_id, score in results:
            snippet = texts[doc_id]["body"][:120].replace("\n", " ")
            print(f"  {doc_id} ({score:.4f}) {texts[doc_id]['title']!r} {snippet}...")
        print()
