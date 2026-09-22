"""Tier-2 rerank: cross-encoder re-scores tier-1 candidates. Runs in the service layer, not the DB."""

from sentence_transformers import CrossEncoder

RERANKER_NAME = "BAAI/bge-reranker-v2-m3"

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_NAME, max_length=512)
    return _reranker


def rerank(query: str, candidates: list[tuple[str, str]], k: int) -> list[tuple[str, float]]:
    """candidates: list of (doc_id, text). Returns top-k (doc_id, score) sorted best first."""
    if not candidates:
        return []
    reranker = get_reranker()
    pairs = [(query, text) for _, text in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip((doc_id for doc_id, _ in candidates), scores), key=lambda item: item[1], reverse=True
    )
    return ranked[:k]
