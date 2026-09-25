"""Tier-2 rerank: a cross-encoder re-scores tier-1 candidates. Runs in the service layer, not the DB."""

from sentence_transformers import CrossEncoder

from rag.config import DEFAULT_RERANKER, RERANK_MAX_LENGTH, RERANKERS

_rerankers: dict[str, CrossEncoder] = {}


def get_reranker(name: str = DEFAULT_RERANKER) -> CrossEncoder:
    """Load (once) the reranker registered under `name` in config.RERANKERS."""
    if name not in RERANKERS:
        raise ValueError(f"unknown reranker {name!r}, expected one of {tuple(RERANKERS)}")
    if name not in _rerankers:
        _rerankers[name] = CrossEncoder(RERANKERS[name], max_length=RERANK_MAX_LENGTH)
    return _rerankers[name]


def rerank(
    query: str, candidates: list[tuple[str, str]], k: int, reranker: str = DEFAULT_RERANKER
) -> list[tuple[str, float]]:
    """candidates: [(doc_id, text), ...]. Returns the top-k (doc_id, score), best first."""
    if not candidates:
        return []
    scores = get_reranker(reranker).predict([(query, text) for _, text in candidates])
    ranked = sorted(
        zip((doc_id for doc_id, _ in candidates), scores), key=lambda item: item[1], reverse=True
    )
    return [(doc_id, float(score)) for doc_id, score in ranked[:k]]
