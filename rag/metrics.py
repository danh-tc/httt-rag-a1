"""Ranking metrics over a single query's top-k list. Average across queries for the aggregate."""

import math


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
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    return dcg / idcg if idcg > 0 else 0.0
