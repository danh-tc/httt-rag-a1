"""Bi-encoders used for both indexing and querying, so the per-model prefixes live in one place."""

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import DEFAULT_EMBEDDER, EMBED_BATCH_SIZE, EMBEDDERS

_models: dict[str, SentenceTransformer] = {}


def get_model(embedder: str = DEFAULT_EMBEDDER) -> SentenceTransformer:
    if embedder not in _models:
        _models[embedder] = SentenceTransformer(EMBEDDERS[embedder]["model"])
    return _models[embedder]


# e5 models are trained with asymmetric "query: " / "passage: " prefixes; dropping them hurts recall.
def encode_query(query: str, embedder: str = DEFAULT_EMBEDDER) -> np.ndarray:
    prefix, _ = EMBEDDERS[embedder]["prefix"]
    return get_model(embedder).encode(f"{prefix}{query}", normalize_embeddings=True)


def encode_passages(passages: list[str], embedder: str = DEFAULT_EMBEDDER, show_progress_bar: bool = False) -> np.ndarray:
    _, prefix = EMBEDDERS[embedder]["prefix"]
    return get_model(embedder).encode(
        [f"{prefix}{p}" for p in passages],
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
    )
