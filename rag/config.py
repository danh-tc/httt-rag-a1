"""Project-wide settings. Anything deployment-specific can be overridden via environment variables."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "VieQuADRetrieval"
STATIC_DIR = ROOT_DIR / "web" / "dist"  # built React UI: npm --prefix web run build
RESULTS_DIR = ROOT_DIR / "results"

DB_DSN = os.getenv("RAG_DB_DSN", "postgresql://rag:rag@localhost:5432/rag")

# Embedders for the model ablation. Each has its own vector column on `documents`; "e5-base" keeps
# the original `embedding` column. `prefix` = the (query, passage) prefixes the model was trained with.
EMBEDDERS = {
    "e5-base": {  # 278M params, multilingual
        "model": "intfloat/multilingual-e5-base",
        "dim": 768,
        "column": "embedding",
        "prefix": ("query: ", "passage: "),
    },
    "e5-small": {  # 118M params, multilingual, same family: isolates model size
        "model": "intfloat/multilingual-e5-small",
        "dim": 384,
        "column": "embedding_e5_small",
        "prefix": ("query: ", "passage: "),
    },
    "bge-m3": {  # 568M params, multilingual, 1024-d: does a stronger embedder still help? No prefixes needed.
        "model": "BAAI/bge-m3",
        "dim": 1024,
        "column": "embedding_bge_m3",
        "prefix": ("", ""),
    },
    "minilm": {  # 22M params, English-only training: does a multilingual model matter for Vietnamese?
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "column": "embedding_minilm",
        "prefix": ("", ""),
    },
}
DEFAULT_EMBEDDER = os.getenv("RAG_EMBEDDER", "e5-base")
EMBED_BATCH_SIZE = 64
# Rerankers selectable per request (API/UI) and per eval run (--reranker), for the model ablation.
RERANKERS = {
    "bge-m3": "BAAI/bge-reranker-v2-m3",  # 568M params, multilingual
    "mminilm": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # 118M params, multilingual (mMARCO)
}
DEFAULT_RERANKER = os.getenv("RAG_RERANKER", "bge-m3")
RERANK_MAX_LENGTH = 512

# Where tier-1 RRF fusion runs: "backend" (Python) or "sql" (one Postgres query). Same ranking, different latency.
FUSIONS = ("backend", "sql")
DEFAULT_FUSION = "backend"

# pgvector HNSW returns at most ef_search rows (default 40), which silently truncated the
# 50-candidate dense list and made it noticeably approximate. The corpus is small, so search wide.
HNSW_EF_SEARCH = 200

RRF_K = 60  # standard RRF damping constant
CANDIDATE_N = 50  # tier-1 candidates per retriever, and the pool handed to the reranker
DEFAULT_K = 10
