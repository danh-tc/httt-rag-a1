"""FastAPI backend: GET /search over the 4 modes (ablation knobs optional, for curl), dataset browsing, and the static frontend at /.

    python -m uvicorn rag.api:app --port 8000
"""

import logging
import threading
import time
import unicodedata
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag.config import CANDIDATE_N, DEFAULT_K, EMBEDDERS, FUSIONS, RERANKERS, STATIC_DIR
from rag.dataset import load_corpus, load_qrels, load_queries
from rag.dataset_stats import FIGURES_DIR, profile, public
from rag.db import get_conn
from rag.embedding import get_model
from rag.rerank import get_reranker
from rag.retrieval import fetch_texts
from rag.search import DEFAULT_CONFIG, SearchConfig, search

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SearchResult(BaseModel):
    doc_id: str
    score: float
    title: str | None
    body: str
    kind: str  # "paragraph" (context passage) | "span" (answer span cut from a paragraph)


class SearchResponse(BaseModel):
    mode: str
    device: str
    took_ms: float
    # Tier-1 (retrieval + RRF) vs tier-2 (cross-encoder) time; rerank_ms is 0 for non-rerank modes.
    retrieval_ms: float
    rerank_ms: float
    # The knobs that actually shaped this result (empty for bm25/dense).
    config: dict
    results: list[SearchResult]
    # Set when q is exactly a dataset question: its id and {doc_id: "paragraph" | "span"} ground truth.
    qid: str | None = None
    judged: dict[str, str] | None = None


class QueryItem(BaseModel):
    qid: str
    text: str


class QueryPage(BaseModel):
    total: int
    items: list[QueryItem]


class JudgedDoc(BaseModel):
    doc_id: str
    kind: str
    title: str | None
    body: str


class QueryDetail(QueryItem):
    relevant: list[JudgedDoc]


state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load models and open the DB connection once, so the first real request isn't slow.
    state["conn"] = get_conn()
    # Sync endpoints run in a threadpool; a psycopg2 connection must not be used by two threads at once.
    state["lock"] = threading.Lock()
    get_model()
    get_reranker()

    # The dataset is small (~2.5k docs), so keep it in memory for the Dataset tab.
    corpus, queries, qrels = load_corpus(), load_queries(), load_qrels()
    stats = profile(corpus, queries, qrels)
    state.update(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        kinds=stats["_kinds"],
        stats=public(stats),
        qid_by_text={text.strip(): qid for qid, text in queries.items()},
        sorted_qids=sorted(queries, key=int),
    )
    yield
    state["conn"].close()


app = FastAPI(title="VieQuAD Search", lifespan=lifespan)


Mode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]  # keep in sync with rag.search.MODES
Fusion = Literal[FUSIONS]
Reranker = Literal[tuple(RERANKERS)]
Embedder = Literal[tuple(EMBEDDERS)]


@app.get("/search", response_model=SearchResponse)
def search_endpoint(
    q: Annotated[str, Query(min_length=1, pattern=r"\S")],
    mode: Mode = "hybrid_rerank",
    k: Annotated[int, Query(ge=1, le=100)] = DEFAULT_K,
    candidate_n: Annotated[int, Query(ge=5, le=200)] = CANDIDATE_N,
    fusion: Fusion = DEFAULT_CONFIG.fusion,
    reranker: Reranker = DEFAULT_CONFIG.reranker,
    embedder: Embedder = DEFAULT_CONFIG.embedder,
):
    # Some Vietnamese IMEs emit decomposed (NFD) diacritics; the corpus is NFC, so BM25 would miss.
    q = unicodedata.normalize("NFC", q).strip()
    config = SearchConfig(embedder=embedder, candidate_n=candidate_n, fusion=fusion, reranker=reranker)
    timings: dict = {}
    start = time.perf_counter()
    with state["lock"]:
        results = search(state["conn"], q, mode, k=k, config=config, timings=timings)
        texts = fetch_texts(state["conn"], [doc_id for doc_id, _ in results])
    took_ms = (time.perf_counter() - start) * 1000

    qid = state["qid_by_text"].get(q)
    return SearchResponse(
        mode=mode,
        device=DEVICE,
        took_ms=took_ms,
        **timings,
        config=config.applies_to(mode),
        results=[
            SearchResult(doc_id=doc_id, score=float(score), kind=state["kinds"][doc_id], **texts[doc_id])
            for doc_id, score in results
        ],
        qid=qid,
        judged={d: state["kinds"][d] for d in state["qrels"][qid]} if qid else None,
    )


@app.get("/dataset/stats")
def dataset_stats() -> dict:
    return state["stats"]


@app.get("/dataset/queries", response_model=QueryPage)
def dataset_queries(
    q: str = "",
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    needle = unicodedata.normalize("NFC", q).strip().lower()
    qids = [qid for qid in state["sorted_qids"] if needle in state["queries"][qid].lower() or needle == qid]
    return QueryPage(
        total=len(qids),
        items=[QueryItem(qid=qid, text=state["queries"][qid]) for qid in qids[offset : offset + limit]],
    )


@app.get("/dataset/queries/{qid}", response_model=QueryDetail, responses={404: {"description": "Unknown query id"}})
def dataset_query(qid: str):
    if qid not in state["queries"]:
        raise HTTPException(404, f"unknown query id {qid!r}")
    corpus = state["corpus"]
    relevant = sorted(state["qrels"][qid], key=lambda d: state["kinds"][d] != "paragraph")  # paragraph first
    return QueryDetail(
        qid=qid,
        text=state["queries"][qid],
        relevant=[
            JudgedDoc(doc_id=d, kind=state["kinds"][d], title=corpus[d]["title"], body=corpus[d]["text"])
            for d in relevant
        ],
    )


# Figures come from `python -m rag.dataset_stats`; the Dataset tab hides any that don't exist yet.
app.mount("/figures", StaticFiles(directory=FIGURES_DIR, check_dir=False), name="figures")
if not (STATIC_DIR / "index.html").exists():
    logging.getLogger("uvicorn.error").warning(
        "UI not built (%s missing); the API still works. Build it: npm --prefix web ci && npm --prefix web run build",
        STATIC_DIR,
    )
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True, check_dir=False), name="static")
