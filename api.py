"""FastAPI backend exposing /search over the 4 retrieval configurations."""

import time

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rerank import get_reranker
from retrieval import fetch_texts, get_conn, get_model
from search import MODES, search as run_search

app = FastAPI(title="VieQuAD Search")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_conn = None


class SearchResult(BaseModel):
    doc_id: str
    score: float
    title: str | None
    body: str


class SearchResponse(BaseModel):
    mode: str
    device: str
    took_ms: float
    results: list[SearchResult]


@app.on_event("startup")
def warm_up():
    # Load models + open the DB connection once, so the first real request isn't slow.
    global _conn
    _conn = get_conn()
    get_model()
    get_reranker()


@app.get("/search", response_model=SearchResponse)
def search_endpoint(q: str, mode: str = "hybrid_rerank", k: int = 10):
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    if not q.strip():
        raise HTTPException(400, "q must not be empty")

    start = time.perf_counter()
    results = run_search(q, mode, k=k, conn=_conn)
    texts = fetch_texts([doc_id for doc_id, _ in results], conn=_conn)
    took_ms = (time.perf_counter() - start) * 1000

    return SearchResponse(
        mode=mode,
        device=DEVICE,
        took_ms=took_ms,
        results=[
            SearchResult(
                doc_id=doc_id, score=float(score), title=texts[doc_id]["title"], body=texts[doc_id]["body"]
            )
            for doc_id, score in results
        ],
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")
