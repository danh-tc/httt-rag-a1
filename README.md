# Two-stage semantic search: VieQuADRetrieval (A1)

Search Vietnamese Wikipedia passages with free-form questions.

- **Stage 1:** BM25 (`pg_search`) + dense (`multilingual-e5-base`, `pgvector` HNSW), fused with RRF.
- **Stage 2:** cross-encoder `bge-reranker-v2-m3` reranks the top 50 candidates (Python, GPU).

## Prerequisites

NVIDIA GPU (driver ≥ 570 on Windows, ≥ 525 on Linux), Python 3.12, Docker, Node.js ≥ 20 (UI build only), ~10 GB disk, free ports 5432 and 8000.

Optional: put `HF_TOKEN=hf_...` in a `.env` file at the repo root for faster model downloads. Models are cached after the first run.

## Quick start

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\setup_venv_windows.ps1   # once
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
```
```bash
# Linux
bash scripts/setup_venv_linux.sh   # once
bash scripts/run_all.sh
```

Open http://localhost:8000. `run_all` starts the DB, ingests the corpus, builds the UI (each only if needed), then starts the API.
Options: `-Port 8080` / `PORT=8080`, `-Reingest` / `REINGEST=1`, `-RebuildUi` / `REBUILD_UI=1`, `-NoBrowser` / `NO_BROWSER=1`.
Stop with `Ctrl+C`; stop the DB with `docker compose stop`.

## Manual steps

```bash
docker compose up -d                                 # Postgres + pg_search + pgvector
python -m rag.ingest                                 # encode + load 2490 documents
npm --prefix web ci && npm --prefix web run build    # build the UI
python -m uvicorn rag.api:app --port 8000            # API + UI
python -m rag.search "Gandhi đấu tranh bất bạo động năm nào?"   # compare modes in the terminal
```

Modes: `bm25`, `dense`, `hybrid` (RRF), `hybrid_rerank`.

## Evaluation

```bash
python -m rag.evaluate                    # 4 modes, 2048 queries -> results/eval_full.json
python -m rag.ablation                    # full ablation sweep (resumable; --list shows progress)
python -m rag.evaluate --modes hybrid_rerank --candidate-n 20   # single run; also --embedder, --reranker, --fusion
```

Metrics: Recall / MRR / nDCG @10, with original labels (paragraph + answer span) and paragraph-only labels.