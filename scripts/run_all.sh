#!/usr/bin/env bash
# Start the whole system on Linux: Docker DB -> ingest (if needed) -> dataset figures (if needed) -> UI build (if needed) -> API + UI.
#
# Usage (after scripts/setup_venv_linux.sh):
#   bash scripts/run_all.sh
#   PORT=8080 HOST=0.0.0.0 bash scripts/run_all.sh     # expose on the LAN / a remote server
#   REINGEST=1 bash scripts/run_all.sh                 # force re-encoding the corpus
#   REBUILD_UI=1 bash scripts/run_all.sh               # force rebuilding the React UI (needs Node.js)
#
# Stop with Ctrl+C (the DB keeps running; stop it with: docker compose stop).
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
REINGEST="${REINGEST:-0}"
REBUILD_UI="${REBUILD_UI:-0}"
NO_BROWSER="${NO_BROWSER:-0}"
DB_CONTAINER="httt-rag-a1-db"

cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTHONIOENCODING=utf-8

wait_for() {  # wait_for <seconds> <description> <command...>
    local timeout=$1 what=$2; shift 2
    local deadline=$((SECONDS + timeout))
    until "$@" >/dev/null 2>&1; do
        if (( SECONDS > deadline )); then echo "$what not ready within ${timeout}s." >&2; return 1; fi
        sleep 2
    done
}

echo "==> [1/7] Checking venv"
if [[ ! -x "$PY" ]]; then
    echo ".venv not found. Run first: bash scripts/setup_venv_linux.sh" >&2
    exit 1
fi

echo "==> [2/7] Checking Docker"
if ! command -v docker >/dev/null; then
    echo "docker not found. Install Docker Engine + the compose plugin." >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "Cannot talk to the Docker daemon. Either it is stopped (sudo systemctl start docker)" >&2
    echo "or your user lacks permission (sudo usermod -aG docker \$USER, then log in again)." >&2
    exit 1
fi

echo "==> [3/7] Starting Postgres (ParadeDB)"
docker compose up -d
wait_for 60 "Postgres" docker exec "$DB_CONTAINER" pg_isready -U rag -d rag || { echo "Check: docker logs $DB_CONTAINER" >&2; exit 1; }

echo "==> [4/7] Checking indexed documents"
in_db=$(docker exec "$DB_CONTAINER" psql -U rag -d rag -tAc "SELECT count(*) FROM documents" 2>/dev/null | head -n1 || true)
in_corpus=$("$PY" -c "from rag.dataset import load_corpus; print(len(load_corpus()))")
echo "    documents in DB: ${in_db:-?} / corpus: $in_corpus"
if [[ "$REINGEST" == "1" || "$in_db" != "$in_corpus" ]]; then
    echo "    Ingesting (first run downloads the embedding model, ~1 GB)..."
    "$PY" -m rag.ingest
fi

echo "==> [5/7] Checking dataset figures"
if [[ ! -f results/figures/doc_length.png ]]; then
    "$PY" -m rag.dataset_stats
else
    echo "    results/figures already present (regenerate with: python -m rag.dataset_stats)"
fi

echo "==> [6/7] Checking UI build"
if [[ "$REBUILD_UI" == "1" || ! -f web/dist/index.html ]]; then
    command -v npm >/dev/null || { echo "npm not found. Install Node.js >= 20 to build the UI (web/)." >&2; exit 1; }
    npm --prefix web ci
    npm --prefix web run build
else
    echo "    web/dist already built (rebuild with: REBUILD_UI=1)"
fi

echo "==> [7/7] Starting API + UI on http://$HOST:$PORT"
if "$PY" -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1', $PORT)) != 0)"; then
    echo "Port $PORT is already in use. Stop that process or set PORT=<other>." >&2
    exit 1
fi
if [[ "$NO_BROWSER" != "1" && -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v xdg-open >/dev/null; then
    # Open the browser once the server answers (model loading takes a while on the first run).
    ( wait_for 600 "API" curl -sf "http://127.0.0.1:$PORT/" && xdg-open "http://127.0.0.1:$PORT/" ) >/dev/null 2>&1 &
fi
echo "    First start loads the models (~3 GB download on a fresh machine). Ctrl+C to stop."
exec "$PY" -m uvicorn rag.api:app --host "$HOST" --port "$PORT"
