#!/usr/bin/env bash
# Create the project venv from scratch on Linux + RTX 3090 (Ampere, sm_86).
#
# cu126 is the default because it runs on older data-center drivers (>= 525 via CUDA minor-version
# compatibility). If the driver is >= 570 you can use cu128 instead: TORCH_CUDA=cu128 ./scripts/setup_venv_linux.sh
#
# Usage (from anywhere):
#   bash scripts/setup_venv_linux.sh
#   RECREATE=1 bash scripts/setup_venv_linux.sh      # delete and rebuild .venv
set -euo pipefail

PYTHON_VERSION="3.12"
TORCH_VERSION="2.11.0"
TORCH_CUDA="${TORCH_CUDA:-cu126}"
VENV_DIR="${VENV_DIR:-.venv}"
RECREATE="${RECREATE:-0}"
MIN_DRIVER=525

cd "$(dirname "$0")/.."

echo "==> [1/5] Checking NVIDIA driver"
if ! command -v nvidia-smi >/dev/null; then
    echo "nvidia-smi not found. Install the NVIDIA driver (>= $MIN_DRIVER) first." >&2
    exit 1
fi
IFS=',' read -r gpu_name driver cc < <(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -n1)
echo "    GPU:${gpu_name} | driver${driver} | compute capability${cc}"
if (( ${driver%%.*} < MIN_DRIVER )); then
    echo "Driver${driver} is too old for CUDA 12.x wheels. Update the NVIDIA driver to >= $MIN_DRIVER." >&2
    exit 1
fi

echo "==> [2/5] Checking Python $PYTHON_VERSION"
PYTHON_BIN="python$PYTHON_VERSION"
if ! command -v "$PYTHON_BIN" >/dev/null; then
    echo "$PYTHON_BIN not found. On Ubuntu:" >&2
    echo "  sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python$PYTHON_VERSION python$PYTHON_VERSION-venv" >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c "import venv, ensurepip" 2>/dev/null; then
    echo "venv/ensurepip missing. On Ubuntu: sudo apt install python$PYTHON_VERSION-venv" >&2
    exit 1
fi
"$PYTHON_BIN" --version

echo "==> [3/5] Creating venv at $VENV_DIR"
if [[ -e "$VENV_DIR" ]]; then
    if [[ "$RECREATE" != "1" ]]; then
        echo "$VENV_DIR already exists. Re-run with RECREATE=1 to delete and rebuild it." >&2
        exit 1
    fi
    rm -rf "$VENV_DIR"
fi
"$PYTHON_BIN" -m venv "$VENV_DIR"
PY="$VENV_DIR/bin/python"
"$PY" -m pip install --upgrade pip

echo "==> [4/5] Installing torch $TORCH_VERSION+$TORCH_CUDA, then requirements.txt"
"$PY" -m pip install "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
"$PY" -m pip install -r requirements.txt

echo "==> [5/5] Verifying GPU"
"$PY" scripts/check_gpu.py

echo
echo "Done. Activate with:  source $VENV_DIR/bin/activate"
echo "Next: docker compose up -d && python -m rag.ingest && python -m uvicorn rag.api:app --host 0.0.0.0 --port 8000"
