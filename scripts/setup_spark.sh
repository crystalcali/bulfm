#!/usr/bin/env bash
# Preserve the CUDA-enabled PyTorch already installed on DGX Spark.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != '--browser-only' ]]; then
  source scripts/ensure_uv.sh
  ensure_uv "$PWD/.tools/uv" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"
  if [[ -n "${BULFM_PYTHON:-}" ]]; then
    PYTHON_BIN=$(command -v "$BULFM_PYTHON") || {
      echo "BULFM_PYTHON not found: $BULFM_PYTHON" >&2; exit 1;
    }
    "$PYTHON_BIN" scripts/check_cuda.py
  else
    DISCOVERY_PYTHON=$(command -v python3 || command -v python || true)
    PYTHON_BIN=''
    if [[ -n "$DISCOVERY_PYTHON" ]]; then
      PYTHON_BIN=$("$DISCOVERY_PYTHON" scripts/find_cuda_python.py) || PYTHON_BIN=''
    fi
  fi
  if [[ -z "$PYTHON_BIN" ]]; then
    command -v nvidia-smi >/dev/null || {
      echo 'NVIDIA driver not found (nvidia-smi). Run this installer on your Spark.' >&2; exit 1;
    }
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo 'No compatible CUDA Python found. Creating an isolated Python 3.12 environment.'
    if [[ -e .venv ]]; then
      VENV_BACKUP=".venv.backup-$(date +%Y%m%d-%H%M%S)-$$"
      mv .venv "$VENV_BACKUP"
      echo "Previous environment preserved at $VENV_BACKUP"
    fi
    "$BULFM_UV" venv --python 3.12 .venv
    PYTHON_BIN="$PWD/.venv/bin/python"
  elif [[ "$PYTHON_BIN" != "$PWD/.venv/bin/python" ]]; then
    # A stale CPU-only/incompatible project venv must not hide the selected working torch.
    if [[ -e .venv ]]; then
      VENV_BACKUP=".venv.backup-$(date +%Y%m%d-%H%M%S)-$$"
      mv .venv "$VENV_BACKUP"
      echo "Previous environment preserved at $VENV_BACKUP"
    fi
    "$BULFM_UV" venv --python "$PYTHON_BIN" --system-site-packages .venv
  fi
  if ! .venv/bin/python scripts/check_cuda.py; then
    # External venv packages are not inherited by a new venv. Install only into our own environment.
    echo 'Installing CUDA PyTorch into BULFM .venv (system Python is unchanged).'
    "$BULFM_UV" pip install --python .venv/bin/python 'torch==2.14.1+cu130' \
      --index-url https://download.pytorch.org/whl/cu130
    .venv/bin/python scripts/check_cuda.py
  fi
  # A constraint prevents uv from replacing the NVIDIA torch build with another wheel.
  TORCH_CONSTRAINT=$(mktemp)
  trap 'rm -f "$TORCH_CONSTRAINT"' EXIT
  .venv/bin/python -c 'import torch; print("torch==" + torch.__version__)' > "$TORCH_CONSTRAINT"
  "$BULFM_UV" pip install --python .venv/bin/python -c "$TORCH_CONSTRAINT" -e '.[train]'
  # find_spec alone misses errors in lazy Transformers imports and inherited system packages.
  .venv/bin/python -m bulfm.diagnostics
fi
[[ -x .venv/bin/python ]] || { echo 'Create the environment first'; exit 1; }
.venv/bin/python -m playwright install chromium
MINIWOB_COMMIT=7fd85d71a4b60325c6585396ec4f48377d049838
mkdir -p vendor
if [[ ! -d vendor/miniwob-plusplus/.git ]]; then
  git clone https://github.com/Farama-Foundation/miniwob-plusplus.git vendor/miniwob-plusplus
  git -C vendor/miniwob-plusplus checkout --detach "$MINIWOB_COMMIT"
fi
ACTUAL=$(git -C vendor/miniwob-plusplus rev-parse HEAD)
if [[ "$ACTUAL" != "$MINIWOB_COMMIT" ]]; then
  echo "MiniWoB commit differs: $ACTUAL. Expected $MINIWOB_COMMIT. Existing checkout was left intact."
  exit 1
fi
.venv/bin/bulfm doctor
echo 'Ready. Run ./start.sh, then /train configs/spark-smoke.yaml'
