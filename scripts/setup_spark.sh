#!/usr/bin/env bash
# Preserve the CUDA-enabled PyTorch already installed on DGX Spark.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${BULFM_PYTHON:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null; then
  PYTHON_BIN=python3
fi
if [[ "${1:-}" != '--browser-only' ]]; then
  source scripts/ensure_uv.sh
  ensure_uv "$PWD/.tools/uv" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"
  "$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
except ImportError:
    raise SystemExit('Use the Python environment with NVIDIA CUDA PyTorch. Set BULFM_PYTHON=/path/to/python')
if sys.version_info[:2] != (3, 12):
    raise SystemExit('Python 3.12 is required. Set BULFM_PYTHON=/path/to/python')
if not torch.cuda.is_available():
    raise SystemExit('CUDA PyTorch is required; this installer will not replace it with a CPU wheel.')
a = torch.randn(32, 32, device='cuda', dtype=torch.bfloat16)
(a @ a).sum().item()
print('CUDA works:', torch.__version__, torch.cuda.get_device_name())
PY
  if [[ ! -x .venv/bin/python ]]; then
    "$BULFM_UV" venv --python "$PYTHON_BIN" --system-site-packages .venv
  fi
  # A constraint prevents uv from replacing the NVIDIA torch build with another wheel.
  TORCH_CONSTRAINT=$(mktemp)
  trap 'rm -f "$TORCH_CONSTRAINT"' EXIT
  .venv/bin/python -c 'import torch; print("torch==" + torch.__version__)' > "$TORCH_CONSTRAINT"
  "$BULFM_UV" pip install --python .venv/bin/python -c "$TORCH_CONSTRAINT" -e '.[train]'
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
