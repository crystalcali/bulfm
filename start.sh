#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/bulfm ]]; then
  echo 'First run: bash scripts/setup_spark.sh'
  exit 1
fi
exec .venv/bin/bulfm "$@"
