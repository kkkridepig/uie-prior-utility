#!/usr/bin/env bash
set -euo pipefail
MPA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MPA_ROOT"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
exec "$MPA_ROOT/.venv/bin/python" "$@"
