#!/usr/bin/env bash
set -euo pipefail
UIE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UIE_SYSTEM_PYTHON="${UIE_SYSTEM_PYTHON:-python3}"
UIE_VENV="${UIE_VENV:-$UIE_ROOT/environments/ppu}"
UIE_PIP_INDEX="${UIE_PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
cd "$UIE_ROOT"
# Probe inherited torch before installing anything. No global pip/runtime changes.
UIE_BASE_TORCH="$("$UIE_SYSTEM_PYTHON" -c 'import os,torch; print(os.path.realpath(torch.__file__))')"
"$UIE_SYSTEM_PYTHON" -c 'import torch; print("system torch:",torch.__version__,torch.__file__); print("CUDA-compatible devices:",torch.cuda.device_count())'
if [[ ! -x "$UIE_VENV/bin/python" ]]; then
  "$UIE_SYSTEM_PYTHON" -m venv --system-site-packages "$UIE_VENV"
fi
"$UIE_VENV/bin/python" -m pip install -r requirements-ppu.txt --index-url "$UIE_PIP_INDEX"
"$UIE_VENV/bin/python" -m pip install --no-deps --no-build-isolation -e .
UIE_AFTER_TORCH="$("$UIE_VENV/bin/python" -c 'import os,torch; print(os.path.realpath(torch.__file__))')"
if [[ "$UIE_BASE_TORCH" != "$UIE_AFTER_TORCH" ]]; then
  printf 'ERROR: venv shadows the system torch. Inspect this venv before training.\n' >&2
  exit 1
fi
"$UIE_VENV/bin/python" -m uie doctor --device cuda --precision fp32
printf 'Activate with: source "%s/bin/activate"\n' "$UIE_VENV"
printf 'Then run BF16 doctor and the PPU smoke commands in README.md.\n'
