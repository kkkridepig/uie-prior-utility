#!/usr/bin/env bash
set -euo pipefail
MPA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MPA_ROOT"
MPA_PYTHON="${MPA_PYTHON:-/usr/local/bin/python3}"
if [[ ! -x .venv/bin/python ]]; then "$MPA_PYTHON" -m venv --system-site-packages .venv; fi
.venv/bin/python - <<'PY'
import torch,numpy
assert torch.__version__=='2.0.0a0+nv2303', 'Use server vendor torch; do not pip install torch'
assert torch.__file__.startswith('/usr/local/lib/python3.8/site-packages/torch/')
assert numpy.__version__=='1.23.5'
print(torch.__version__,torch.__file__,torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
# Optional only when cv2 is absent. --no-deps cannot replace torch/numpy.
if ! .venv/bin/python -c 'import cv2' >/dev/null 2>&1; then
 .venv/bin/python -m pip install --no-deps opencv-python-headless==4.8.1.78 --index-url https://mirrors.aliyun.com/pypi/simple/
fi
printf 'Use .venv/bin/python -m mpa_diff.cli.<command> from %s\n' "$MPA_ROOT"
