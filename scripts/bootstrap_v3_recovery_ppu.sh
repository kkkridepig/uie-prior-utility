#!/usr/bin/env bash
set -euo pipefail
UIE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UIE_SYSTEM_PYTHON="${UIE_SYSTEM_PYTHON:-python3}"
UIE_VENV="${UIE_VENV:-$UIE_ROOT/.venv}"
UIE_PIP_INDEX="${UIE_PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
cd "$UIE_ROOT"
UIE_VENDOR_ID="$("$UIE_SYSTEM_PYTHON" - <<'PY'
import json, os, torch, torchvision, numpy, scipy
print(json.dumps({name: [module.__version__, os.path.realpath(module.__file__)]
                  for name, module in [('torch', torch), ('torchvision', torchvision),
                                       ('numpy', numpy), ('scipy', scipy)]}, sort_keys=True))
if not torch.cuda.is_available():
    raise SystemExit('Vendor PPU runtime is unavailable; no CPU substitution.')
PY
)"
if [[ ! -x "$UIE_VENV/bin/python" ]]; then
  "$UIE_SYSTEM_PYTHON" -m venv --system-site-packages "$UIE_VENV"
fi
# The earlier Flow package metadata requires Python 3.10. V3 is imported from
# the checkout, without an editable install or replacement of vendor packages.
"$UIE_VENV/bin/python" -m pip install --no-deps -r requirements-recovery-ppu.txt --index-url "$UIE_PIP_INDEX"
UIE_AFTER_ID="$("$UIE_VENV/bin/python" - <<'PY'
import json, os, torch, torchvision, numpy, scipy
print(json.dumps({name: [module.__version__, os.path.realpath(module.__file__)]
                  for name, module in [('torch', torch), ('torchvision', torchvision),
                                       ('numpy', numpy), ('scipy', scipy)]}, sort_keys=True))
import cv2, lpips, libarchive, skimage, pywt, imageio, tifffile, lazy_loader
PY
)"
if [[ "$UIE_VENDOR_ID" != "$UIE_AFTER_ID" ]]; then
  printf 'ERROR: recovery environment changed a protected vendor package.\n' >&2
  exit 1
fi
"$UIE_VENV/bin/python" -m uie doctor --device cuda --precision fp32
printf 'V3 dependencies ready. No experiment or training was started.\n'
printf 'Use: cd "%s" && source "%s/bin/activate"\n' "$UIE_ROOT" "$UIE_VENV"
