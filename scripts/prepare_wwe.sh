#!/usr/bin/env bash
set -euo pipefail
UIE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$UIE_ROOT"
UIE_DATA_ROOT="${1:-$UIE_ROOT/data/wwe}"
# Keep the author's test sets; derive validation/calibration only from train.
for UIE_DATASET in UIEB LSUI UFO-120; do
  UIE_EXTRA=()
  if [[ "$UIE_DATASET" == "UFO-120" ]]; then UIE_EXTRA+=(--allow-target-resize); fi
  python -m uie prepare --layout wwe --dataset "$UIE_DATASET" \
    --root "$UIE_DATA_ROOT" --output "data/manifests/$UIE_DATASET.json" \
    --validation-from-train --val-fraction 0.1 --calibration-fraction 0.05 "${UIE_EXTRA[@]}"
done
python - "$UIE_DATA_ROOT" <<'PY'
from pathlib import Path
from types import SimpleNamespace
from uie.prepare import prepare
import sys
root = Path(sys.argv[1]).resolve()
for name in ("Challenging-60", "U45"):
    dirs = [p for p in root.rglob(name) if p.is_dir()]
    if len(dirs) != 1:
        raise SystemExit(f"Expected one {name} directory, got {dirs}; use explicit nonref prepare")
    image_dir = dirs[0] / "test" if (dirs[0] / "test").is_dir() else dirs[0]
    prepare(SimpleNamespace(
        layout="nonref", dataset=name, root=str(root), input=str(image_dir),
        target=None, output=f"data/manifests/{name}.json", seed=42,
        val_fraction=0, test_fraction=0, calibration_fraction=0,
        groups=None, depth=None, validation_from_train=False, allow_target_resize=False))
PY
