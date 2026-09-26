#!/usr/bin/env bash
set -euo pipefail
UIE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$UIE_ROOT"
UIE_DATA_ROOT="${1:-$UIE_ROOT/data/uieb}"
python - "$UIE_DATA_ROOT" <<'PY'
from pathlib import Path
from uie.cli import main
import sys
root = Path(sys.argv[1]).resolve()
def locate(name):
    candidates = [p for p in (root, *root.rglob(name)) if p.is_dir() and p.name == name]
    if len(candidates) != 1:
        raise SystemExit(f"Expected one {name} directory under {root}; got {candidates}. "
                         "Use explicit uie prepare --input/--target if archive layout differs.")
    return str(candidates[0])
main(["prepare", "--layout", "pairs", "--dataset", "UIEB", "--root", str(root),
      "--input", locate("raw-890"), "--target", locate("reference-890"),
      "--output", "data/manifests/UIEB.json", "--seed", "42",
      "--val-fraction", "0.1", "--test-fraction", "0.1", "--calibration-fraction", "0.05"])
PY
