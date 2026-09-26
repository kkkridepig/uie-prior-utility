#!/usr/bin/env bash
set -euo pipefail
UIE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$UIE_ROOT"
UIE_DATASET="${1:-UIEB}"
UIE_SEED="${2:-42}"
UIE_DEVICE="${UIE_DEVICE:-cuda}"
UIE_PRECISION="${UIE_PRECISION:-fp32}"
UIE_SAMPLER="${UIE_SAMPLER:-flow}"
UIE_PATCH="${UIE_PATCH:-dense}"
UIE_BATCH="${UIE_BATCH:-8}"
UIE_WORKERS="${UIE_WORKERS:-4}"
UIE_RUN="outputs/$UIE_DATASET/seed$UIE_SEED/$UIE_SAMPLER-$UIE_PATCH"
UIE_MANIFEST="data/manifests/$UIE_DATASET.json"
UIE_COMMON=(--manifest "$UIE_MANIFEST" --seed "$UIE_SEED" --device "$UIE_DEVICE" \
  --precision "$UIE_PRECISION" --sampler "$UIE_SAMPLER" --patch-mode "$UIE_PATCH" --workers "$UIE_WORKERS")
run_stage() {
  local stage="$1"
  shift
  local resume_args=()
  if [[ -f "$UIE_RUN/$stage/last.pt" ]]; then resume_args=(--resume "$UIE_RUN/$stage/last.pt"); fi
  python -m uie train --config "configs/$stage.yaml" "${UIE_COMMON[@]}" \
    --output "$UIE_RUN/$stage" "$@" "${resume_args[@]}"
}
run_stage baseline --batch-size "$UIE_BATCH"
run_stage flow --batch-size "$UIE_BATCH" --init "$UIE_RUN/baseline/best.pt"
run_stage utility --init "$UIE_RUN/flow/best.pt"
if [[ ! -f "$UIE_RUN/utility/calibrated.pt" ]]; then
  python -m uie calibrate --checkpoint "$UIE_RUN/utility/best.pt" \
    --manifest "$UIE_MANIFEST" --output "$UIE_RUN/utility/calibrated.pt" \
    --device "$UIE_DEVICE" --precision "$UIE_PRECISION" --seed "$UIE_SEED"
fi
if [[ ! -f "$UIE_RUN/evaluation/summary.json" ]]; then
  python -m uie evaluate --checkpoint "$UIE_RUN/utility/calibrated.pt" \
    --manifest "$UIE_MANIFEST" --output "$UIE_RUN/evaluation" \
    --device "$UIE_DEVICE" --precision "$UIE_PRECISION" --seed "$UIE_SEED" --diagnostics --save-images
fi
