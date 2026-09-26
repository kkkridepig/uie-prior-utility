#!/usr/bin/env bash
set -euo pipefail
UIE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$UIE_ROOT"
UIE_SOURCE="${1:-UIEB}"
UIE_SEED="${2:-42}"
UIE_DEVICE="${UIE_DEVICE:-cuda}"
UIE_PRECISION="${UIE_PRECISION:-fp32}"
UIE_SAMPLER="${UIE_SAMPLER:-flow}"
UIE_PATCH="${UIE_PATCH:-dense}"
UIE_SIZE="${UIE_SIZE:-256}"
UIE_RUN="outputs/$UIE_SOURCE/seed$UIE_SEED/$UIE_SAMPLER-$UIE_PATCH"
UIE_SOURCE_MANIFEST="data/manifests/$UIE_SOURCE.json"
UIE_CHECKPOINT="$UIE_RUN/utility/best.pt"
UIE_EVAL="$UIE_RUN/suite-size$UIE_SIZE-$UIE_PRECISION"
read -r -a UIE_NFES <<< "${UIE_NFES:-1 2 4 8}"
read -r -a UIE_TARGETS <<< "${UIE_TARGETS:-UIEB LSUI UFO-120 Challenging-60 U45}"
if [[ ! -f "$UIE_CHECKPOINT" ]]; then
  printf 'Missing utility checkpoint: %s\n' "$UIE_CHECKPOINT" >&2
  exit 1
fi
eval_one() {
  local checkpoint="$1" manifest="$2" destination="$3"
  shift 3
  if [[ -f "$destination/summary.json" ]]; then
    printf 'Already evaluated: %s\n' "$destination"
    return
  fi
  python -m uie evaluate --checkpoint "$checkpoint" --manifest "$manifest" \
    --source-manifest "$UIE_SOURCE_MANIFEST" --output "$destination" --device "$UIE_DEVICE" \
    --precision "$UIE_PRECISION" --seed "$UIE_SEED" --size "$UIE_SIZE" "$@"
}
# Each NFE gets its own temperature, fitted using source calibration data only.
for UIE_NFE in "${UIE_NFES[@]}"; do
  UIE_CALIBRATED="$UIE_EVAL/calibrated-nfe$UIE_NFE.pt"
  if [[ ! -f "$UIE_CALIBRATED" ]]; then
    python -m uie calibrate --checkpoint "$UIE_CHECKPOINT" --manifest "$UIE_SOURCE_MANIFEST" \
      --output "$UIE_CALIBRATED" --device "$UIE_DEVICE" --precision "$UIE_PRECISION" \
      --size "$UIE_SIZE" --steps "$UIE_NFE" --seed "$UIE_SEED"
  fi
  for UIE_TARGET in "${UIE_TARGETS[@]}"; do
    UIE_MANIFEST="data/manifests/$UIE_TARGET.json"
    if [[ ! -f "$UIE_MANIFEST" ]]; then
      printf 'Skipped missing manifest: %s\n' "$UIE_MANIFEST"
      continue
    fi
    UIE_EXTRA=()
    case "$UIE_TARGET" in
      UIEB|LSUI|UFO-120) UIE_EXTRA+=(--diagnostics) ;;
    esac
    eval_one "$UIE_CALIBRATED" "$UIE_MANIFEST" "$UIE_EVAL/$UIE_TARGET/nfe$UIE_NFE-utility" \
      --steps "$UIE_NFE" --gate utility "${UIE_EXTRA[@]}"
    if [[ "$UIE_TARGET" == "$UIE_SOURCE" ]]; then
      for UIE_GATE in none all fixed; do
        eval_one "$UIE_CALIBRATED" "$UIE_MANIFEST" "$UIE_EVAL/$UIE_TARGET/nfe$UIE_NFE-$UIE_GATE" \
          --steps "$UIE_NFE" --gate "$UIE_GATE" "${UIE_EXTRA[@]}"
      done
    fi
  done
done
# The baseline is evaluated once per dataset. Clean calibration is not claimed
# for intentionally corrupted priors; use the uncalibrated checkpoint below.
for UIE_TARGET in "${UIE_TARGETS[@]}"; do
  UIE_MANIFEST="data/manifests/$UIE_TARGET.json"
  if [[ -f "$UIE_MANIFEST" ]]; then
    eval_one "$UIE_RUN/baseline/best.pt" "$UIE_MANIFEST" "$UIE_EVAL/$UIE_TARGET/baseline" --save-images
  fi
done
for UIE_CORRUPTION in clean color shift missing invert; do
  for UIE_GATE in none all utility; do
    eval_one "$UIE_CHECKPOINT" "$UIE_SOURCE_MANIFEST" "$UIE_EVAL/stress/$UIE_CORRUPTION-$UIE_GATE" \
      --steps 4 --gate "$UIE_GATE" --corruption "$UIE_CORRUPTION" --diagnostics
  done
done
printf 'Evaluation reports: %s\n' "$UIE_EVAL"
