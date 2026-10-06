# CDE V3 execution

Use the existing vendor environment. Never install or replace torch/torchvision.

```bash
cd /mnt/workspace/uie-prior-utility
PYTHONPATH=. scripts/mpa_python.sh scripts/cde_v3/dispatch.py --dry-run
PYTHONPATH=. scripts/mpa_python.sh -m pytest tests/cde_v3 tests/mpa
PYTHONPATH=. scripts/mpa_python.sh scripts/cde_v3/pipeline.py --resume
```

The serial pipeline waits for an existing tracked child instead of duplicating it. A second pipeline cannot acquire `pipeline.lock`. Device work is charged every two seconds and identified by a durable task ID, package and command. Safe task exits resume the same checkpoints. One bounded retry is permitted; CPU operator fallback never changes PPU torch. A crash tail of unknown duration is conservatively charged until its deadline.

Outputs: `runs/prior_utility_cde_v3_20261004`; reports: `docs/experiments/PRIOR_UTILITY_CDE_V3_20261004`. `budget.json`, `dispatch_state.json`, `LIVE_STATUS.md` and per-package `progress.json` are live records. `FINAL_REPORT.md` is written only when the dependency pipeline reaches a terminal state (scientific negative, completed evaluation or documented block).

C weights are adapter-only deltas and optimizer states. Reconstruct with the exact lineage parent SHA and `eval_development.load_branch`. `finetune_seed` controls independent, step-keyed data/time/noise/initialization/mode/dropout. The base continuation uses a fresh Adam with a fixed 5000-update LR endpoint. It is not exact optimizer continuation of the 85000-step parent.

`source_dev` is for selection. `route_fit` is the only label-optimization source. `route_cal` only chooses the fixed mode and optional utility-only deployment threshold. Main CE/shuffle/utility tables use unthresholded argmax with null zero. Historical UIEB/LSUI tests are regression-only. The final evaluator requires a hash freeze before touching the candidate new holdout scores.

Missing required evidence stops advancement; code success and gate thresholds are never claims of originality or publication readiness. E uses standard DDIM and is an engineering/diagnostic contribution only. C is selected for a falsifiable question and contractual repair, not for having the highest previous score. D shallow phase was an allowed previous simplification.
