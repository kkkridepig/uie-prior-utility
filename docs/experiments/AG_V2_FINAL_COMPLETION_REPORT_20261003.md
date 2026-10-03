# A-G single-seed exploration: final completion record

Date: 2026-10-03 UTC. Protocol: `explore_ag_single_seed_v2_20261003`.

## Outcome and evidence level

This round is complete under the single-seed A-G protocol. The historical UIEB training was first parked safely at step 225000, then all eligible directions were screened from one frozen parent checkpoint. No new 400000-step job was launched. There is no running PPU training process at close-out.

This is exploratory evidence from one parent and one training seed. It supports neither a general innovation claim nor stability across random initializations. Test results were generated only after the selection freeze; they were not used to choose an experiment.

| Direction | What was implemented and run | Result | Evidence state |
|---|---|---|---|
| P1 | Matched continuation control | BASE_CONT, 5000 steps | completed; continuation improves over parent on both paired tests |
| A | Coupled and split conditioning controls | both 5000 steps | completed; no candidate met the validation gate; coupled branch finished after test freeze |
| B | Proxy direction correction | 5000 steps | completed; worse than matched continuation |
| B_ADAPT | Atlantis depth adaptation | not trained | blocked_data: generator UIEB exposure is unknown and depth is synthetic relative depth |
| C | convolution, fixed route, routed residual controls | 3 x 5000 steps | completed; routed route lacks supported advantage |
| D | Sobel and phase controls | 2 x 5000 steps | completed; small validation difference is insufficient for a candidate |
| E0 | sampler comparison | full 91-image validation | completed; DDIM20 frozen |
| F | uniform and routed source-expert controls | 2 x 5000 steps | completed; no supported advantage |
| G | frozen detector feature/task evaluation | not trained | blocked_dependency: no audited, permitted underwater detector backend |

## Safety, parent, and budget

The parked historical checkpoint is `runs/s2_grouped_v1/uieb/seed_20260927/last.pt`, step 225000, SHA-256 `113f3c964a1151b8d4bbe642e1f24b46527b3cf403998202f4c65f99a86f7c0f`. It was never overwritten.

The common parent is `runs/explore_ag_single_seed_v2_20261003/parent_0.pt`, SHA-256 `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`. All branch checkpoints, logs, state, and the cumulative device-time ledger remain under `runs/explore_ag_single_seed_v2_20261003/`.

Charged PPU time is 14.784 h of the 168 h cap: 1.026 h while safely parking the legacy queue and 13.758 h in new protocol work. The final Benchmark reserve was 16 h; its actual charged time was 0.520 h. This leaves 153.216 h unspent. The accounting is cumulative and persisted in `budget.json`.

| Work package | Charged PPU h |
|---|---:|
| P1 continuation/parent/stress | 1.407 |
| A | 2.231 |
| B | 1.116 |
| C | 3.441 |
| D | 2.295 |
| E sampler | 0.375 |
| F | 2.290 |
| final benchmark | 0.520 |
| profiling | 0.084 |

## Fixed validation protocol and screening result

Selection used only frozen UIEB `grouped_v1` validation (91 images), common parent initialization, matched added training steps, fixed evaluation noise, and DDIM20. E0 established that DDIM20 retained the parent validation quality while reducing median inference from 9.802 s for DDPM1000 to 0.236 s, about 41.5x faster.

The predeclared quality gate was a matched-control PSNR improvement of at least +0.10 dB. The relevant 5000-step differences were:

| Comparison | Delta PSNR (dB) | 95% bootstrap interval | Decision |
|---|---:|---|---|
| B_PROXY - BASE_CONT | -0.045454 | [-0.095436, +0.005098] | reject |
| A_SPLIT - A_COUPLED | -0.011508 | [-0.030912, +0.009047] | reject |
| C_FIXED - C_CONV | -0.021979 | [-0.063701, +0.020207] | reject |
| C_ROUTED - C_FIXED | +0.005954 | [-0.002573, +0.013948] | reject |
| D_PHASE - D_SOBEL_CONTROL | +0.008533 | [+0.000393, +0.016228] | below gate |
| F_ROUTED - F_UNIFORM | +0.001465 | [-0.008555, +0.011921] | reject |

No innovation branch passed. The protocol therefore correctly stopped before the optional 10000-step continuation. Added training is kept separate from method attribution: BASE_CONT is the matched amount-of-training control.

## Frozen final benchmark

`selection_freeze_before_test.json` was written before any final test job. It froze the parent, DDIM20, the UIEB validation-only decision procedure, and the permitted checkpoint hashes. Tests covered 97 paired UIEB images, 427 paired LSUI images, and 45 unpaired U45 images for every allowed representative. U45 has no clean paired references, so PSNR and SSIM are intentionally null.

| Comparison | UIEB test PSNR delta (dB) | LSUI test PSNR delta (dB) | Interpretation |
|---|---:|---:|---|
| BASE_CONT - PARENT | +0.573922, CI [+0.156068, +1.033250] | +0.729533, CI [+0.559428, +0.894910] | continuation is supported relative to its parent |
| B_PROXY - BASE_CONT | -0.015327 | -0.052720 | proxy correction is not supported |
| C_ROUTED - C_FIXED | -0.001613 | -0.001325 | routed residual is not supported |
| D_PHASE - D_SOBEL_CONTROL | +0.007652 | -0.001831 | inconsistent and too small |
| F_ROUTED - F_UNIFORM | +0.006027 | +0.000162 | no supported advantage |

The UIEB paired comparison also has a 96-image scene-overlap sensitivity analysis because a validation/test scene relation was found. It is recorded in `benchmark/paired_summary.json`; this does not change the frozen 97-image result.

`A_COUPLED` completed after the test selection freeze because its original attempt exited with SIGSEGV before a checkpoint existed. One serialized bounded retry completed and is preserved in `control_recovery.json`, but it was not retroactively benchmarked. A's final test comparison is therefore incomplete under the frozen protocol.

## Robustness diagnostics and failures

The supplemental stress protocol used a fixed 24-image subset after the clean validation screen. It perturbs depth, histogram, and high-frequency inputs, including C's added-route-only intervention. Results are diagnostic, not preregistered selection evidence and did not alter the model or benchmark list. The raw records are in `supplemental_stress/`; the reading guide is `AG_V2_SUPPLEMENTAL_STRESS.md`.

The repeated high-frequency noise intervention (`highfreq_noise_003`) caused roughly 0.19-0.23 dB loss for the screened branches. It is a shared vulnerability, not an advantage of a proposed direction. Worst paired UIEB cases under all benchmarked models include `385_img_` and `255_img_`; per-model, per-image values and outputs are retained under `benchmark/<branch>/uieb_test/`.

## Data and dependency blocks

Atlantis was integrity-checked and grouped by its 400 base IDs, but it remains unsuitable for B_ADAPT in this round. Its published generator is documented as trained with UIEB without publishing the contributing IDs, so custom UIEB test exposure cannot be excluded. Its 8-bit MiDaS-derived depth is relative synthetic conditioning rather than independent metric underwater depth.

RUOD/DUO sources were inspected, but dataset-specific image/annotation reuse permission and a frozen, overlap-audited underwater detector are unresolved. No detector AP, no `G_FEATURE`, and no G task conclusion was fabricated. Details and source receipts are in `EXPLORATION_AG_RESOURCE_AND_SPLIT_AUDIT.md` and `resources/`.

## Reproduction and artifacts

Run from `/mnt/workspace/uie-prior-utility` using the preserved PPU environment:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/mpa
PYTHONPATH=. .venv/bin/python scripts/explore_ag/run_e0.py --checkpoint runs/explore_ag_single_seed_v2_20261003/parent_0.pt --output /tmp/e0 --full --solvers ddim:20
PYTHONPATH=. .venv/bin/python scripts/explore_ag/benchmark.py --help
PYTHONPATH=. .venv/bin/python scripts/explore_ag/run_supplemental_stress.py --help
```

To resume research, first obtain a legally reusable detector with a documented training split disjoint from every intended test set, or an independent calibrated depth resource with the same property. Then create a new versioned protocol and selection freeze; do not alter this run's frozen state or append results as if they were part of the original screen.

Detailed contemporaneous records: `AG_V2_FINAL_EVIDENCE.md`, `AG_V2_E0_RESULTS.md`, `AG_V2_*_RESULTS.md`, `AG_V2_SUPPLEMENTAL_STRESS.md`, `EXPLORATION_AG_RESOURCE_AND_SPLIT_AUDIT.md`, `dispatch_state.json`, and `budget.json`.
