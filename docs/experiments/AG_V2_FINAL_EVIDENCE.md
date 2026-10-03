# A-G single-seed evidence at dispatcher stop

Terminal state: `screening_finished_with_documented_blocks`. Charged single-device time: 14.784 / 168 h; the final Benchmark has a separate 16 h reserve.

This is one shared-parent exploratory seed. An implemented branch, a passed CPU test, a PPU training run, and a supported improvement are distinct evidence levels.

| Branch | Status | Added steps | 5000-step validation PSNR | Limitation |
|---|---|---:|---:|---|
| BASE_CONT | completed_screen | 5000 | 24.6601 |  |
| B_PROXY | completed_screen | 5000 | 24.6147 |  |
| A_COUPLED | completed_screen | 5000 | 24.6196 |  |
| A_SPLIT | completed_screen | 5000 | 24.6081 |  |
| C_CONV | completed_screen | 5000 | 24.6624 |  |
| C_FIXED | completed_screen | 5000 | 24.6405 |  |
| C_ROUTED | completed_screen | 5000 | 24.6464 |  |
| D_SOBEL_CONTROL | completed_screen | 5000 | 24.6378 |  |
| D_PHASE | completed_screen | 5000 | 24.6463 |  |
| F_UNIFORM | completed_screen | 5000 | 24.6463 |  |
| F_ROUTED | completed_screen | 5000 | 24.6478 |  |
| B_ADAPT | blocked_data | 0 | not measured | No verified depth labels and original-scene split available |
| G_FEATURE | blocked_dependency | 0 | not measured | No licensed, validated frozen underwater detector backend available |
| G0_G1 | blocked_dependency | 0 | not measured | G detector prerequisite unresolved |

Completed training screens: 11/14 planned branch/control runs. E0 is separate and requires no added training.

Selected parent: `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`; fixed sampler: `ddim:20`. Selection used UIEB validation only.

Per-branch training, migration, matched-step validation deltas, and cost: `docs/experiments/AG_V2_*_RESULTS.md`. Raw checkpoints, per-image records, logs and the cumulative ledger: `runs/explore_ag_single_seed_v2_20261003/`.

## Frozen Benchmark

Frozen test selection: `yes`. Test and cross-dataset numbers below are never used to change the selected model.

| Model/dataset | Coverage | Images | PSNR (dB) | SSIM | Median end-to-end (s) |
|---|---|---:|---:|---:|---:|
| A_SPLIT/lsui_test | completed | 427 | 22.2209 | 0.87381 | 0.172 |
| A_SPLIT/u45 | completed | 45 | null | null | 0.188 |
| A_SPLIT/uieb_test | completed | 97 | 24.9382 | 0.93669 | 0.180 |
| BASE_CONT/lsui_test | completed | 427 | 22.2396 | 0.87392 | 0.171 |
| BASE_CONT/u45 | completed | 45 | null | null | 0.189 |
| BASE_CONT/uieb_test | completed | 97 | 24.9539 | 0.93704 | 0.183 |
| B_PROXY/lsui_test | completed | 427 | 22.1868 | 0.87359 | 0.173 |
| B_PROXY/u45 | completed | 45 | null | null | 0.162 |
| B_PROXY/uieb_test | completed | 97 | 24.9386 | 0.93680 | 0.180 |
| C_CONV/lsui_test | completed | 427 | 22.2591 | 0.87386 | 0.173 |
| C_CONV/u45 | completed | 45 | null | null | 0.163 |
| C_CONV/uieb_test | completed | 97 | 24.9042 | 0.93682 | 0.184 |
| C_FIXED/lsui_test | completed | 427 | 22.2101 | 0.87368 | 0.192 |
| C_FIXED/u45 | completed | 45 | null | null | 0.198 |
| C_FIXED/uieb_test | completed | 97 | 24.9254 | 0.93694 | 0.205 |
| C_ROUTED/lsui_test | completed | 427 | 22.2088 | 0.87366 | 0.197 |
| C_ROUTED/u45 | completed | 45 | null | null | 0.188 |
| C_ROUTED/uieb_test | completed | 97 | 24.9238 | 0.93694 | 0.213 |
| D_PHASE/lsui_test | completed | 427 | 22.2557 | 0.87389 | 0.170 |
| D_PHASE/u45 | completed | 45 | null | null | 0.190 |
| D_PHASE/uieb_test | completed | 97 | 24.9382 | 0.93685 | 0.173 |
| D_SOBEL_CONTROL/lsui_test | completed | 427 | 22.2575 | 0.87411 | 0.171 |
| D_SOBEL_CONTROL/u45 | completed | 45 | null | null | 0.171 |
| D_SOBEL_CONTROL/uieb_test | completed | 97 | 24.9305 | 0.93685 | 0.177 |
| F_ROUTED/lsui_test | completed | 427 | 22.2573 | 0.87395 | 0.196 |
| F_ROUTED/u45 | completed | 45 | null | null | 0.187 |
| F_ROUTED/uieb_test | completed | 97 | 24.9095 | 0.93685 | 0.210 |
| F_UNIFORM/lsui_test | completed | 427 | 22.2572 | 0.87392 | 0.191 |
| F_UNIFORM/u45 | completed | 45 | null | null | 0.184 |
| F_UNIFORM/uieb_test | completed | 97 | 24.9035 | 0.93686 | 0.202 |
| PARENT/lsui_test | completed | 427 | 21.5100 | 0.86687 | 0.167 |
| PARENT/u45 | completed | 45 | null | null | 0.182 |
| PARENT/uieb_test | completed | 97 | 24.3800 | 0.93002 | 0.177 |

Paired per-image differences, improvement fractions, 1000-draw intervals, and the UIEB 96-image scene-overlap sensitivity are in `runs/explore_ag_single_seed_v2_20261003/benchmark/paired_summary.json` when available.

Atlantis is a licensed synthetic relative-depth candidate, not measured underwater geometry. RUOD dataset permission and an audited underwater detector remain unresolved; no G AP is reported without a real backend and annotations. LPIPS and audited no-reference metrics remain null if their weights/formulas are unavailable. A single training seed cannot establish initialization stability.
