# A-G live exploration status

This file reflects dispatcher records, not a completed seven-day experiment. The authoritative budget remains `runs/explore_ag_single_seed_v2_20261003/budget.json`.

Dispatcher status: `screening_finished_with_documented_blocks`; active task: `None`.

Charged device occupation: 14.784 h completed + 0.000 h active = 14.784 h / 168 h. Exploration ceiling: 152 h; final Benchmark reserve: 16 h.

Parent: `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`; evaluation protocol: `ddim:20`.

| Direction/branch | State | Added steps | Full source validation |
|---|---|---:|---|
| BASE_CONT | completed_screen | 5000 | 24.6601 dB / 0.92955 SSIM |
| B_PROXY | completed_screen | 5000 | 24.6147 dB / 0.92875 SSIM |
| A_COUPLED | completed_screen | 5000 | 24.6196 dB / 0.92911 SSIM |
| A_SPLIT | completed_screen | 5000 | 24.6081 dB / 0.92896 SSIM |
| C_CONV | completed_screen | 5000 | 24.6624 dB / 0.92964 SSIM |
| C_FIXED | completed_screen | 5000 | 24.6405 dB / 0.92952 SSIM |
| C_ROUTED | completed_screen | 5000 | 24.6464 dB / 0.92956 SSIM |
| D_SOBEL_CONTROL | completed_screen | 5000 | 24.6378 dB / 0.92949 SSIM |
| D_PHASE | completed_screen | 5000 | 24.6463 dB / 0.92950 SSIM |
| F_UNIFORM | completed_screen | 5000 | 24.6463 dB / 0.92961 SSIM |
| F_ROUTED | completed_screen | 5000 | 24.6478 dB / 0.92963 SSIM |
| B_ADAPT | blocked_data | 0 | pending |
| G_FEATURE | blocked_dependency | 0 | pending |
| G0_G1 | blocked_dependency | 0 | pending |

E0 and each branch with a run configuration have separate `AG_V2_*_RESULTS.md` records in this directory. Missing results remain pending or blocked; no test score is used for model selection.

Supplemental A control recovery: `completed`; source-only stress diagnostics: `completed`. These queues share the device lock and cumulative ledger; their existence does not mean the diagnostics have completed.
