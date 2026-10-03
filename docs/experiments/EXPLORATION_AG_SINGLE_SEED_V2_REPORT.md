# Single-seed A-G exploration report (superseded status snapshot)

This early status snapshot was written before the bounded A_COUPLED recovery and supplemental diagnostics completed. It is retained for chronology only. The authoritative terminal record is `AG_V2_FINAL_COMPLETION_REPORT_20261003.md`; current charged device time is 14.784 / 168 h, and A_COUPLED is `completed_screen` but remains outside the earlier frozen benchmark list.

Historical snapshot state: screening_finished_with_documented_blocks. All figures below are source-validation measurements unless explicitly labeled otherwise.

Common parent: `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`; evaluation sampler: `ddim:20`.

Device hours charged: 13.435 / 168; 16 hours reserved for final Benchmark.

| Direction | Run count | State |
|---|---:|---|
| P1 | 1/1 | completed_screen |
| B | 1/1 | completed_screen |
| A | 1/2 | inconclusive |
| C | 3/3 | completed_screen |
| D | 2/2 | completed_screen |
| F | 2/2 | completed_screen |
| E | 1/1 | completed_screen |
| G | 0/2 | blocked_dependency |

Detailed branch metrics and failures: `runs/explore_ag_single_seed_v2_20261003/branches/`; E0 results: `runs/explore_ag_single_seed_v2_20261003/e0/`.
Final Benchmark coverage: `{'PARENT/uieb_test': 'completed', 'BASE_CONT/uieb_test': 'completed', 'B_PROXY/uieb_test': 'completed', 'A_SPLIT/uieb_test': 'completed', 'C_CONV/uieb_test': 'completed', 'C_FIXED/uieb_test': 'completed', 'C_ROUTED/uieb_test': 'completed', 'D_SOBEL_CONTROL/uieb_test': 'completed', 'D_PHASE/uieb_test': 'completed', 'F_UNIFORM/uieb_test': 'completed', 'F_ROUTED/uieb_test': 'completed', 'PARENT/lsui_test': 'completed', 'BASE_CONT/lsui_test': 'completed', 'B_PROXY/lsui_test': 'completed', 'A_SPLIT/lsui_test': 'completed', 'C_CONV/lsui_test': 'completed', 'C_FIXED/lsui_test': 'completed', 'C_ROUTED/lsui_test': 'completed', 'D_SOBEL_CONTROL/lsui_test': 'completed', 'D_PHASE/lsui_test': 'completed', 'F_UNIFORM/lsui_test': 'completed', 'F_ROUTED/lsui_test': 'completed', 'PARENT/u45': 'completed', 'BASE_CONT/u45': 'completed', 'B_PROXY/u45': 'completed', 'A_SPLIT/u45': 'completed', 'C_CONV/u45': 'completed', 'C_FIXED/u45': 'completed', 'C_ROUTED/u45': 'completed', 'D_SOBEL_CONTROL/u45': 'completed', 'D_PHASE/u45': 'completed', 'F_UNIFORM/u45': 'completed', 'F_ROUTED/u45': 'completed'}`. Missing values are not estimated.
