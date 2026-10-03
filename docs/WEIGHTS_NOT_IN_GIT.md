# Local weights excluded from Git

This repository intentionally excludes `runs/` and `weights/`. The experiment reports, configuration, exact checkpoint hashes, and reproduction commands are committed, but model binaries and external pretrained files remain local.

GitHub's normal Git transport rejects every file larger than 100 MiB. The following 69 local artifacts exceed that limit and must be downloaded or transferred separately if the full training state is needed.

| Artifact group | Local paths | Number of files | Size per file |
|---|---|---:|---:|
| VGG perceptual feature extractor | `weights/vgg19-dcbb9e9d.pth` | 1 | 574.7 MB |
| Parked UIEB S2 state | `runs/s2_grouped_v1/uieb/seed_20260927/{best,last}.pt` | 2 | 171.5 MB |
| S1 real-prior states | `runs/s1/real_prior_smallfit/{best,last}.pt`, `runs/s1/ppu_336_preflight/last.pt` | 3 | 171.5 MB |
| Exploration common parent | `runs/explore_ag_single_seed_v2_20261003/{parent_0,legacy_step_220000}.pt` | 2 | 171.5 MB |
| Exploration branch states | `runs/explore_ag_single_seed_v2_20261003/branches/{A_COUPLED,A_SPLIT,B_PROXY,BASE_CONT,C_CONV,C_FIXED,C_ROUTED,D_PHASE,D_SOBEL_CONTROL,F_ROUTED,F_UNIFORM}/{step_00000,step_02000,step_05000,best,last}.pt` | 55 | 123.3-172.5 MB |
| Micro-batch profile states | `runs/explore_ag_single_seed_v2_20261003/performance/micro_{1,2,4}/{step_00000,last}.pt` | 6 | 123.3-171.5 MB |

The identical `best`, `last`, and `step_05000` aliases in a branch are separate files on disk. They are listed separately because each must be transferred if those exact paths are required.

`weights/depth_anything_v2_vits.pth` is 99.2 MB, below the hard limit but above GitHub's 50 MiB warning threshold. It is an external pretrained dependency, so it is also intentionally excluded and should be obtained through the documented fetch script instead of committing a copy.

Small synthetic smoke checkpoints and `diagnostics.pt` files are also excluded by repository policy. They are regenerable and are not required to reproduce the final A-G report.

For the exact selected final weights and integrity hashes, use:

- `runs/explore_ag_single_seed_v2_20261003/selection_freeze_before_test.json`
- `runs/explore_ag_single_seed_v2_20261003/budget.json`
- `runs/explore_ag_single_seed_v2_20261003/dispatch_state.json`
- `docs/experiments/AG_V2_FINAL_COMPLETION_REPORT_20261003.md`

The externally shared archive `uie-prior-utility-runs-records-20261003.zip` contains logs and structured records only, never weights.
