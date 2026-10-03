# C minimum screening

Protocol: `explore_ag_single_seed_v2_20261003`; common parent and sampler are in `runs/explore_ag_single_seed_v2_20261003/`.

| Branch | Added steps | Validation PSNR | Validation SSIM | Evidence |
|---|---:|---:|---:|---|
| C_CONV | 5000 | 24.662444774819456 | 0.9296423119822269 | completed_screen |
| C_FIXED | 5000 | 24.64046588392171 | 0.9295237590126844 | completed_screen |
| C_ROUTED | 5000 | 24.646420220393637 | 0.929559141742737 | completed_screen |

Only validation data were used for this screening. Per-image records and checkpoint hashes reside under the branch run directories. Different achieved steps are not a matched ablation.
