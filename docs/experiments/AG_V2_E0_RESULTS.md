# E0: source-validation sampler screening

Same latest recoverable checkpoint, sample-keyed noise and UIEB grouped_v1 validation. This is not test-set evidence. Each reported NFE is the actual denoiser-call count.

| Split | Sampler | Images | PSNR (dB) | SSIM | Median end-to-end (s) |
|---|---|---:|---:|---:|---:|
| fixed 24 | ddpm:1000 | 24 | 25.097316 | 0.923722 | 9.855 |
| fixed 24 | ddim:20 | 24 | 25.097444 | 0.923721 | 0.240 |
| fixed 24 | ddim:50 | 24 | 25.097335 | 0.923722 | 0.534 |
| fixed 24 | ddim:100 | 24 | 25.097547 | 0.923723 | 1.030 |
| full 91 | ddpm:1000 | 91 | 24.648250 | 0.930443 | 9.802 |
| full 91 | ddim:20 | 91 | 24.648291 | 0.930443 | 0.236 |
| full 91 | ddim:50 | 91 | 24.648267 | 0.930443 | 0.531 |
| full 91 | ddim:100 | 91 | 24.648310 | 0.930443 | 1.022 |

E package charged device time so far: 0.375 h. Full validation and parent confirmation must pass before the new sampler is frozen.
Protocol: `runs/explore_ag_single_seed_v2_20261003/e0/eval_protocol_v2.json` (exists only after full validation); per-image records: `e0/full_latest/per_image.jsonl`.
