# C_ROUTED: single-seed screening

Evidence state: `completed_screen`; added optimizer steps: 5000. Source validation only. All trained branches start independently from the same frozen parent.

Parent SHA-256: `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`; parent training step: 85000. Train/evaluate implementation SHA-256: `983519fc2e1b921ac08b5460b87ebde78dce672fe078f783146d27d6ca643ad2`.

Trainable parameter groups: `{'inherited': 5995762, 'new': 74896, 'physical': 0}`. Migration: `{'loaded_parent_tensors': 465, 'missing_parent_tensors': [], 'shape_mismatches': [], 'semantic_changes': [], 'new_trainable_parameters': 74896}`.

Effective global batch 4 = micro batch 4 x accumulation 1; new parameters and inherited parameters use the recorded separate learning rates.

Recorded device occupation for this branch: 1.120 h (completed dispatcher tasks only; the active task is charged on completion).

| Validation checkpoint | Images | Mean PSNR (dB) | Mean SSIM | Control | Paired PSNR delta (dB) | 95% interval |
|---:|---:|---:|---:|---|---:|---|
| 1000 | 24 | 24.547446 | 0.918325 | C_FIXED | -0.003670 | [-0.0503, +0.0259] |
| 2000 | 91 | 24.598015 | 0.929193 | C_FIXED | +0.008599 | [-0.0033, +0.0212] |
| 3000 | 24 | 24.615223 | 0.918911 | C_FIXED | +0.009196 | [-0.0070, +0.0261] |
| 4000 | 24 | 24.623776 | 0.918937 | C_FIXED | +0.029232 | [+0.0147, +0.0447] |
| 5000 | 91 | 24.646420 | 0.929559 | C_FIXED | +0.005954 | [-0.0026, +0.0139] |

Matched comparison uses the same added-step checkpoint and source validation images. The interval resamples fixed-weight validation scenes/images, not training seeds. Any difference also incurs the added training cost shown above.
Checkpoints, train log, per-image metrics and worst cases are in `runs/explore_ag_single_seed_v2_20261003/branches/C_ROUTED/`.
