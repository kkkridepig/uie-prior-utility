# C_CONV: single-seed screening

Evidence state: `completed_screen`; added optimizer steps: 5000. Source validation only. All trained branches start independently from the same frozen parent.

Parent SHA-256: `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`; parent training step: 85000. Train/evaluate implementation SHA-256: `983519fc2e1b921ac08b5460b87ebde78dce672fe078f783146d27d6ca643ad2`.

Trainable parameter groups: `{'inherited': 5995762, 'new': 74870, 'physical': 0}`. Migration: `{'loaded_parent_tensors': 465, 'missing_parent_tensors': [], 'shape_mismatches': [], 'semantic_changes': [], 'new_trainable_parameters': 74870}`.

Effective global batch 4 = micro batch 4 x accumulation 1; new parameters and inherited parameters use the recorded separate learning rates.

Recorded device occupation for this branch: 1.106 h (completed dispatcher tasks only; the active task is charged on completion).

| Validation checkpoint | Images | Mean PSNR (dB) | Mean SSIM | Control | Paired PSNR delta (dB) | 95% interval |
|---:|---:|---:|---:|---|---:|---|
| 1000 | 24 | 24.522667 | 0.918342 | BASE_CONT | -0.054352 | [-0.1562, +0.0317] |
| 2000 | 91 | 24.608869 | 0.929205 | BASE_CONT | +0.013335 | [-0.0332, +0.0539] |
| 3000 | 24 | 24.649729 | 0.919208 | BASE_CONT | +0.053677 | [-0.0439, +0.1530] |
| 4000 | 24 | 24.656605 | 0.919307 | BASE_CONT | +0.067506 | [-0.0001, +0.1383] |
| 5000 | 91 | 24.662445 | 0.929642 | BASE_CONT | +0.002308 | [-0.0401, +0.0438] |

Matched comparison uses the same added-step checkpoint and source validation images. The interval resamples fixed-weight validation scenes/images, not training seeds. Any difference also incurs the added training cost shown above.
Checkpoints, train log, per-image metrics and worst cases are in `runs/explore_ag_single_seed_v2_20261003/branches/C_CONV/`.
