# F_ROUTED: single-seed screening

Evidence state: `completed_screen`; added optimizer steps: 5000. Source validation only. All trained branches start independently from the same frozen parent.

Parent SHA-256: `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`; parent training step: 85000. Train/evaluate implementation SHA-256: `983519fc2e1b921ac08b5460b87ebde78dce672fe078f783146d27d6ca643ad2`.

Trainable parameter groups: `{'inherited': 5995762, 'new': 31594, 'physical': 0}`. Migration: `{'loaded_parent_tensors': 465, 'missing_parent_tensors': [], 'shape_mismatches': [], 'semantic_changes': [], 'new_trainable_parameters': 31594}`.

Effective global batch 4 = micro batch 4 x accumulation 1; new parameters and inherited parameters use the recorded separate learning rates.

Recorded device occupation for this branch: 1.126 h (completed dispatcher tasks only; the active task is charged on completion).

| Validation checkpoint | Images | Mean PSNR (dB) | Mean SSIM | Control | Paired PSNR delta (dB) | 95% interval |
|---:|---:|---:|---:|---|---:|---|
| 1000 | 24 | 24.543896 | 0.918367 | F_UNIFORM | +0.001919 | [-0.0192, +0.0216] |
| 2000 | 91 | 24.596391 | 0.929249 | F_UNIFORM | -0.012346 | [-0.0322, +0.0066] |
| 3000 | 24 | 24.610477 | 0.918794 | F_UNIFORM | +0.005122 | [-0.0242, +0.0374] |
| 4000 | 24 | 24.620885 | 0.918894 | F_UNIFORM | +0.016070 | [-0.0032, +0.0372] |
| 5000 | 91 | 24.647770 | 0.929625 | F_UNIFORM | +0.001465 | [-0.0086, +0.0119] |

Matched comparison uses the same added-step checkpoint and source validation images. The interval resamples fixed-weight validation scenes/images, not training seeds. Any difference also incurs the added training cost shown above.
Checkpoints, train log, per-image metrics and worst cases are in `runs/explore_ag_single_seed_v2_20261003/branches/F_ROUTED/`.
