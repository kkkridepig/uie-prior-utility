# F_UNIFORM: single-seed screening

Evidence state: `completed_screen`; added optimizer steps: 5000. Source validation only. All trained branches start independently from the same frozen parent.

Parent SHA-256: `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`; parent training step: 85000. Train/evaluate implementation SHA-256: `983519fc2e1b921ac08b5460b87ebde78dce672fe078f783146d27d6ca643ad2`.

Trainable parameter groups: `{'inherited': 5995762, 'new': 21600, 'physical': 0}`. Migration: `{'loaded_parent_tensors': 465, 'missing_parent_tensors': [], 'shape_mismatches': [], 'semantic_changes': [], 'new_trainable_parameters': 21600}`.

Effective global batch 4 = micro batch 4 x accumulation 1; new parameters and inherited parameters use the recorded separate learning rates.

Recorded device occupation for this branch: 1.128 h (completed dispatcher tasks only; the active task is charged on completion).

| Validation checkpoint | Images | Mean PSNR (dB) | Mean SSIM | Control | Paired PSNR delta (dB) | 95% interval |
|---:|---:|---:|---:|---|---:|---|
| 1000 | 24 | 24.541977 | 0.918352 | BASE_CONT | -0.035042 | [-0.0743, +0.0027] |
| 2000 | 91 | 24.608737 | 0.929194 | BASE_CONT | +0.013203 | [-0.0140, +0.0396] |
| 3000 | 24 | 24.605355 | 0.918774 | BASE_CONT | +0.009303 | [-0.0492, +0.0765] |
| 4000 | 24 | 24.604815 | 0.918857 | BASE_CONT | +0.015716 | [-0.0271, +0.0623] |
| 5000 | 91 | 24.646304 | 0.929609 | BASE_CONT | -0.013832 | [-0.0372, +0.0076] |

Matched comparison uses the same added-step checkpoint and source validation images. The interval resamples fixed-weight validation scenes/images, not training seeds. Any difference also incurs the added training cost shown above.
Checkpoints, train log, per-image metrics and worst cases are in `runs/explore_ag_single_seed_v2_20261003/branches/F_UNIFORM/`.
