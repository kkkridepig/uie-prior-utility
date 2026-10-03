# B_PROXY: single-seed screening

Evidence state: `completed_screen`; added optimizer steps: 5000. Source validation only. All trained branches start independently from the same frozen parent.

Parent SHA-256: `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`; parent training step: 85000. Train/evaluate implementation SHA-256: `983519fc2e1b921ac08b5460b87ebde78dce672fe078f783146d27d6ca643ad2`.

Trainable parameter groups: `{'inherited': 5995762, 'new': 0, 'physical': 0}`. Migration: `{'loaded_parent_tensors': 465, 'missing_parent_tensors': [], 'shape_mismatches': [], 'semantic_changes': [], 'new_trainable_parameters': 0}`.

Effective global batch 4 = micro batch 4 x accumulation 1; new parameters and inherited parameters use the recorded separate learning rates.

Recorded device occupation for this branch: 1.099 h (completed dispatcher tasks only; the active task is charged on completion).

| Validation checkpoint | Images | Mean PSNR (dB) | Mean SSIM | Control | Paired PSNR delta (dB) | 95% interval |
|---:|---:|---:|---:|---|---:|---|
| 1000 | 24 | 24.694406 | 0.919483 | BASE_CONT | +0.117387 | [+0.0072, +0.2289] |
| 2000 | 91 | 24.561805 | 0.928253 | BASE_CONT | -0.033729 | [-0.0975, +0.0227] |
| 3000 | 24 | 24.677470 | 0.919550 | BASE_CONT | +0.081418 | [+0.0135, +0.1589] |
| 4000 | 24 | 24.686766 | 0.919359 | BASE_CONT | +0.097667 | [+0.0336, +0.1683] |
| 5000 | 91 | 24.614683 | 0.928754 | BASE_CONT | -0.045454 | [-0.0954, +0.0051] |

Matched comparison uses the same added-step checkpoint and source validation images. The interval resamples fixed-weight validation scenes/images, not training seeds. Any difference also incurs the added training cost shown above.
Checkpoints, train log, per-image metrics and worst cases are in `runs/explore_ag_single_seed_v2_20261003/branches/B_PROXY/`.
