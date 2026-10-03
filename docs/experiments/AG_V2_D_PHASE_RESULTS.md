# D_PHASE: single-seed screening

Evidence state: `completed_screen`; added optimizer steps: 5000. Source validation only. All trained branches start independently from the same frozen parent.

Parent SHA-256: `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`; parent training step: 85000. Train/evaluate implementation SHA-256: `983519fc2e1b921ac08b5460b87ebde78dce672fe078f783146d27d6ca643ad2`.

Trainable parameter groups: `{'inherited': 5995762, 'new': 10144, 'physical': 0}`. Migration: `{'loaded_parent_tensors': 465, 'missing_parent_tensors': [], 'shape_mismatches': [], 'semantic_changes': [], 'new_trainable_parameters': 10144}`.

Effective global batch 4 = micro batch 4 x accumulation 1; new parameters and inherited parameters use the recorded separate learning rates.

Recorded device occupation for this branch: 1.148 h (completed dispatcher tasks only; the active task is charged on completion).

| Validation checkpoint | Images | Mean PSNR (dB) | Mean SSIM | Control | Paired PSNR delta (dB) | 95% interval |
|---:|---:|---:|---:|---|---:|---|
| 1000 | 24 | 24.577062 | 0.918724 | D_SOBEL_CONTROL | -0.007358 | [-0.0286, +0.0126] |
| 2000 | 91 | 24.602219 | 0.929149 | D_SOBEL_CONTROL | +0.015124 | [+0.0045, +0.0255] |
| 3000 | 24 | 24.592924 | 0.918792 | D_SOBEL_CONTROL | -0.011368 | [-0.0280, +0.0050] |
| 4000 | 24 | 24.599065 | 0.918874 | D_SOBEL_CONTROL | +0.006706 | [-0.0105, +0.0268] |
| 5000 | 91 | 24.646283 | 0.929502 | D_SOBEL_CONTROL | +0.008533 | [+0.0004, +0.0162] |

Matched comparison uses the same added-step checkpoint and source validation images. The interval resamples fixed-weight validation scenes/images, not training seeds. Any difference also incurs the added training cost shown above.
Checkpoints, train log, per-image metrics and worst cases are in `runs/explore_ag_single_seed_v2_20261003/branches/D_PHASE/`.
