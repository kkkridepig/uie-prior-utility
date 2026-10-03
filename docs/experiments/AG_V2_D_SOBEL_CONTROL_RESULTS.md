# D_SOBEL_CONTROL: single-seed screening

Evidence state: `completed_screen`; added optimizer steps: 5000. Source validation only. All trained branches start independently from the same frozen parent.

Parent SHA-256: `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`; parent training step: 85000. Train/evaluate implementation SHA-256: `983519fc2e1b921ac08b5460b87ebde78dce672fe078f783146d27d6ca643ad2`.

Trainable parameter groups: `{'inherited': 5995762, 'new': 10144, 'physical': 0}`. Migration: `{'loaded_parent_tensors': 465, 'missing_parent_tensors': [], 'shape_mismatches': [], 'semantic_changes': [], 'new_trainable_parameters': 10144}`.

Effective global batch 4 = micro batch 4 x accumulation 1; new parameters and inherited parameters use the recorded separate learning rates.

Recorded device occupation for this branch: 1.113 h (completed dispatcher tasks only; the active task is charged on completion).

| Validation checkpoint | Images | Mean PSNR (dB) | Mean SSIM | Control | Paired PSNR delta (dB) | 95% interval |
|---:|---:|---:|---:|---|---:|---|
| 1000 | 24 | 24.584420 | 0.918863 | BASE_CONT | +0.007401 | [-0.0430, +0.0574] |
| 2000 | 91 | 24.587095 | 0.929076 | BASE_CONT | -0.008439 | [-0.0342, +0.0152] |
| 3000 | 24 | 24.604292 | 0.918910 | BASE_CONT | +0.008240 | [-0.0389, +0.0504] |
| 4000 | 24 | 24.592360 | 0.918967 | BASE_CONT | +0.003261 | [-0.0512, +0.0509] |
| 5000 | 91 | 24.637751 | 0.929486 | BASE_CONT | -0.022386 | [-0.0477, +0.0020] |

Matched comparison uses the same added-step checkpoint and source validation images. The interval resamples fixed-weight validation scenes/images, not training seeds. Any difference also incurs the added training cost shown above.
Checkpoints, train log, per-image metrics and worst cases are in `runs/explore_ag_single_seed_v2_20261003/branches/D_SOBEL_CONTROL/`.
