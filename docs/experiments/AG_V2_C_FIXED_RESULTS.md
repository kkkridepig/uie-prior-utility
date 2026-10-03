# C_FIXED: single-seed screening

Evidence state: `completed_screen`; added optimizer steps: 5000. Source validation only. All trained branches start independently from the same frozen parent.

Parent SHA-256: `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`; parent training step: 85000. Train/evaluate implementation SHA-256: `983519fc2e1b921ac08b5460b87ebde78dce672fe078f783146d27d6ca643ad2`.

Trainable parameter groups: `{'inherited': 5995762, 'new': 73600, 'physical': 0}`. Migration: `{'loaded_parent_tensors': 465, 'missing_parent_tensors': [], 'shape_mismatches': [], 'semantic_changes': [], 'new_trainable_parameters': 73600}`.

Effective global batch 4 = micro batch 4 x accumulation 1; new parameters and inherited parameters use the recorded separate learning rates.

Recorded device occupation for this branch: 1.117 h (completed dispatcher tasks only; the active task is charged on completion).

| Validation checkpoint | Images | Mean PSNR (dB) | Mean SSIM | Control | Paired PSNR delta (dB) | 95% interval |
|---:|---:|---:|---:|---|---:|---|
| 1000 | 24 | 24.551116 | 0.918355 | C_CONV | +0.028449 | [-0.0683, +0.1292] |
| 2000 | 91 | 24.589416 | 0.929095 | C_CONV | -0.019453 | [-0.0747, +0.0374] |
| 3000 | 24 | 24.606027 | 0.918852 | C_CONV | -0.043702 | [-0.1509, +0.0531] |
| 4000 | 24 | 24.594544 | 0.918725 | C_CONV | -0.062062 | [-0.1345, +0.0150] |
| 5000 | 91 | 24.640466 | 0.929524 | C_CONV | -0.021979 | [-0.0637, +0.0202] |

Matched comparison uses the same added-step checkpoint and source validation images. The interval resamples fixed-weight validation scenes/images, not training seeds. Any difference also incurs the added training cost shown above.
Checkpoints, train log, per-image metrics and worst cases are in `runs/explore_ag_single_seed_v2_20261003/branches/C_FIXED/`.
