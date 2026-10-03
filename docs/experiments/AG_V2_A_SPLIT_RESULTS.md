# A_SPLIT: single-seed screening

Evidence state: `completed_screen`; added optimizer steps: 5000. Source validation only. All trained branches start independently from the same frozen parent.

Parent SHA-256: `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`; parent training step: 85000. Train/evaluate implementation SHA-256: `983519fc2e1b921ac08b5460b87ebde78dce672fe078f783146d27d6ca643ad2`.

Trainable parameter groups: `{'inherited': 4192579, 'new': 0, 'physical': 1803195}`. Migration: `{'loaded_parent_tensors': 465, 'missing_parent_tensors': [], 'shape_mismatches': [], 'semantic_changes': ['Beta head copied to independent kappa_D and kappa_B outputs', 'sigmoid to softplus; linear RGB; relative distance; not legacy-equivalent'], 'new_trainable_parameters': 0}`.

Effective global batch 4 = micro batch 4 x accumulation 1; new parameters and inherited parameters use the recorded separate learning rates.

Recorded device occupation for this branch: 1.098 h (completed dispatcher tasks only; the active task is charged on completion).

| Validation checkpoint | Images | Mean PSNR (dB) | Mean SSIM | Control | Paired PSNR delta (dB) | 95% interval |
|---:|---:|---:|---:|---|---:|---|
| 1000 | 24 | 24.600052 | 0.913614 | A_COUPLED | +0.006509 | [-0.0490, +0.0751] |
| 2000 | 91 | 24.574002 | 0.928157 | A_COUPLED | +0.006061 | [-0.0150, +0.0277] |
| 3000 | 24 | 24.636840 | 0.919661 | A_COUPLED | -0.034010 | [-0.0763, +0.0041] |
| 4000 | 24 | 24.639188 | 0.919464 | A_COUPLED | -0.021679 | [-0.0582, +0.0116] |
| 5000 | 91 | 24.608107 | 0.928964 | A_COUPLED | -0.011508 | [-0.0309, +0.0090] |

Matched comparison uses the same added-step checkpoint and source validation images. The interval resamples fixed-weight validation scenes/images, not training seeds. Any difference also incurs the added training cost shown above.
Checkpoints, train log, per-image metrics and worst cases are in `runs/explore_ag_single_seed_v2_20261003/branches/A_SPLIT/`.
