# A_COUPLED: single-seed screening

Evidence state: `completed_screen`; added optimizer steps: 5000. Source validation only. All trained branches start independently from the same frozen parent.

Parent SHA-256: `3f8e7b50834d069d85b4f9b57c964242554aad45f44826ef4fc000491fad4bea`; parent training step: 85000. Train/evaluate implementation SHA-256: `983519fc2e1b921ac08b5460b87ebde78dce672fe078f783146d27d6ca643ad2`.

Trainable parameter groups: `{'inherited': 4192579, 'new': 0, 'physical': 1803183}`. Migration: `{'loaded_parent_tensors': 465, 'missing_parent_tensors': [], 'shape_mismatches': [], 'semantic_changes': ['sigmoid to softplus; linear RGB; relative distance; not legacy-equivalent'], 'new_trainable_parameters': 0}`.

Effective global batch 4 = micro batch 4 x accumulation 1; new parameters and inherited parameters use the recorded separate learning rates.

Recorded device occupation for this branch: 1.099 h (completed dispatcher tasks only; the active task is charged on completion).

| Validation checkpoint | Images | Mean PSNR (dB) | Mean SSIM | Control | Paired PSNR delta (dB) | 95% interval |
|---:|---:|---:|---:|---|---:|---|
| 1000 | 24 | 24.593543 | 0.913166 | BASE_CONT | +0.016524 | [-0.2303, +0.2888] |
| 2000 | 91 | 24.567941 | 0.928226 | BASE_CONT | -0.027592 | [-0.0970, +0.0476] |
| 3000 | 24 | 24.670850 | 0.919968 | BASE_CONT | +0.074799 | [-0.0329, +0.1833] |
| 4000 | 24 | 24.660867 | 0.919679 | BASE_CONT | +0.071768 | [-0.0349, +0.1838] |
| 5000 | 91 | 24.619616 | 0.929105 | BASE_CONT | -0.040521 | [-0.0948, +0.0191] |

Matched comparison uses the same added-step checkpoint and source validation images. The interval resamples fixed-weight validation scenes/images, not training seeds. Any difference also incurs the added training cost shown above.
Checkpoints, train log, per-image metrics and worst cases are in `runs/explore_ag_single_seed_v2_20261003/branches/A_COUPLED/`.
