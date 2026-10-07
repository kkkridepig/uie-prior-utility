# Mathematical and integration tests

| Receipt | Tests | Failures | Errors |
| --- | --- | --- | --- |
| /mnt/workspace/uie-prior-utility/runs/ssuie_local_utility_v1_20261007/tests/pytest_acceptance.xml | 135 | 0 | 0 |
| /mnt/workspace/uie-prior-utility/runs/ssuie_local_utility_v1_20261007/tests/pytest_delivery.xml | 140 | 0 | 0 |
| /mnt/workspace/uie-prior-utility/runs/ssuie_local_utility_v1_20261007/tests/pytest_final.xml | 2 | 0 | 2 |
| /mnt/workspace/uie-prior-utility/runs/ssuie_local_utility_v1_20261007/tests/pytest_official_dispatch.xml | 60 | 0 | 0 |
| /mnt/workspace/uie-prior-utility/runs/ssuie_local_utility_v1_20261007/tests/pytest_official_dispatch_final.xml | 60 | 0 | 0 |
| /mnt/workspace/uie-prior-utility/runs/ssuie_local_utility_v1_20261007/tests/pytest_official_resume_cpu.xml | 57 | 0 | 0 |

Real official-backbone integration receipt:
```json
{
  "passed": true,
  "official_public_simplified_backbone_used": true,
  "paper_complete_model": false,
  "device": "PPU-ZW810E",
  "candidate_updates": 100,
  "utility_updates": 100,
  "formal_training_updates": 0,
  "training_samples_per_role": 8,
  "candidate_initial_loss_mean10": 0.0009365305828396231,
  "candidate_final_loss_mean10": 0.0008106040651910007,
  "utility_initial_loss_mean10": 0.5409036636352539,
  "utility_final_loss_mean10": 0.3795450747013092,
  "candidate_parameter_update": true,
  "utility_parameter_update": true,
  "T27_backbone_parameters_and_BN_unchanged": true,
  "T28_candidate_unchanged_during_utility": true,
  "cache_live_sample_count": 16,
  "cache_live_max_abs_errors": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "fresh_process_recovery_max_parameter_error": 0.0,
  "fresh_process_recovery_same_scheduler": true,
  "commands": [
    {
      "argv": [
        "/mnt/workspace/uie-prior-utility/.venv/bin/python",
        "-B",
        "-m",
        "uie_next.real_resume_probe",
        "continuous",
        "--directory",
        "/mnt/workspace/uie-prior-utility/runs/ssuie_local_utility_v1_20261007/tests/real_integration"
      ],
      "exit_code": 0
    },
    {
      "argv": [
        "/mnt/workspace/uie-prior-utility/.venv/bin/python",
        "-B",
        "-m",
        "uie_next.real_resume_probe",
        "first",
        "--directory",
        "/mnt/workspace/uie-prior-utility/runs/ssuie_local_utility_v1_20261007/tests/real_integration"
      ],
      "exit_code": 0
    },
    {
      "argv": [
        "/mnt/workspace/uie-prior-utility/.venv/bin/python",
        "-B",
        "-m",
        "uie_next.real_resume_probe",
        "resume",
        "--directory",
        "/mnt/workspace/uie-prior-utility/runs/ssuie_local_utility_v1_20261007/tests/real_integration"
      ],
      "exit_code": 0
    }
  ],
  "fixture_sha256": "fefcb62cbb243836390c92346819b4728f852a4eae08c6d418aa40cbbabc570f",
  "sealed_eval_accessed": false
}
```
Temporary smoke/profile heads are disposable engineering artifacts, not formal method checkpoints.
