# Actual training matrix

| Method | Updates | Selected step | Device hours | Convergence unresolved |
| --- | --- | --- | --- | --- |
| B1 | 4000 | 0 | 0.142448042233785 | False |
| B3 | 4000 | 0 | 0.09762991567452749 | False |
| B4 | not_run | - | - | - |
| G0 | not_run | - | - | - |
| G1 | not_run | - | - | - |
| G2 | not_run | - | - | - |
| F0 | not_run | - | - | - |
| R0 | not_run | - | - | - |
| O | not_run | - | - | - |
| O-NI | not_run | - | - | - |
| O-NP | not_run | - | - | - |
| O-NS | not_run | - | - | - |
| O-ND | not_run | - | - | - |

Seed 20261007 only. B4 initializes from selected B3; its model_fit/utility_fit 4+4 stream matches the additional data sources, not necessarily total exposure or wall time.
B0 uses both frozen output policies; B2 is a calibrated fixed alpha on B1. All utility methods share frozen B1, fit-only scales and paired source/view streams.
Five validation fractions 0/.25/.5/.75/1 and earlier-checkpoint tie rule; candidate/B4 selection=model_val, utility selection=utility_val nominal.
