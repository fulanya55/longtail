# Route B R1 development aggregate

- Status: **verified_development**
- Run: `E:\LJQ_code\CJ_Inbalance_0724\outputs\route_b_r1_full_development\r1_20260826_122730`
- Locked test used: `False`

## Seed coverage

| Variant | Seeds |
|---|---:|
| B0 | 5 |
| B1 | 5 |
| B2 | 5 |
| B3 | 5 |
| B4 | 5 |

## Per-variant metrics (mean ± SD; deterministic 95% percentile bootstrap CI)

| Variant | Macro AUPR | Macro F1 | Parameter macro NMAE |
|---|---:|---:|---:|
| B0 | 0.9884 ± 0.0015 [0.9873, 0.9898] | 0.9459 ± 0.0040 [0.9431, 0.9494] | NA |
| B1 | 0.9870 ± 0.0009 [0.9863, 0.9877] | 0.9421 ± 0.0046 [0.9386, 0.9459] | 0.5768 ± 0.0125 [0.5663, 0.5860] |
| B2 | 0.9874 ± 0.0016 [0.9861, 0.9886] | 0.9420 ± 0.0048 [0.9381, 0.9453] | 0.6085 ± 0.0216 [0.5930, 0.6266] |
| B3 | 0.9869 ± 0.0014 [0.9857, 0.9881] | 0.9424 ± 0.0045 [0.9396, 0.9465] | 0.6263 ± 0.0189 [0.6124, 0.6423] |
| B4 | 0.9870 ± 0.0009 [0.9864, 0.9877] | 0.9431 ± 0.0043 [0.9407, 0.9470] | 0.6272 ± 0.0261 [0.6046, 0.6457] |

## Gate decision

`{"complete_seed_coverage": true, "locked_test_unused": true, "classification_safety": true, "parameter_improvement": false, "hard_gate_pass": false}`

Only a complete five-seed development run can receive a gate decision; no test-v2 or locked-test file is opened by this script.
