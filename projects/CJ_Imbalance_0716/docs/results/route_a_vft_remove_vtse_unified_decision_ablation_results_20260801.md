# VFT remove-VTSE unified decision ablation

## Result

This five-seed development-v3 decision ablation directly isolates the
visibility-targeted Ridge-STFT expert (VTSE) in the Q4 six-label policy. Q4
is compared with Q2, which removes the VTSE contribution, and with Q3, a
base-only-head transparency control. The stored non-VFT logits are exactly
unchanged; each arm retains its pre-assigned calibration and
FPR-constrained-threshold roles.

| Q4 aligned VTSE minus comparator | Mean delta | 95% bootstrap CI | Positive seeds |
|---|---:|---:|---:|
| Q2 no-VTSE: VFT weak-vs-negative AUROC | +0.01864 | [+0.01418, +0.02358] | 5/5 |
| Q2 no-VTSE: VFT weak recall | +0.02310 | [+0.01053, +0.03793] | 5/5 |
| Q2 no-VTSE: VFT AUPR | +0.01296 | [+0.01106, +0.01421] | 5/5 |
| Q2 no-VTSE: VFT strong recall | +0.02858 | [+0.02115, +0.03600] | 5/5 |
| Q2 no-VTSE: VFT negative FPR | -0.00097 | [-0.00290, +0.00097] | 1/5 |

Q4-minus-Q3 is numerically identical to Q4-minus-Q2. This is expected from
the stored predictions: the base-only control yields the same VFT score as
the no-VTSE arm. It makes the control useful for transparency, but it is not
same-parameter MLP-capacity evidence.

## Decision

All pre-registered mechanism checks pass: Q4 exceeds Q2 by at least 0.010
weak AUROC in all five seeds, exceeds Q3 by at least 0.005 in all five, has
non-inferior AUPR, remains below VFT FPR 0.20, and leaves all non-VFT logits
exact. The direct conclusion is narrow: the aligned VTSE component supplies
VFT weak-evidence discrimination in the current frozen six-label decision
policy. It is not a new trained candidate, does not repair the external
VFT/ISRJ OOD failure, and does not authorize test-v3.

## Reproducibility and evidence boundary

- Plan: [route_a_vft_remove_vtse_unified_decision_ablation_plan_20260801.md](../plans/route_a_vft_remove_vtse_unified_decision_ablation_plan_20260801.md)
- Runner: `experiments/run_route_a_vft_remove_vtse_unified_decision_ablation.py`
- Summary: `outputs/tapc/route_a_vft_remove_vtse_unified_decision_ablation/route_a_vft_remove_vtse_unified_decision_ablation_summary.json`
- Development-v3 data SHA-256: `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`
- Identity-isolated seeds: `2039, 2040, 2041, 2042, 2043`.

No training, candidate selection, locked-test, test-v2, or test-v3 use was
performed. Every authoritative receipt records `locked_test_used=false`,
`test_v2_used=false`, and `test_v3_used=false`.
