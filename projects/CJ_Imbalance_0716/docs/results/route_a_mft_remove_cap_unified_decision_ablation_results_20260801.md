# MFT residual-cap removal unified-decision ablation

## Result

This five-seed replay places the MFT peak-count residual back into the current
Q4 six-label decision. `Q4_NaturalCap100` changes only the MFT residual scale
from `0.75` to `1.0`; the negative-only direction, uncertainty multiplier,
`tanh` natural bound, VFT head, and all non-MFT logits are retained. Each arm
uses Platt calibration on the assigned `calibration` role and a weak-recall
threshold chosen on the assigned `threshold` role.

| Natural cap 1.0 minus cap 0.75 | Mean delta | 95% bootstrap CI | Positive seeds |
|---|---:|---:|---:|
| MFT weak-vs-negative AUROC | +0.00598 | [+0.00514, +0.00665] | 5/5 |
| MFT weak recall | +0.00802 | [+0.00337, +0.01244] | 4/5 |
| MFT AUPR | +0.00139 | [+0.00114, +0.00157] | 5/5 |
| MFT negative FPR | -0.00065 | [-0.00694, +0.00677] | 2/5 |
| MFT weak-vs-order>=4-negative AUROC | +0.00416 | [+0.00336, +0.00483] | 5/5 |
| MFT order>=4-negative FPR | +0.00250 | [-0.00750, +0.01333] | 2/5 |
| Macro AUPR | +0.00023 | [+0.00019, +0.00026] | 5/5 |
| Macro F1 | +0.00036 | [+0.00008, +0.00063] | 4/5 |

The stored cap-0.75 reconstruction is exact to at most `1e-6` in every seed
and role. All non-MFT logits are exact. Thus the ranking gains are a genuine
scale sensitivity of the MFT residual, not an interaction with the VTSE head
or another label.

## Pre-registered failure and interpretation

The protocol-integrity gate is **not passed**. The registration incorrectly
required the threshold-role FPR cap to also hold on the independent `early`
role. The unchanged Q4 cap-0.75 arm has MFT early negative FPR `0.20161` in
seed 2042, while the other four seeds are `0.18145`, `0.18226`, `0.17581`, and
`0.18871`. Threshold selection itself still enforces FPR<=0.20 on its assigned
threshold role by construction; it does not promise that the independent
early role will be below the same value.

This is a protocol diagnosis, not a post-hoc relaxation: the registered gate
remains failed. Consequently the experiment cannot support a cap change,
cannot close core-14 item 8, and cannot be used to modify the candidate. Its
only admissible conclusion is that `0.75` is not the sole explanation for MFT
ranking and that the next fresh protocol must separate threshold-role
feasibility from independent-role FPR generalization in advance.

## Reproducibility and boundary

- Plan: [route_a_mft_remove_cap_unified_decision_ablation_plan_20260801.md](../plans/route_a_mft_remove_cap_unified_decision_ablation_plan_20260801.md)
- Runner: `experiments/run_route_a_mft_remove_cap_unified_decision_ablation.py`
- Summary: `outputs/tapc/route_a_mft_remove_cap_unified_decision_ablation/route_a_mft_remove_cap_unified_decision_ablation_summary.json`
- Development-v3 data SHA-256: `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`
- Identity-isolated seeds: `2039, 2040, 2041, 2042, 2043`.

No training, candidate selection, locked-test, test-v2, or test-v3 use was
performed. The receipt records `locked_test_used=false`, `test_v2_used=false`,
and `test_v3_used=false`.
