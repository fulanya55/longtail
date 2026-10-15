# Route A: VFT fusion capacity-matched MLP ablation results

## Question and protocol

This five-seed development-v3 ablation tested whether the frozen VFT fusion
decision was insensitive to nonlinear head capacity.  Both arms used the same
Q2 H1+B2+MFT-guard source, the same VFT Ridge-STFT evidence, fit weak-positive
weight 4, fit-only normalization, per-arm calibration and FPR <= 0.20
threshold roles.  The `Linear6FixedCapacity` arm had six fixed inputs and one
bias; `MLP1CapacityControl` was `4 -> 1 -> 1` with Tanh.  Each had exactly
seven trainable parameters.  No locked test, test-v2, or test-v3 was used.

## Result: capacity is a live confound

MLP1 minus Linear6FixedCapacity gives:

| VFT metric | Mean delta | paired bootstrap 95% CI | positive seeds |
|---|---:|---:|---:|
| weak-vs-negative AUROC | +0.031020 | [+0.019306, +0.042188] | 5/5 |
| AUPR | +0.013097 | [+0.004794, +0.021524] | 5/5 |
| weak recall | +0.027263 | [+0.008081, +0.042890] | 4/5 |
| strong recall | +0.027313 | [+0.008497, +0.042950] | 4/5 |
| negative FPR | +0.003226 | [-0.005645, +0.013710] | 3/5 |

The preregistered rejection condition for a capacity explanation fails: MLP
has a material, consistently positive ranking gain.  This experiment is a
negative result for the claim that the prior VFT decision gain can be explained
without considering nonlinear decision capacity.  It does **not** select the
MLP as a new candidate and does not change the external-OOD failure boundary.

## Consequence

The appropriate follow-up is not a larger head.  It is the fixed-capacity
three-arm physical-information test: the same MLP with no VTSE, aligned VTSE,
and role-wise shuffled VTSE inputs.  That test separates nonlinear capacity
from sample-aligned physical evidence.

## Artifacts and boundary

- Preregistration: `docs/plans/route_a_vft_capacity_matched_mlp_unified_decision_ablation_plan_20260801.md`
- Runner: `experiments/run_route_a_vft_capacity_matched_mlp_unified_decision_ablation.py`
- Tests: `tests/test_route_a_vft_capacity_matched_mlp_unified_decision_ablation.py` (3 passed)
- Summary SHA-256: `A7F74E8B7AF1EAE0AEB4C46F79679703C8CABAAA5B09427E85126212BBCA4412`
- `locked_test_used=false`, `test_v2_used=false`, `test_v3_used=false`.
