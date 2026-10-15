# Route A: fixed-capacity MLP VFT physical-evidence control results

## Question and protocol

After the capacity-matched study found a nonlinear-head effect, this five-seed
development-v3 experiment held the head itself fixed: all three arms used the
same 7-parameter `Linear(4,1) -> Tanh -> Linear(1,1)` MLP, Q2 source logits,
fit weight 4 for weak VFT positives, optimizer, early selection, calibration,
and FPR <= 0.20 threshold procedure.  Only its four deterministic inputs
differed: no VTSE base transforms, aligned Ridge-STFT evidence, or seeded
role-wise shuffled Ridge-STFT evidence.  Non-VFT logits were exactly preserved.

## Pre-registered gate

The point-estimate gate passes all eight checks.  In particular, aligned VTSE
has weak-AUROC gain >= 0.005, nonnegative AUPR, and at least four positive seeds
against both controls.  This is valid *development mechanism evidence*, not a
candidate promotion or a test-v3 authorization.

| Aligned VTSE comparison | VFT weak-AUROC mean [95% CI] | seeds | VFT AUPR mean [95% CI] |
|---|---:|---:|---:|
| minus no-VTSE MLP | +0.006490 [-0.002201, +0.014396] | 4/5 | +0.005134 [+0.000372, +0.010830] |
| minus shuffled-VTSE MLP | +0.018050 [+0.001386, +0.031864] | 4/5 | +0.065421 [+0.007596, +0.173627] |

Additional aligned-minus-no-VTSE changes are weak recall `+0.025608`
([+0.001488, +0.054625]), strong recall `+0.034212`
([+0.026278, +0.042146]), and FPR `-0.004032`
([-0.013710, +0.006613]).

## Evidence interpretation

Two statements must remain separate:

1. The aligned feature has an incremental signal over the shuffled control;
   both its weak-AUROC and AUPR bootstrap intervals are positive.
2. The aligned-vs-no-VTSE weak-AUROC interval crosses zero.  Thus the
   point-estimate gate passes, but the primary ranking effect is not yet a
   statistically robust evidence chain for an SCI innovation claim.

The high variance against the shuffled control (one seed has much poorer
shuffled selection) also means this result must not be summarized as a single
stable effect size.  A fresh identity-isolated development confirmation with
the entire three-arm protocol frozen is required before this mechanism is used
as a paper-level claim; it must still precede any one-shot test-v3 use.

## Artifacts and boundary

- Preregistration: `docs/plans/route_a_vft_mlp_capacity_physics_control_unified_decision_ablation_plan_20260801.md`
- Runner: `experiments/run_route_a_vft_mlp_capacity_physics_control_unified_decision_ablation.py`
- Tests: `tests/test_route_a_vft_mlp_capacity_physics_control_unified_decision_ablation.py` (3 passed)
- Summary SHA-256: `47B88F9DDB01FF0B8A185B2EBCB934501122B3AADFAA72FC4C35C498CEC55B21`
- `locked_test_used=false`, `test_v2_used=false`, `test_v3_used=false`.
