# Route A v21: visibility-balanced morphology selector result

## Decision

**Stop the visibility-balanced residual-coefficient selector.** The candidate met the predeclared Q4 hard-negative FPR reduction target, but it did not preserve weak-VFT recall on every held-out seed and surrendered too much of the FPR benefit relative to the global-only control. The observed result is retained as a negative transport result; no coefficient, calibration, or threshold will be revisited on v21.

## Protocol and evidence boundary

v21 used a new train-role-only source (base seed `600000000`, 5,040 unique receiver identities, SHA-256 `ae775f8ef154962eb65f81ea75f94e93d16e2038e70fe7dc64fd356b37ba797b`). Identity-disjoint fit, selection, and evaluation roles contained 3,024, 1,008, and 1,008 ISRJ-present identities respectively. The evaluation role was not used to select a coefficient.

Q4, its calibration and thresholds, and the v19 morphology checkpoints were frozen. The only selected value was the scalar residual coefficient on the preregistered grid. All non-VFT Q4 logits were bitwise unchanged. `locked_test_used=false`, `test_v2_used=false`, and `test_v3_used=false`; the sealed test-v3 manifest remained sealed.

## Five-seed held-out evaluation

| Comparison | Hard-negative VFT FPR delta | VFT positive recall delta | weak-VFT recall delta | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Candidate minus Q4 | `-0.0344`, CI `[-0.0609,-0.0109]` | `-0.0043` | `-0.0159`, CI `[-0.0283,-0.0035]` | FPR improves, but weak recall is unsafe |
| Candidate minus global-only control | `+0.0555`, CI `[+0.0375,+0.0734]` | `+0.0090` | `+0.0248`, CI `[+0.0142,+0.0372]` | Weak recall is recovered only by forfeiting FPR benefit |

The candidate's VFT AUPR changed by `+0.00515`; ISRJ AUPR, SawFM AUPR, and SawFM weak recall remained exactly unchanged because their frozen Q4 outputs were not modified. These secondary values cannot override failed safety gates.

## Gate outcome

- Q4 FPR reduction at least `0.030` in 4/5 seeds: pass.
- Overall VFT recall safety on every seed: fail.
- Weak-VFT recall safety on every seed: fail.
- Weak recall gain over global-only in 4/5 seeds: pass.
- Candidate versus global-only FPR tradeoff no worse than `+0.020`: fail.
- Frozen contracts and test-v3 seal: pass.

The preregistered decision is `stop_visibility_balanced_selector`.

## Failure mechanism and next falsifiable step

The selection constraint chooses a smaller morphology residual than global-only selection, so it correctly trades some FPR reduction for weak recall *within the source*. That tradeoff does not transport sufficiently to new held-out identities: it still loses weak recall against Q4 and costs `+0.0555` FPR against the global control. This rules out coefficient selection as the primary remedy, rather than supporting a finer grid or another threshold sweep.

The next isolated hypothesis is upstream: the fixed v19 morphology expert was trained with ordinary BCE, so strong VFT positives dominate its learned decision boundary. On a new source, a same-capacity morphology expert trained with visibility-balanced positive mass will be compared directly with an ordinary-BCE expert. It must improve weak-VFT-versus-hard-negative ranking while preserving strong-positive ranking before any new Q4 fusion is considered.

## Artifacts

- Plan: `docs/plans/route_a_v21_visibility_balanced_morphology_selector_plan_20260802.md`
- Generator/audit/runner: `experiments/generate_route_a_v21_visibility_balanced_morphology_selector.m`, `experiments/audit_route_a_v21_visibility_balanced_morphology_selector.py`, `experiments/run_route_a_v21_visibility_balanced_morphology_selector.py`
- Formal artifacts: `outputs/tapc/route_a_v21_visibility_balanced_morphology_selector/`
