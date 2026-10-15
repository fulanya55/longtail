# Route A v23: weak-positive/hard-negative ranking morphology expert result

## Decision

**Stop the fixed weak-positive/hard-negative pair-ranking loss on the current 16x16 STFT morphology representation.** It is safe for strong and overall ranking, but it does not improve the weak-positive ranking that it explicitly targets.

## Protocol and evidence boundary

v23 used a third fresh train-only source, base seed `640000000`, SHA-256 `f528987a4ca2fbec65d492d8819f3b12b6330c62bdc5a13b010edc3456a8b57b`, and 5,040 unique receiver identities. The ISRJ-present roles were 1,536/512/512 fit/early-stop/evaluation identities; evaluation had 63 weak VFT positives, 193 strong VFT positives, and 256 matched hard negatives.

The two 17,281-parameter heads, features, initialization, AdamW settings, batch size, 30-epoch budget, and unweighted three-stratum checkpoint criterion were matched. `M_WPR` alone added the preregistered `0.25 * softplus(-(s_weak-s_hard))` minibatch pair term to BCE. No coefficient sweep, threshold, calibration, Q4 score, width, v19-v22 source/prediction, historical locked test, test-v2, or test-v3 was accessed.

## Five-seed held-out evaluation

| Metric: M_WPR minus M_BCE | Mean delta | 95% paired bootstrap CI | Decision |
| --- | ---: | ---: | --- |
| weak VFT versus hard-negative AUROC | `-0.00097` | `[-0.00219,+0.00026]` | no improvement |
| weak VFT versus hard-negative AUPR | `+0.00100` | recorded in summary | below gain gate |
| strong VFT versus hard-negative AUROC | `-0.00084` | recorded in summary | safe |
| overall VFT AUROC | `-0.00087` | recorded in summary | safe |

Only one of five weak-AUROC deltas was positive. The weak AUROC magnitude, confidence-interval, consistency, and weak-AUPR gates failed; strong and overall safety, matched architecture, and test-v3 sealing passed. The preregistered decision is `stop_weak_pair_ranking_morphology`.

## Cumulative diagnosis

v19 proved that the compact STFT morphology feature is informative (including weak-positive ranking above chance). v22 showed that changing weak-positive mass harms that ranking. v23 showed that directly adding a weak-positive/hard-negative ranking objective produces an essentially null result while leaving other strata safe. Together, the evidence says the next bottleneck is not a scalar fusion coefficient or a loss reweighting term on this fixed single-resolution feature: the weak evidence is likely compressed before the classifier sees it.

The next permissible structural test is a matched-capacity, multi-resolution temporal-frequency feature probe: retain the original 16x16 feature as the base, add a fixed higher-resolution/local temporal view, and compare it with an input-dimension- and parameter-matched information-free projection control on a fourth new identity source. It must first improve weak ranking before any Q4 fusion is redesigned.

## Artifacts

- Plan: `docs/plans/route_a_v23_weak_pair_ranking_morphology_expert_plan_20260802.md`
- Generator/audit/runner: `experiments/generate_route_a_v23_weak_pair_ranking_morphology_expert.m`, `experiments/audit_route_a_v23_weak_pair_ranking_morphology_expert.py`, `experiments/run_route_a_v23_weak_pair_ranking_morphology_expert.py`
- Formal artifacts: `outputs/tapc/route_a_v23_weak_pair_ranking_morphology_expert/`
