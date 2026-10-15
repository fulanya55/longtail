# VFT weighting and uncertainty-gate core ablation

## Direct answer to core ablations 4--6

This is the first direct comparison in which all arms share the same frozen
H1+B2+MFT-guard logits, Ridge-STFT VFT score, four-input VFT logistic head,
development-v3 roles, Platt calibration, and FPR-constrained threshold rule.
`A3_WeakWeight_SqrtGate` reproduces the historical Q4 early VFT logits within
`1e-5` for all five seeds; non-VFT logits are exactly unchanged.

| Mechanism comparison on early VFT | Delta weak AUROC, 95% CI | Delta weak recall, 95% CI | Delta strong recall | Delta FPR |
|---|---:|---:|---:|---:|
| weak-positive weight 4, no gate minus uniform BCE, no gate | -0.00557, [-0.00816, -0.00338] | -0.00052, [-0.00710, +0.00350] | -0.00445 | +0.00339 |
| three-stratum equal mass, no gate minus uniform BCE, no gate | -0.00751, [-0.01132, -0.00392] | -0.00871, [-0.01942, +0.00201] | -0.00749 | +0.00242 |
| three-stratum equal mass, no gate minus weak-only weight, no gate | -0.00194, [-0.00342, -0.00037] | -0.00819, [-0.01722, +0.00131] | -0.00304 | -0.00097 |
| weak weight 4 with sqrt gate minus the same weight without gate | -0.00267, [-0.00455, -0.00131] | +0.00531, [-0.00212, +0.01347] | +0.00629 | -0.00097 |

## Interpretation

Neither weak-positive upweighting nor three-stratum equal mass passes its
pre-registered gain gate: both have negative weak-AUROC effects in all five
or four of five seeds, respectively, and neither approaches the required
weak-recall gain. This directly rules out the claim that the VFT effect is
created by a simple reweighting of labels that share the same frozen physical
score.

The sqrt uncertainty gate passes its narrower **safety** test. It gives a
small but consistently negative ranking shift, yet stays well above the
`-0.010` weak-AUROC safety floor, reduces mean FPR slightly, and improves
strong-positive recall in all five seeds. Accordingly it is an acceptable
safety/implementation choice, not a demonstrated source of the method's VFT
gain or an independent innovation claim.

These outcomes agree with the earlier hard-versus-soft and weak-only-versus-
mixed-positive Ridge experiments: weak evidence cannot be recovered merely by
redistributing loss mass after the relevant time-frequency evidence has been
fixed. The remaining defensible VFT mechanism is the visibility-selective
Ridge-STFT boundary itself, whose external ISRJ confusion remains unresolved.

## Boundary and reproducibility

- Plan: [route_a_vft_weighting_uncertainty_core_ablation_plan_20260801.md](../plans/route_a_vft_weighting_uncertainty_core_ablation_plan_20260801.md)
- Runner: `experiments/run_route_a_vft_weighting_uncertainty_core_ablation.py`
- Summary: `outputs/tapc/route_a_vft_weighting_uncertainty_core_ablation/route_a_vft_weighting_uncertainty_core_ablation_summary.json`
- Five identity-isolated development-v3 seeds: `2039, 2040, 2041, 2042, 2043`.

No candidate was selected or altered. All records state
`locked_test_used=false`, `test_v2_used=false`, and `test_v3_used=false`.
