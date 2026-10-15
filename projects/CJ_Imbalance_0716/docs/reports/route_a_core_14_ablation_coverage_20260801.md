# Route A core-14 ablation coverage audit

## Evidence status

The 14 requested ablations have not all been established on one current,
end-to-end unified-candidate protocol. Ten have direct current-protocol
evidence; four remain partial evidence and must not be merged into a claim of
complete core ablation coverage.

| No. | Required mechanism | Coverage | Authoritative artifact | Reason for status |
|---:|---|---|---|---|
| 1 | remove VTSE | direct | VFT-remove-VTSE unified ablation | Q4 removes the VFT expert while every non-VFT logit and the role protocol remain fixed |
| 2 | include strong positives | direct | VFT strict-mask unified ablation | All-positive Ridge enters identical Q4 fusion and role protocol |
| 3 | remove weak mask | direct | VFT strict-mask unified ablation | Same unified all-positive arm removes strict mask |
| 4 | BCE vs three strata | direct | VFT weighting/gate core ablation | A2 vs A0, all remaining factors fixed |
| 5 | weak-positive weighting only | direct | VFT weighting/gate core ablation | A1 vs A0, all remaining factors fixed |
| 6 | remove uncertainty gate | direct | VFT weighting/gate core ablation | A3 vs A1, identical weights |
| 7 | linear vs sqrt gate | direct | VFT sqrt/linear gate ablation | Same four-input VFT policy |
| 8 | remove residual cap | partial | MFT cap unified-decision replay | Current Q4 replay completed, but the preregistered early-FPR integrity gate fails for the original arm |
| 9 | global/final/stage2/stage3 | partial | layerwise separability study | Diagnostic probes; no promoted layer |
| 10 | same-capacity MLP | partial | frozen-head capacity baselines | Not final current decision policy |
| 11 | top-k 4/8/16 | partial | RawSaw top-k study | Equal-budget retraining but no exact historical reproduction |
| 12 | weak threshold 5/10/15 dB | direct | frozen threshold sensitivity | Fixed candidate, no retuning |
| 13 | mask measurement error | direct | VFT hard/soft/power-error study | Fit-only independent 1 dB perturbation |
| 14 | hard vs soft mask | direct | VFT hard/soft/power-error study | Fixed-temperature direct comparison |

## New direct result for item 1

The newly completed five-seed decision ablation isolates the deployed
visibility-targeted Ridge-STFT expert. Removing VTSE from Q4 lowers VFT
weak-vs-negative AUROC by `-0.01864` (95% CI `[-0.02358, -0.01418]`) and AUPR
by `-0.01296`; both effects occur in all five seeds. Q4 also improves weak
recall by `+0.02310` and strong recall by `+0.02858`. The negative-FPR change
is `-0.00097`, so the evidence is not a recall-for-FPR trade. The base-only
head has numerically identical VFT predictions to the no-VTSE arm, so it is a
transparency check rather than evidence for the still-partial MLP-capacity
control. See
[route_a_vft_remove_vtse_unified_decision_ablation_results_20260801.md](../results/route_a_vft_remove_vtse_unified_decision_ablation_results_20260801.md).

## New direct result for items 4--6

The newly completed direct VFT-head ablation rejects a simple weighting
explanation: weak-only weight 4 and three-stratum equal mass reduce early
weak-AUROC by `-0.00557` and `-0.00751`, respectively, versus uniform BCE.
The sqrt gate satisfies only its safety criterion, not a performance-gain
claim. Details are in
[route_a_vft_weighting_uncertainty_core_ablation_results_20260801.md](../results/route_a_vft_weighting_uncertainty_core_ablation_results_20260801.md).

The direct all-positive unified ablation further shows that removing strict
weak supervision decreases VFT weak-AUROC by `-0.01171` in all five seeds.
See [route_a_vft_strict_mask_unified_ablation_results_20260801.md](../results/route_a_vft_strict_mask_unified_ablation_results_20260801.md).

## Attempted current-decision result for item 8

The Q4 `cap=0.75` versus natural-cap `1.0` replay is reproducible and shows
positive MFT ranking differences for cap `1.0`, but it is deliberately kept
partial: the pre-registered early-role FPR integrity condition fails for the
unchanged cap `0.75` arm in seed 2042 (`0.20161`). The threshold selector did
apply its FPR constraint on the separate `threshold` role; the failure shows
that such a role constraint is not an early-role guarantee. It is therefore a
useful protocol diagnosis, not safe evidence to promote a cap change or close
core item 8. Details are in
[route_a_mft_remove_cap_unified_decision_ablation_results_20260801.md](../results/route_a_mft_remove_cap_unified_decision_ablation_results_20260801.md).

## Consequence

The positive current evidence supports a narrow VFT visibility-selective
Ridge-STFT boundary, not the stronger claim that generic loss reweighting or
the specific sqrt gate creates the gain. Because the full candidate still
fails its external VFT/ISRJ OOD gate, it is not appropriate to fill the seven
partial entries by post-hoc retuning or to initiate test-v3. A new candidate
requires a fresh, falsifiable source-level mechanism before development-v13.

Machine-readable audit:
`outputs/tapc/route_a_core_14_ablation_coverage/core_14_ablation_coverage.json`.
All sources record `locked_test_used=false` and `test_v3_used=false`.
