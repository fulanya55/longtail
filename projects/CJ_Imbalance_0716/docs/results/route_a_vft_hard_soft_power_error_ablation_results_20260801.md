# Route A VFT hard-mask, power-error, and soft-visibility ablation

## Result

On the fixed five-seed development-v3 protocol, the current hard weak-positive
supervision remains the only justified VFT training rule.  A continuous
visibility weight improves the fit-adjacent validation role but does not
improve the pre-specified early held-out role; it therefore cannot replace the
hard mask or create a new unified candidate.

| Comparison with HardNominal | Held-out role | Delta AUROC, 95% bootstrap CI | Delta AUPR, 95% bootstrap CI |
|---|---|---:|---:|
| HardPowerError (fit mask receives independent active-component N(0, 1 dB) error) | validation | +0.02478, [+0.00403, +0.04898] | +0.04079, [-0.00040, +0.07922] |
|  | early | -0.00044, [-0.00922, +0.00834] | -0.00198, [-0.00544, +0.00120] |
| SoftVisibility (all positives with fixed 2.5 dB sigmoid weight) | validation | +0.04404, [+0.01286, +0.07331] | +0.10194, [+0.05599, +0.15823] |
|  | early | -0.00016, [-0.01272, +0.01240] | -0.00447, [-0.01319, +0.00443] |

The corresponding early-role means are: HardNominal AUROC/AUPR
`0.64402 / 0.25466`, HardPowerError `0.64358 / 0.25268`, and SoftVisibility
`0.64386 / 0.25019`.  Thus the soft arm's attractive validation improvement
does not reproduce on the role that was reserved for this mechanism decision.

## Interpretation and decision

The power-error sensitivity gate passes: a 1 dB active-component measurement
error during *fit-mask construction only* has no material early weak-AUROC
loss (mean `-0.00044`, well above the pre-registered `-0.010` floor).  This is
evidence that the strict weak-mask mechanism is not precariously dependent on
exact nominal component powers at this perturbation scale.

The two SoftVisibility early gates fail.  Its early AUROC and AUPR deltas have
intervals spanning zero and are slightly negative in mean.  The outcome is
consistent with the earlier weak-only versus mixed-positive study: admitting
stronger VFT-positive examples, even with down-weighting, improves an
easy/fit-adjacent role but does not provide reproducible strictly-weak evidence
separation.  We retain the current strict hard weak-positive Ridge-STFT VFT
expert and do **not** alter its calibration, threshold, fusion, or the frozen
unified candidate.

This is a construction-level mechanism and robustness result only.  It does
not establish complete-candidate external OOD robustness, does not select a
new candidate, and does not close any frozen-test evidence gap.

## Reproducibility and evidence boundary

- Plan: [route_a_vft_hard_soft_power_error_ablation_plan_20260801.md](../plans/route_a_vft_hard_soft_power_error_ablation_plan_20260801.md)
- Runner: `experiments/run_route_a_vft_hard_soft_power_error_ablation.py`
- Summary: `outputs/tapc/route_a_vft_hard_soft_power_error_ablation/route_a_vft_hard_soft_power_error_ablation_summary.json`
- Five identity-isolated development-v3 seeds: `2039, 2040, 2041, 2042, 2043`.
- Fixed descriptor: `16x16` STFT VFT, fit-only z-standardisation, balanced
  class mass, and `RidgeClassifier(alpha=10)`.

All records state `locked_test_used=false`, `test_v2_used=false`, and
`test_v3_used=false`; the sealed test-v3 manifest remains unread by the
experiment other than its three no-access flag check.
