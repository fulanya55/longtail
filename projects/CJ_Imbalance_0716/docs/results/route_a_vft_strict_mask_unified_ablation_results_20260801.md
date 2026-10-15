# VFT strict-mask versus all-positive unified decision ablation

## Result

This direct six-label decision-level experiment confirms that the strict weak
VFT mask is necessary. Replacing the strict Ridge-STFT score with an
all-positive (weak plus strong) composition-matched Ridge score, while holding
the Q4 fusion-head architecture, calibration-role protocol, and threshold-role
protocol fixed, degrades every primary VFT weak-evidence metric.

| AllPositive_NoWeakMask minus StrictWeak_Q4 | Mean delta | 95% bootstrap CI | Positive seeds |
|---|---:|---:|---:|
| VFT weak-vs-negative AUROC | -0.01171 | [-0.01526, -0.00815] | 0/5 |
| VFT weak recall | -0.01592 | [-0.03267, -0.00159] | 1/5 |
| VFT AUPR | -0.00487 | [-0.00672, -0.00360] | 0/5 |
| VFT strong recall | -0.00689 | [-0.01745, +0.00142] | 1/5 |
| VFT negative FPR | -0.00452 | [-0.01097, +0.00258] | 2/5 |

All non-VFT logits remain exactly unchanged. Thus the degradation cannot be
attributed to RawSaw, MFT guard, ISRJ B2, calibration/threshold data leakage,
or an unrelated label change.

## Decision

The all-positive arm fails every pre-registered promotion condition. Its small
mean FPR reduction is not an acceptable exchange for stable weak-AUROC,
weak-recall, and AUPR losses. We retain strict weak-positive Ridge supervision
and stop both “include strong VFT positives” and “remove the weak mask” as
candidate changes.

Together with the construction-only study, this establishes a direct
mechanism: stronger VFT examples move the expert boundary toward easy positive
patterns and reduce discrimination of the weak-positive frontier even after
the same downstream fusion, Platt calibration, and FPR-constrained threshold
procedure are applied.

## Reproducibility and boundary

- Plan: [route_a_vft_strict_mask_unified_ablation_plan_20260801.md](../plans/route_a_vft_strict_mask_unified_ablation_plan_20260801.md)
- Runner: `experiments/run_route_a_vft_strict_mask_unified_ablation.py`
- Summary: `outputs/tapc/route_a_vft_strict_mask_unified_ablation/route_a_vft_strict_mask_unified_ablation_summary.json`
- Five identity-isolated development-v3 seeds: `2039, 2040, 2041, 2042, 2043`.

No candidate was promoted or altered. Every receipt records
`locked_test_used=false`, `test_v2_used=false`, and `test_v3_used=false`.
