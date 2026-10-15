# Route A RawSaw top-k 4/8/16 ablation result

## Result

Under the same five development-v3 receiver-identity roles, parameter count,
optimizer, budget, checkpoint selection, calibration and threshold protocol,
the retrained RawSaw arms give:

| Pooling | SawFM AUPR | Weak recall | Weak AUROC | FPR |
|---|---:|---:|---:|---:|
| top-k=4 | 0.92233 | 0.54248 | 0.76970 | 0.18887 |
| top-k=8 | 0.92974 | 0.56389 | 0.78196 | 0.19726 |
| top-k=16 | 0.92922 | 0.58520 | 0.78994 | 0.19516 |

Relative to top-k=16, top-k=4 has weak AUROC `-0.02024` (95% bootstrap CI
`[-0.04225,+0.00274]`) and top-k=8 has `-0.00798`
(`[-0.03224,+0.01142]`).  Neither interval establishes a robust pooling-scale
winner, though top-k=16 has the highest point estimate for weak recall and
weak AUROC.  The top-k=4 FPR is lower by `0.00629`, illustrating the expected
recall/FPR tradeoff rather than a free gain.

## Safety and reproducibility boundary

All three arms have the same trainable parameter count.  Non-SawFM
probabilities and predictions are exactly identical across all arms, so no
SawFM pooling change leaked into the five safe labels.  `test-v3` was not used.

The newly retrained top-k=16 arm did not reproduce the historical RawSaw
checkpoint bit-for-bit (maximum state difference across seeds: `0.04577` to
`0.13899`).  Its final metrics nevertheless stayed close to the historical C1
values (per-seed SawFM AUPR differences within about `[-0.00367,+0.00501]`).
The discrepancy begins after several CUDA training epochs and is consistent
with numerical training nondeterminism, not a changed top-k mechanism.

Therefore this is valid **within-run, equal-budget retraining evidence**, but
not a license to alter the frozen top-k=16 candidate or claim a new optimal
pooling scale.  top-k remains an implementation sensitivity, not an innovation
claim.

## Artifacts

- Plan: [route_a_rawsaw_topk_4_8_16_ablation_plan_20260801.md](../plans/route_a_rawsaw_topk_4_8_16_ablation_plan_20260801.md)
- Runner: `experiments/run_route_a_rawsaw_topk_ablation.py`
- Summary: `outputs/tapc/route_a_rawsaw_topk_4_8_16_ablation/route_a_rawsaw_topk_4_8_16_ablation_summary.json`

Every run receipt records `locked_test_used=false`, `test_v2_used=false`, and
`test_v3_used=false`.
