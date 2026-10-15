# Route A VFT ISRJ-OOD frozen-threshold transport diagnostic

## Result

The external ISRJ parameter-OOD VFT failure is not merely a calibration or
threshold-transfer issue. With frozen Q4 probabilities and thresholds, mean
VFT negative FPR is `0.32653`, weak recall is `0.44091`, and weak-versus-
negative AUROC is only `0.59332`.

For diagnosis only, an oracle OOD threshold was computed independently per
seed as the lowest threshold giving FPR <=0.20. It controls mean FPR at
`0.19960`, but weak recall falls to `0.31721` (delta `-0.12370`), requiring an
average threshold increase of `+0.10511`.

| Seed | frozen FPR | frozen weak recall | oracle FPR | oracle weak recall | threshold shift |
|---:|---:|---:|---:|---:|---:|
| 2039 | 0.29315 | 0.38636 | 0.19960 | 0.30195 | +0.07792 |
| 2040 | 0.36371 | 0.42370 | 0.19960 | 0.24026 | +0.16477 |
| 2041 | 0.34234 | 0.46266 | 0.19960 | 0.34253 | +0.08637 |
| 2042 | 0.25887 | 0.47403 | 0.19960 | 0.42532 | +0.04681 |
| 2043 | 0.37460 | 0.45779 | 0.19960 | 0.27597 | +0.14968 |

The predefined diagnosis conditions both hold: mean ranking is below `0.65`,
and FPR control costs more than `0.03` weak recall. A threshold-only repair is
therefore rejected. The oracle calculation is strictly explanatory and cannot
be used to refit calibration, choose a threshold, construct a candidate, or
touch test-v3.

Artifacts: `docs/plans/route_a_vft_ood_threshold_transport_diagnostic_plan_20260801.md`,
`experiments/audit_route_a_vft_ood_threshold_transport.py`, and
`outputs/tapc/route_a_vft_ood_threshold_transport_diagnostic/route_a_vft_ood_threshold_transport_diagnostic_summary.json`.
