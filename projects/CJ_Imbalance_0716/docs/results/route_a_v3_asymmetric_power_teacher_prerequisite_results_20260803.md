# Route A H0t: asymmetric high-to-low power posterior-transfer prerequisite result

## Conclusion

H0t passes as a **frozen-score feasibility condition**. For all three target
labels, a Focal-36 model frozen at the 30 dB development-v3 condition ranks
same-identity 50 dB target evidence much more accurately than the 10 dB view.
This admits one subsequent fresh-identity H1 design for an asymmetric,
label-local posterior-transfer mechanism. It does **not** establish that
distillation improves a classifier, repair C3, or authorize test-v3.

## Fixed protocol and boundary

- Five frozen `E2_Focal_36` checkpoints (seeds 2039--2043) were reconstructed
  from their matching F2 states and only forwarded.
- For each pre-existing `early` role (2,520 identities), matching train power
  views at 10 dB and 50 dB were evaluated. Both views retained exactly the
  same identities and labels.
- No optimization, teacher update, calibrator, threshold selection or model
  selection occurred. The report is ranking-only.
- test-v3 was read only as a seal manifest: `access_authorized=false`,
  `analysis_allowed=false`, `used=false`. No test MAT was opened.

## Five-seed aggregate

| Target | 50 dB AUROC | 10 dB AUROC | Delta AUROC | Delta AUPR | 50 dB top-decile lift | Positive score increase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MFT | 0.98313 | 0.57470 | +0.40843 | +0.35905 | 1.96563 | 0.87594 |
| VFT | 0.89655 | 0.56668 | +0.32987 | +0.31264 | 1.96875 | 0.78344 |
| ISRJ | 0.96652 | 0.54815 | +0.41838 | +0.35855 | 1.96875 | 0.87188 |

Every target has a positive high-minus-low AUROC in 5/5 seeds. The fixed
top-decile lift is descriptive ranking evidence, not a probability threshold.

## Pre-registered decision

All four H0t gates pass: high-view mean AUROC >= 0.60, high-minus-low mean
AUROC >= +0.030, at least 4/5 positive per-target seed deltas, and high-view
top-decile lift >= 1.50. The only permitted follow-up is a separately
pre-registered, new-identity H1 learned comparison. It must include a
same-capacity, composition-preserving paired-teacher permutation control and
must retain supervised labels for the student.

## Artifacts

- Plan: `docs/plans/route_a_v3_asymmetric_power_teacher_prerequisite_20260803.md`
- Runner: `experiments/audit_route_a_v3_asymmetric_power_teacher_prerequisite.py`
- Immutable receipt and per-seed logits:
  `outputs/tapc/route_a_v3_asymmetric_power_teacher_prerequisite/summary.json`

All artifacts record `locked_test_used=false`, `test_v2_used=false`, and
`test_v3_used=false`.
