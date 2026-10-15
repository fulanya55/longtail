# Route A H0g: Frozen farther-parameter ISRJ replication result

## Decision

**H0g passes as a narrow external replication of the ISRJ aligned-periodicity
mechanism.** With every learned artifact, calibration parameter and threshold
frozen, B2 improves ISRJ weak evidence over its exact B0 parent in all five
training seeds on a new identity-isolated high-replica regime. This strengthens
the ISRJ-specific evidence chain; it does not repair the stopped VFT OOD route,
does not promote a changed unified candidate, and does not authorize test-v3.

## Source and evidence boundary

- Fresh train-only source: `C:\LJQ_code\干扰数据\复合干扰数据0802_isrj_ood_h0g\closedset_6class_train.mat`; SHA-256 `8cdd8a78a3371e059dea8ab9448655e4d8e309d5cb41efc8bf307011b124daf9`.
- 5,040 samples / 63 label combinations / 80 samples per combination, with
  5,040 unique receiver-noise identities in `958677743--958682782`.
- ISRJ width is 25--30 us and replica count is exactly 9. The nominal
  development range is 5--20 us / 1--5 replicas; OOD-v1 was 22--30 us / 6--8.
  H0g is therefore a feasible, new high-replica extrapolation.
- The generator rejected the initially proposed 31--36 us / 9--11 regime at
  dry-run because it violates the physical record-capacity invariant
  `(replicas+1)*width <= 3000` samples. That illegal source was never
  generated or evaluated.
- The audit found zero receiver overlap with all live historical train sources
  and with H0e/H0f preserved identity ranges. No locked test, test-v2 or
  test-v3 was generated, opened, analyzed or used.

## Exact frozen comparison

For training seeds 2039--2043, both arms use the saved Focal-36 encoder,
frozen H1 linear head, RawSaw route, periodicity expert (where applicable),
fusion model, Platt calibration and thresholds from the original H1/B2
experiment. No H0g data were used to fit, choose or alter any artifact.

| Arm | Difference |
| --- | --- |
| B0_H1_Frozen | H1 plus RawSaw, saved B0 calibration/thresholds |
| B2_H1AlignedPeriodicity_Frozen | B0 plus saved aligned 310-D periodicity score and saved B2 fusion, calibration/thresholds |

## Five-seed formal result: B2 minus B0

| Endpoint | Mean delta | Bootstrap 95% CI | Positive seeds |
| --- | ---: | --- | ---: |
| ISRJ AUPR | +0.11927 | [+0.09376, +0.14503] | 5/5 |
| ISRJ weak recall | +0.07125 | [+0.06314, +0.08078] | 5/5 |
| ISRJ strong recall | +0.19418 | [+0.15334, +0.23502] | 5/5 |
| ISRJ weak-vs-negative AUROC | +0.09965 | [+0.08981, +0.11069] | 5/5 |
| ISRJ FPR | +0.00242 | [-0.00395, +0.00863] | 2/5 lower |
| Six-label macro-AUPR | +0.01988 | [+0.01563, +0.02414] | 5/5 |
| Six-label macro-F1 | +0.02740 | [+0.02185, +0.03265] | 5/5 |

B2 absolute ISRJ AUPR / weak recall / weak AUROC / FPR are
`0.71004 / 0.26949 / 0.58755 / 0.18887`. The per-seed weak-AUROC deltas are
`+0.08362, +0.09196, +0.09792, +0.11839, +0.10635`; weak-recall deltas are
`+0.05820, +0.06702, +0.06702, +0.08995, +0.07407`.

All ten preregistered gates pass: positive ISRJ AUPR and weak AUROC, weak
recall at least +0.030, 4/5 consistency for both weak metrics, B2 FPR below
0.25, non-negative macro-AUPR, exact SawFM preservation, no external refit,
and an untouched test-v3 seal.

## Interpretation and limits

The result is particularly informative because it is not a threshold change:
the B2 FPR shift is near zero while weak ranking and recall both improve. It
also cannot be reduced to a retrained high-capacity backbone—the B0/B2
checkpoints and all decision artifacts are bit-addressed frozen inputs.

The supported statement is limited to **aligned periodicity as an ISRJ
mechanism** across three parameter settings: development-v3/development-v7,
OOD-v1 (22--30 us, 6--8 replicas), and H0g (25--30 us, 9 replicas). It does
not establish VFT robustness under ISRJ OOD, global noise-model robustness,
or final six-label test performance. The H0g source is frozen against any
periodicity feature, fusion, calibration, threshold or range tuning.

Machine-readable result:
`outputs/tapc/route_a_h0g_isrj_parameter_ood_v2_frozen/route_a_h0g_isrj_parameter_ood_v2_frozen_summary.json`.
Source audit:
`outputs/tapc/route_a_h0g_isrj_parameter_ood_v2_generation/identity_parameter_audit.json`.
