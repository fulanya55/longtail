# Route A H0h: frozen AR(1) receiver-noise replication result

## Decision

**H0h passes as a narrow external replication of the ISRJ aligned-periodicity
increment under a changed receiver-noise process.** It uses no retraining,
calibration, threshold, feature or fusion adjustment. This strengthens only
the bounded ISRJ mechanism claim. It does not validate the stopped
counterfactual VTSE route, repair VFT, establish full six-label robustness, or
authorize a final test-v3 evaluation.

## Preregistered source and boundary

- New train-only source: `C:\LJQ_code\干扰数据\复合干扰数据0802_h0h_ar1\closedset_6class_train.mat`; SHA-256 `7f0bdc29302519aa890ff42466c370c60009f59bc9dc24687d1b8ea9f660aa93`.
- It has 5,040 samples, all 63 six-label combinations and 80 examples per combination. Its 5,040 receiver-noise identities are unique and occupy `968677743--968682782`; the audit found zero overlap with every live prior train source and the recorded H0e/H0f/H0g ranges.
- It is derived from a separately generated white-noise base with the same identities and all non-I/Q dataset values preserved. For each identity the exact reference white realization was removed and replaced by a unit-power, proper-complex AR(1) realization: `n[t] = 0.80 n[t-1] + sqrt(1-0.80^2) e[t]`, `e[t] ~ CN(0,1)`.
- The fixed 200,000-sample non-selection process check reports mean power `0.99669`, pseudo-covariance magnitude `0.01104`, and lag-one correlation `0.79684`; all preregistered process tolerances pass. Fixed I/Q columns differ from the base, while labels and receiver identities are exact.
- No locked test, test-v2, or test-v3 file was generated, opened, analyzed, or used. The test-v3 seal remained unchanged.

## Exact frozen comparison

Five saved training seeds (`2039--2043`) compare the same arms as H0g:

| Arm | Frozen content |
| --- | --- |
| B0_H1_Frozen | Focal-36 encoder, H1 linear head and RawSaw route with saved B0 calibration and thresholds |
| B2_H1AlignedPeriodicity_Frozen | B0 plus saved aligned 310-D periodicity expert, fusion, B2 calibration and thresholds |

The AR(1) source was never used for fitting, model selection, calibration or thresholding. The reported five-seed evaluation is therefore a frozen external replication, not a new optimized model result.

## Five-seed formal result: B2 minus B0

| Endpoint | Mean delta | Bootstrap 95% CI | Positive seeds |
| --- | ---: | --- | ---: |
| ISRJ AUPR | +0.02242 | [+0.01577, +0.02852] | 5/5 |
| ISRJ weak recall | +0.10270 | [+0.09658, +0.10991] | 5/5 |
| ISRJ strong recall | +0.04309 | [+0.02394, +0.06284] | 5/5 |
| ISRJ weak-vs-negative AUROC | +0.08037 | [+0.07462, +0.08682] | 5/5 |
| ISRJ FPR | -0.00468 | [-0.01113, +0.00024] | 4/5 lower |
| Six-label macro-AUPR | +0.00374 | [+0.00263, +0.00472] | 5/5 |
| Six-label macro-F1 | +0.00616 | [+0.00425, +0.00808] | 5/5 |

B2 absolute ISRJ AUPR / weak recall / weak AUROC / FPR are `0.91259 / 0.41622 / 0.68465 / 0.18363`. All ten preregistered H0g-derived gates pass: positive ISRJ AUPR and weak AUROC, weak-recall gain at least `+0.030`, 5/5 consistency in both weak endpoints, B2 FPR below `0.25`, non-negative macro-AUPR, exact SawFM preservation, no external refit and an untouched test-v3 seal.

## Descriptive safety result

This report was produced after the frozen comparison and performed no selection. B2 only changes the ISRJ output; all five other label FPRs match B0 exactly. Its mean ISRJ FPR is `0.18363` (B0: `0.18831`). Across all labels and five receiver-identity cohorts defined by `receiver_noise_seed mod 5`, the maximum mean FPR is `0.23387` for both arms (FMB, cohort 0). The full per-seed and cohort table is retained in the safety artifact below.

## Interpretation and limits

Together with development-v3, OOD-v1 and H0g, H0h supports this limited statement: the frozen aligned-periodicity increment provides label-local ISRJ weak evidence beyond its strong parent across new receiver identities, parameter shifts and one preregistered correlated receiver-noise shift.

It does **not** show that the complete VTSE-VBF six-label candidate is ready, that MFT/VFT weak evidence is solved, that all possible noise shifts are safe, or that external development evidence can substitute for a sealed final test. H0h is now frozen against AR coefficient, power, feature, fusion, calibration or threshold tuning.

Machine results:

- `outputs/tapc/route_a_h0h_ar1_noise_frozen/route_a_h0h_ar1_noise_frozen_summary.json`
- `outputs/tapc/route_a_h0h_ar1_noise_frozen/nonselection_safety_report.json`
- `outputs/tapc/route_a_h0h_ar1_noise_generation/identity_noise_audit.json`
