# Route A H0s: frozen ISRJ aligned-periodicity under combined parameter and receiver-noise shift

## Question and boundary

H0s asks one strictly confirmatory question: **does the already frozen
aligned-periodicity increment B2 retain its ISRJ weak-evidence advantage over
its exact B0 parent when the high-replica regime and correlated receiver-noise
regime occur together?**  It is a robustness interaction test, not a new
VTSE-VBF architecture, a model-selection round, or evidence that MFT/VFT have
been repaired.

The two parent replications are H0g (25--30 us, exactly 9 ISRJ replicas under
reference white noise) and H0h (rho=0.80 proper-complex AR(1) receiver noise
under ordinary parameters).  H0s combines these *pre-specified* shifts with a
new source identity range.  No feature, fusion coefficient, checkpoint,
normalization, calibration, threshold, loss, epoch, seed selection, or
external-data refit is permitted.

## New source contract

- Reference-generator base: six labels, train split only, 63 combinations x
  80 samples = 5,040 samples; base seed `810000000` and expected unique
  receiver range `1008677743--1008682782`.
- The ISRJ generator parameters are `sample_width_s=[25e-6,30e-6]` and
  `num_replicas=[9,9]`.  This is feasible because the generator retains its
  physical record-capacity validation.
- The base is emitted to
  `C:\LJQ_code\干扰数据\复合干扰数据0803_isrj_ood_h0s_r1_ar1_base`.
- The I/Q-only transform subtracts the exact reference white realization for
  each saved receiver identity and adds deterministic proper-complex AR(1)
  receiver noise with `rho=0.80` and unchanged noise power.  The derived MAT
  is `C:\LJQ_code\干扰数据\复合干扰数据0803_isrj_ood_h0s_r1_ar1`.
- The audit must establish: source SHA-256, 5,040 unique expected identities,
  zero overlap with all live historical train MATs and recorded H0e/H0f/H0g/
  H0h ranges, active finite ISRJ width/replica values, AR(1) deterministic
  process probe, changed I/Q payload, and exact non-I/Q metadata contract.
- It may read only the test-v3 seal fields and must reject any non-sealed
  condition.  It must not open test-v2, historic locked test, or test-v3 data.

## Exact frozen arms

For seeds `2039--2043`, invoke the audited B0/B2 inference implementation:

| Arm | Frozen contents |
| --- | --- |
| `B0_H1_Frozen` | Focal-36 encoder, H1 linear head, RawSaw route, saved B0 calibration and thresholds |
| `B2_H1AlignedPeriodicity_Frozen` | B0 plus saved aligned 310-D periodicity expert, fusion head, B2 calibration and thresholds |

All five artifacts must be hash-recorded per seed.  H0s has no fitting or
selection split: the new source is inference-only external evidence.

## Endpoints and fixed decision rule

Primary endpoint is paired B2-minus-B0 ISRJ weak-positive-versus-negative
AUROC. Supporting endpoints are ISRJ AUPR, weak and strong recall, FPR,
six-label macro-AUPR/macro-F1, SawFM exact preservation, and non-selection
receiver-cohort FPR tables.

H0s passes only if all of the following hold:

1. mean ISRJ AUPR and weak-AUROC deltas are positive;
2. mean ISRJ weak-recall delta is at least `+0.030`;
3. both weak-recall and weak-AUROC deltas are positive in at least 4/5 seeds;
4. B2 mean ISRJ FPR is at most `0.25` and macro-AUPR delta is non-negative;
5. SawFM AUPR and weak recall are exactly unchanged;
6. source, frozen-artifact and test-v3-seal contracts pass.

The paired bootstrap uses the inherited fixed random-seed schedule of the
frozen H0g runner.  The pass/fail rule is fixed before source generation.  A
failure freezes this interaction source and prohibits adjusting either shift,
any periodicity feature, fusion rule, calibration, threshold, or artifacts to
this result.

## Execution and interpretation limits

The order is source generator smoke/contract test, formal base generation,
formal I/Q transform, materialization and identity audit, one-seed inference
smoke, then the formal five-seed frozen evaluation and descriptive safety
table.  Every receipt records `locked_test_used=false`, `test_v2_used=false`
and `test_v3_used=false`.

A pass only supports a bounded **ISRJ aligned-periodicity robustness** claim
under the joint shift.  It neither converts B2 into the requested unified
six-label VTSE-VBF candidate nor relaxes the remaining MFT/VFT development
gates or the test-v3 prohibition.

## R0 provenance repair before inference

The original, unexecuted source reservation used seed `780000000`.  Its formal
identity audit, performed before any model inference, found a complete
5,040-identity collision with the pre-existing train-only
`复合干扰数据0802_raat_h0_v1` source.  It is therefore **rejected** and no H0s
metric, prediction, model selection, calibration, threshold choice, or test
access was produced from it.  A read-only registry preflight then selected the
first documented unused interval at seed `810000000`.

This is a data-role repair, not response-driven tuning: the ISRJ range,
replica count, AR coefficient, sample count, B0/B2 artifacts, five seeds,
metrics, gates and decision rule above are unchanged.  The rejected R0 files
remain on disk solely as an audit trail and are excluded from all subsequent
H0s-R1 use.
