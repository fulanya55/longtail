# VFT-ISRJ raw-source diagnostic: feasibility failure on development-v3

## Result

The originally proposed source-identification test is **not estimable** under
the current development-v3 label-LT fit/validation roles.  For all five seeds,
the strict weak-VFT-positive versus VFT-absent/ISRJ-present view has zero
exactly context-matched examples in both fit and validation; only early has
the required contexts.

| Role | Weak VFT positives | VFT-absent / ISRJ-present negatives | Matched contexts |
|---|---:|---:|---:|
| fit, every seed | 0 | 0 | 0 |
| validation, every seed | 0 | 0 | 0 |
| early | 277--307 | 640 | 16 |

This is a data-role estimability finding, not a model score.  It prevented raw
feature extraction and fitting.  Relaxing co-label matching after observing
this failure would allow SinFM/MFT co-occurrence shortcuts, so the
development-v3 source test is stopped.

## Fresh source-v1 follow-up

A new train-only, identity-disjoint balanced source diagnostic was generated
instead: 5,040 unique receiver identities, 63 exact combinations x 80 samples,
receiver range `[352677743, 352682782]`, and five per-combination 48/16/16
fit/validation/early role partitions.  Its identity, coverage, and strict
source-view estimability audit passed all checks.

On source-v1, phase organisation is demonstrably informative but the proposed
32-dimensional summary is insufficient to replace the full 16x16 Ridge-STFT
descriptor:

| Comparison on early hard source view | Mean delta | 95% bootstrap CI |
|---|---:|---:|
| RawSource minus RidgeSTFT AUROC | -0.06972 | [-0.08847, -0.05097] |
| RawSource minus RidgeSTFT AUPR | -0.10553 | [-0.13598, -0.07494] |
| RawSource minus PhaseScrambledControl AUROC | +0.08470 | [+0.05474, +0.11457] |
| RawSource minus PhaseScrambledControl AUPR | +0.08582 | [+0.06995, +0.10243] |

The RawSource validation-to-early AUROC gap is only `+0.01359`, so the failure
is not an overfit/selection effect.  Its phase-sensitive information is real,
but compressing it into four lag, segment-copy, ridge-continuity, and reset
statistics loses discriminative information retained by the high-dimensional
STFT grid.  All five roles have negative RawSource-minus-RidgeSTFT early AUROC.

## Decision

Stop the **low-dimensional raw summary as a replacement VFT expert**.  Do not
connect it to the frozen candidate, do not tune its lags/statistics, and do not
create development-v13 from this result.  A future source route would require
a separately pre-registered, capacity-controlled *complementarity* test (full
STFT plus fixed source features versus equal-capacity nonphysical features),
not a new arbitrary fusion head.

## Artifacts and boundary

- development-v3 feasibility runner:
  `experiments/run_route_a_vft_isrj_raw_source_diagnostic.py`
- fresh generator:
  `experiments/generate_route_a_vft_isrj_source_v1.m`
- identity/role audit:
  `experiments/audit_route_a_vft_isrj_source_v1.py`
- source-v1 feature runner:
  `experiments/run_route_a_vft_isrj_source_v1_raw_feature_diagnostic.py`
- source-v1 audit:
  `outputs/tapc/route_a_vft_isrj_source_v1_generation/source_v1_identity_role_audit.json`
- source-v1 metrics:
  `outputs/tapc/route_a_vft_isrj_source_v1_raw_feature_diagnostic/route_a_vft_isrj_source_v1_raw_feature_summary.json`

All artifacts record `locked_test_used=false`, `test_v2_used=false`, and
`test_v3_used=false`.
