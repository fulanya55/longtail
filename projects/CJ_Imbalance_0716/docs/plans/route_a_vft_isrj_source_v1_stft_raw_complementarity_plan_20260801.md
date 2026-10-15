# VFT--ISRJ source-v1: STFT plus raw-source complementarity test

## Single falsifiable hypothesis

The previously rejected 32-dimensional raw-source descriptor may still carry
phase-organisation information that is complementary to, rather than a
replacement for, the 16x16 Ridge-STFT VFT descriptor.  If so, appending the
descriptor to Ridge-STFT should beat both Ridge-STFT alone and an exactly
equal-dimensional phase-scrambled-descriptor control on the held-out early
source role.

## Fixed source protocol

- Source: the already generated, audit-passed `vft_isrj_source_v1` train-only
  MAT file (5,040 receiver-identities; exact identity-disjoint 48/16/16
  fit/validation/early role partitions within every composition).
- Task: strict weak VFT-positive versus VFT-absent/ISRJ-present negative,
  matched on the first four co-labels.  Strong VFT positives are excluded.
- Replications: frozen source roles `2039`--`2043`.
- Features: fixed 256-dimensional 16x16 STFT and the already-defined
  32-dimensional phase/copy/ridge/reset descriptor.  Each feature family is
  fit-only z-standardised before concatenation.
- Classifier: `RidgeClassifier(alpha=10)`, intercept enabled; positive and
  negative total sample masses are each 0.5.

## Arms and capacity control

| Arm | Feature dimension | Purpose |
|---|---:|---|
| RidgeSTFT | 256 | Current high-dimensional reference |
| RidgeSTFTPlusRawSource | 288 | Candidate complementarity test |
| RidgeSTFTPlusPhaseScrambled | 288 | Equal-dimensional nonphysical control |

The control retains the STFT input and all 32 added dimensions, but destroys
the raw descriptor's time-phase organisation deterministically.  There is no
feature, alpha, loss, or threshold search.

## Metrics and pre-registered decision

Primary metrics are early AUROC and AUPR.  The candidate must satisfy all of:

1. Raw-augmented minus phase-scrambled-control early AUROC and AUPR means are
   each at least `+0.010`;
2. Raw-augmented minus RidgeSTFT early AUROC mean is at least `+0.010` in at
   least four of five role replications;
3. Raw-augmented validation-to-early AUROC gap is below `0.150`.

Passing would establish only fixed-feature *source-level complementarity*.  It
would not change the frozen unified candidate.  Failure stops this raw-source
path; no lags, statistics, STFT resolution, or regularisation will be tuned.

## Boundaries and artifacts

No calibration, threshold, fusion, external OOD data, historical locked test,
test-v2, or sealed test-v3 may be used.  The runner stores role-index and data
SHA-256 values, feature normalisation statistics, coefficients, per-seed
metrics, and the three test-use flags.
