# Final shared-policy candidate: preregistered v12 stratified audit

## Fixed inputs

- Candidate: `H1_B2_MFTSharedDecision_VFTRidgeFusion_v2`.
- Baseline: `Q0_H1`.
- Dataset: identity-audited development-v12 only.
- Predictions, calibration, and thresholds are frozen; no refit or selection.
- test-v3 remains sealed.

## Strata

For MFT, VFT, and ISRJ report:

- weak and strong positives;
- target power relative to the strongest coexisting component;
- nominal JNR;
- label-specific physical parameter bins;
- composition order 1--6;
- every co-label present and absent;
- five deterministic receiver-noise cohorts.

All strata must report sample, positive, negative, weak-positive, and
strong-positive counts.  AUPR/FPR require at least 100 positives and 100
negatives; weak metrics require at least 30 weak positives.

Because six labels have all 63 non-empty label combinations in this generator,
there is no mathematically available unseen label-set combination.  The audit
must state this limitation; unseen physical configurations are handled by the
separate parameter/noise OOD stage.

## Preregistered gates

1. No more than 20% of supported target strata may have AUPR delta below
   `-0.01`.
2. At least 70% of supported VFT and ISRJ weak strata must have positive weak-
   recall delta.
3. At least 70% of supported MFT strata must have non-increased negative FPR.
4. Every receiver cohort must have positive mean target weak-recall delta.
5. Worst supported ISRJ subgroup FPR must not exceed `0.35` and must be named.
6. SawFM probability and decisions must be exactly preserved for every seed.

Failure does not authorize threshold tuning on v12.  It triggers either a
narrowed claim or a new mechanism evaluated on a fresh development identity.

