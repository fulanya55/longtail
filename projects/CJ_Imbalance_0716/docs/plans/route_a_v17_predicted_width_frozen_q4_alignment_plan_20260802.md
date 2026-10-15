# Route A v17: frozen predicted-width / Q4 VFT-risk alignment audit

## Question and hypothesis

The completed v16/v17 evidence shows that a small frozen I/Q width probe can
recover an ISRJ pulse-width intervention.  This is not yet evidence that the
signal is relevant to the frozen shared classifier's VFT hard-negative risk.

**H-Q4:** on the fixed paired v17 intervention, a larger frozen v16 predicted
ISRJ width is accompanied by a larger fixed-Q4 VFT probability and fixed-Q4
VFT false-positive rate on the `VFT=0 AND ISRJ=1` subset.

This is a diagnostic of a pre-existing frozen decision rule, not a new model
or a correction fit.  v17 has already been used for width transport, so it
cannot be used to select the coefficient, threshold, or architecture of a
future width-conditioned correction.  Any such candidate must use fresh,
identity-isolated development data.

## Frozen protocol

- **Data:** v17 paired external-physics train-role files only:
  `W10_15_R3_4` (narrow) and `W22_30_R3_4` (wide), each 1,260 receiver
  identities.  The audit establishes matched labels/nuisance/identity and 320
  `VFT=0, ISRJ=1` hard negatives per regime.
- **Width signal:** saved predictions from the five frozen v16 `W_real`
  probes (`2039..2043`) and the saved fit-only normalizer.  No width-model
  refit, normalization update, or parameter selection is allowed.
- **Classifier:** the immutable
  `H1_B2_MFTSharedDecision_VFTRidgeFusion_v2` manifest.  Reconstruct its
  frozen five Q4 inference paths, including the original calibration and the
  original per-label thresholds.  No classifier checkpoint, fusion feature,
  calibration, threshold, or decision rule is changed.
- **Endpoint:** VFT score/probability and binary prediction, restricted to the
  paired hard-negative identities.  The width delta is wide minus narrow;
  risk deltas use exactly the same direction.
- **Forbidden roles:** historic locked test, test-v2, sealed test-v3, and any
  v17 test role are not opened or used.

## Prespecified statistics and gates

For each fixed classifier seed, record (a) mean paired VFT-probability delta,
(b) paired VFT-FPR delta at the frozen threshold, and (c) Spearman association
between the concatenated frozen predicted width and Q4 VFT probability across
the 640 hard-negative views.  Aggregate seed means with a seed-level paired
bootstrap 95% CI.

The Q4-risk alignment screen passes only if all of the following hold:

1. Mean paired Q4 VFT-probability delta is at least `0.020` and positive for
   at least 4/5 seeds.
2. Mean paired Q4 VFT-FPR delta is at least `0.050` and positive for at least
   4/5 seeds.
3. Mean width--Q4-probability Spearman is at least `0.20` and positive for at
   least 4/5 seeds.
4. The hard-negative receiver order, labels, and Q4 thresholds are exactly
   identical across narrow/wide paired regimes for every seed.
5. v17 generation/audit and v16/v17 width-transport gates remain passed; the
   test-v3 seal remains intact.

The practical magnitudes are deliberately small relative to the previously
observed direct width intervention, because this screen asks only whether the
fixed classifier exposes a measurable *risk ordering*, not whether it is
already a useful correction rule.

## Execution and decision

First run one frozen inference seed as a smoke-chain check in a separate
output directory; it has no evidential role.  Then repeat the entire fixed
inference audit over all five frozen seeds, preserving Q4 prediction files,
metrics, source hashes, and receipt.

- **Pass:** permit a separately preregistered, fresh-data-only bounded
  width-conditioned residual experiment.  The present results provide no
  correction coefficient or threshold.
- **Fail:** stop width-conditioned correction; predicted physical width is
  then not demonstrated to be decision-relevant to Q4 VFT risk.

