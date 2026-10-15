# MFT/VFT dual-target decision-contract diagnostic plan

## Hypothesis

The external failure is caused by target-specific replacement logits and
separately fitted decision maps drifting under cross-label ISRJ OOD.  A shared
parent decision contract should retain expert ranking gains while bounding
threshold displacement.

## Fixed candidate

- Keep H1, RawSaw, ISRJ periodic expert, MFT/VFT physical experts, and learned
  fusion heads frozen.
- Use the H1+B2 parent Platt calibration and threshold for MFT and VFT.
- Reduce the MFT negative residual cap from `0.75` to a single preregistered
  common cap `0.50`.
- Replace the VFT full-logit substitution by
  `0.50 * uncertainty(parent) * tanh(expert_logit - parent_logit)`.
- Use the same operation for aligned, base-only-capacity, and shuffled VFT
  controls.
- No cap grid, threshold search, refit, or recalibration.

## Evidence roles

- development-v3: construction replay.
- development-v12: development diagnostic; it was already used by the FC-PTV
  line and is not globally unseen.
- ISRJ parameter OOD v1: failure diagnostic after candidate-v2 evaluation.
- future development-v13: generated only if this mechanism passes all diagnostic
  gates; it will be the fresh frozen confirmation.
- test-v3: sealed.

## Gates on each of the three diagnostic datasets

1. target mean FPR at most `0.20`;
2. target weak-recall gain over H1 at least `0.03`;
3. macro-AUPR gain over H1 at least `0.005` for v3/v12 and nonnegative on OOD;
4. MFT hard-negative FPR delta versus H1+B2 at most `-0.01`;
5. MFT weak-recall loss versus H1+B2 no worse than `-0.02`;
6. VFT FPR at most `0.20`;
7. VFT weak AUROC gain versus H1+B2 at least `0.01`;
8. aligned VFT weak AUROC exceeds capacity and shuffled controls by `0.005`;
9. SawFM metrics and non-target decisions are exactly preserved.

Failure stops this policy.  Passing allows a pre-evaluation freeze and one
fresh development-v13 confirmation.

