# Route A U2 unified-candidate development-v7 frozen confirmation preregistration

## Data and evidence boundary

- Generate one train-only, balanced development-v7 set with 25,200 new receiver identities.
- Base sample seed: `74000000`; sample seeds: `[74000001, 74025200]`; expected receiver-noise seeds: `[272677743, 272702942]`.
- The identity range must be disjoint from development-v2 through v6, existing test contracts, ISRJ OOD-v1, and sealed test-v3.
- Do not generate test-v7. Keep test-v3 at `access_authorized=false`, `analysis_allowed=false`, and `used=false`.

## Frozen evaluation

For every training seed, reuse the artifacts fitted on development-v3:

- C6 and FullAlpha1p5 checkpoints;
- RawSaw checkpoint;
- aligned and shuffled 310-dimensional HGB experts plus frozen development-v3 score-standardization statistics;
- U2 and U3 LogisticRegression fusion heads and normalization parameters;
- development-v3 Platt calibrators and per-label thresholds for U0, U1, U2, and U3.

Development-v7 forbids retraining, refitting expert-score standardization, recalibration, threshold reselection, or seed selection. Set `external_refit=false`.

## One-shot confirmation gates

U2 versus U0 must jointly satisfy:

1. macro-AUPR `>= +0.005`;
2. macro-F1 `>= +0.010`;
3. target mean weak recall `>= +0.030`;
4. target mean weak AUROC `>= +0.015`;
5. absolute U2 target mean FPR `<= 0.20`;
6. no MFT, VFT, or ISRJ AUPR decline larger than `0.010`;
7. target mean strong recall decline no larger than `0.010`;
8. positive target weak-recall gain in at least four of five training seeds;
9. U2 versus U1 ISRJ weak recall `>= +0.010` and weak AUROC `>= +0.005`;
10. U2 versus U3 ISRJ weak AUROC `>= +0.005`;
11. exactly invariant SawFM metrics;
12. `external_refit=false` and intact test-v3 boundary.

If any gate fails, no development-v7 repair is allowed. U2 remains development-v3 combination evidence only.
