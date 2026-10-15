# Route A M5 plus periodicity fresh development-v8 preregistration

## Mechanism

FullAlpha1p5 extrapolates beyond the M5 label-deletion endpoint and gives the strongest average
weak-positive gain, but development-v7 exposes excessive MFT FPR in target-absent high-order
compositions. Test one lower-risk candidate: use the actual M5 endpoint (alpha=1.0) with the same
RawSaw route and aligned ISRJ periodicity fusion. Inference parameter count is unchanged.

## Frozen variants

- Q0: C6 + RawSaw.
- Q1: M5 + RawSaw.
- Q2: M5 + RawSaw + aligned periodicity; fixed candidate.
- Q3: M5 + RawSaw + block-shuffled periodicity.
- U2: the existing FullAlpha1p5 aligned-periodicity candidate, retained as accuracy reference.

Fit Q2/Q3 fusion heads only on development-v3 fit, with the same four features, C=0.1, and weak
positive weight 4 used by U2. Calibrate and select thresholds on the same v3 roles and standard
overall FPR<=0.20 protocol. Do not add a subgroup-specific threshold.

## Fresh data

Generate one train-only balanced development-v8, base sample seed 84000000, 25,200 new identities,
expected receiver-noise seeds [282677743,282702942]. Generate no test-v8. Keep test-v3 sealed.

## One-shot Q2 gates

Q2 versus Q0 must satisfy all original unified gates: macro-AUPR +0.005, macro-F1 +0.010,
target weak recall +0.030, target weak AUROC +0.015, target mean FPR <=0.20, target strong-recall
and AUPR noninferiority, 4/5 positive seeds, and exact SawFM preservation.

Safety and mechanism gates:

1. worst supported target composition-order FPR <=0.30;
2. worst target/order FPR increase versus Q0 <=0.05;
3. MFT order-5 FPR at least 0.05 lower than U2;
4. Q2 macro-AUPR no more than 0.005 below U2;
5. Q2 target weak recall no more than 0.015 below U2;
6. Q2 versus Q3 ISRJ weak AUROC >=+0.005;
7. Q2 versus Q1 ISRJ weak recall >=+0.010 and weak AUROC >=+0.005;
8. external_refit=false and test-v3 boundary intact.

Failure forbids alpha interpolation or v8 threshold repair.
