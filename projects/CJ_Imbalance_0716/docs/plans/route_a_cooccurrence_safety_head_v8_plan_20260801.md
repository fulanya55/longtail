# Route A co-occurrence-conditioned safety head and development-v8 preregistration

## Falsifiable mechanism

The frozen U2 candidate preserves weak evidence but over-scores target-absent, high-order
compound negatives. Test one mechanism only: a low-capacity target-conditional safety head
uses the frozen target logit plus the five aligned non-target logits to suppress co-occurrence
confusions. Negative training mass is equalized across target-absent composition orders.

## Frozen base and variants

- S0: frozen U2, unchanged.
- S1: aligned co-occurrence features plus order-balanced negative weights; preregistered candidate.
- S2: row-shuffled co-occurrence features with identical dimensionality and weighting; alignment control.
- S3: aligned features with ordinary negative weights; weighting ablation.

For each of MFT, VFT, and ISRJ, the nine features are target logit, five sigmoid-transformed
other-label logits, their sum, maximum, and standard deviation. Each head is LogisticRegression
with fixed C=0.1. Weak positives receive weight 4.0. In S1/S2, each negative composition order
receives equal total mass. FMB, SawFM, and SinFM remain bitwise unchanged.

## Training, calibration, and threshold roles

- Fit heads only on the existing development-v3 fit role.
- Fit target-label Platt calibration only on the development-v3 calibration role.
- Reuse U2 calibration and thresholds exactly for FMB, SawFM, and SinFM.
- Select each target threshold only on the development-v3 threshold role by maximizing weak
  recall subject to overall target-negative FPR <=0.20 and order>=4 hard-negative FPR <=0.30.
- No hyperparameter sweep and no selection on development-v7.

## Fresh confirmation data

Generate one train-only balanced development-v8 with base seed 84000000 and 25,200 new
receiver identities. Generate no test-v8. Evaluate S0-S3 once with all heads, calibrators, and
thresholds frozen. test-v3 remains sealed and unused.

## One-shot S1 gate on development-v8

S1 must satisfy all original unified gates versus U0: macro-AUPR +0.005, macro-F1 +0.010,
target weak recall +0.030, target weak AUROC +0.015, target mean FPR <=0.20, target strong
recall noninferiority, target AUPR noninferiority, 4/5 positive seeds, and exact SawFM preservation.

Additionally:

1. worst supported target/order FPR <=0.30;
2. worst target/order FPR increase versus U0 <=0.05;
3. target weak recall decline versus S0 no worse than 0.010;
4. target weak AUROC decline versus S0 no worse than 0.005;
5. S1 worst-order FPR is at least 0.02 lower than S3 or S2;
6. S1 has positive weak-AUROC advantage over S2 in at least two of three targets;
7. external_refit=false on development-v8 and test-v3 boundary intact.

Failure forbids development-v8 threshold or feature repair.
