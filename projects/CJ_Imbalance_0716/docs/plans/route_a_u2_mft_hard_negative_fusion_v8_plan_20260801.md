# Route A U2 MFT hard-negative physical fusion preregistration

## Hypothesis

The frozen final-temporal MFT expert has strict matched AUROC 0.652 but fixed residual formulas do
not separate U2 weak positives from high-order negatives strongly enough. Fit one four-feature,
low-capacity fusion head using the successful ISRJ template:

`[base_mft, expert_z, uncertainty*expert_z, tanh(base_mft)*expert_z]`.

Weak positives and target-absent order>=4 hard negatives each receive weight 4; all other samples
receive weight 1. LogisticRegression C is fixed at 0.1.

## Controls

- P0: frozen U2.
- P1: aligned physical fusion; fixed candidate.
- P2: base-logit-only LogisticRegression capacity control.
- P3: role-wise shuffled expert-z fusion with identical capacity and weights.

Only MFT changes. Non-MFT calibration and thresholds are reused exactly. MFT Platt calibration is
fit on the v3 calibration role; its threshold maximizes weak recall on the v3 threshold role under
overall FPR <=0.20 and order>=4 FPR <=0.30.

## Construction guard

Before development-v8 generation, P1 on v3 early must satisfy:

- macro-AUPR decline versus P0 <=0.005;
- target weak-recall decline <=0.010;
- target weak-AUROC decline <=0.005;
- MFT order>=4 FPR decrease >=0.030;
- MFT weak-vs-order>=4-negative AUROC improvement >=0.010;
- aligned hard-negative AUROC exceeds shuffled by >=0.005 and base-only by >=0.005;
- non-MFT logits exactly invariant and all thresholds feasible.

Passing only authorizes fresh development-v8; it is not promotion evidence.
