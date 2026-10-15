# Route A MFT one-sided bounded co-occurrence safeguard preregistration

## Failure diagnosis carried forward

The first replacement safety head is stopped before fresh-v8 evaluation because its held-out
sanity result destroyed U2 ranking. The likely cause is an unconstrained score replacement fitted
from a small role, not evidence that co-occurrence risk is absent.

## Single revised hypothesis

Only MFT violates the absolute worst-order FPR gate. Preserve the frozen U2 MFT score and permit
only a downward, bounded correction inferred from aligned co-occurrence evidence:

`new_logit = base_logit + min(0, 0.5 * tanh(head_logit - base_logit))`.

The head uses the same nine features as the failed construction, LogisticRegression C=0.01,
weak-positive weight 4, and equal total negative mass across composition orders. It cannot increase
any MFT logit and cannot change it by more than 0.5. All other five label logits remain unchanged.

## Controls

- R0: frozen U2.
- R1: aligned, order-balanced, one-sided bounded MFT safeguard; fixed candidate.
- R2: row-shuffled co-occurrence control with identical capacity and bound.
- R3: aligned but unweighted-negative ablation.

MFT Platt calibration and threshold are fitted on the existing v3 calibration/threshold roles.
The MFT threshold maximizes weak recall under overall FPR <=0.20 and order>=4 negative FPR <=0.30.
All non-MFT calibration and thresholds are reused exactly.

## Fail-fast construction guard

Before generating development-v8, the fixed R1 v3 early diagnostic must have:

- macro-AUPR decline versus R0 no worse than 0.005;
- target mean weak-recall decline no worse than 0.010;
- target mean weak-AUROC decline no worse than 0.005;
- non-MFT logits exactly invariant;
- feasible MFT threshold in all five seeds.

This guard rejects a broken construction but is not promotion evidence. If it passes, generate the
fresh development-v8 specified in the preceding preregistration and apply the full one-shot gates.
