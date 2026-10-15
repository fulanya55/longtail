# Route A VFT: fixed-capacity MLP physical-evidence control

Date: 2026-08-01

## Trigger and hypothesis

The completed 7-parameter MLP versus 7-parameter fixed-statistics linear
ablation shows that a small nonlinear decision head can improve VFT metrics.
It therefore rejects the claim that head capacity is irrelevant, but does not
test whether aligned Ridge-STFT evidence adds information beyond that capacity.

Hypothesis: with *identical* 7-trainable-parameter MLP architecture,
deterministic optimization, fit weak-positive weighting, calibration and FPR
threshold protocol, the aligned Ridge-STFT inputs improve VFT weak-negative
ranking over both a no-VTSE MLP and a role-wise shuffled VTSE MLP.

## Arms

All arms use `Linear(4,1) -> Tanh -> Linear(1,1)` (exactly seven trainable
parameters) and replace only the VFT logit of frozen `Q2_H1B2_MFTGuard`.

| Arm | Four fixed inputs |
|---|---|
| `MLP1_NoVTSECapacity` | `base`, `tanh(base)`, `base^2`, `abs(base)` |
| `MLP1_AlignedVTSE` | `base`, aligned Ridge-STFT `z`, `uncertainty*z`, `tanh(base)*z` |
| `MLP1_ShuffledVTSE` | the aligned four-input form with role-wise seeded permutation of `z` |

No arm changes the feature extractor, Ridge-STFT fit, MFT guard, non-VFT
logits, source roles, or test boundary.  The no-VTSE arm is deliberately a
strong nonlinear base-only control rather than a lower-capacity linear refit.

## Fixed training and decision protocol

- development-v3 seeds 2039--2043; source SHA and identity roles are checked.
- Fit weak VFT positives weight 4; all remaining examples weight 1.
- Fit-only standardization; full-batch AdamW (`lr=0.01`, `weight_decay=1e-4`),
  200 epochs maximum, patience 20 based only on validation-role VFT AUPR.
- Independent Platt calibration and FPR <= 0.20 thresholding use the existing
  frozen evaluation protocol for every arm.
- `test-v2`, historic locked tests and sealed test-v3 are prohibited.

## Gate and action

Aligned VTSE passes its physical-information gate only when versus *both*
no-VTSE and shuffled VTSE it has weak-negative AUROC gain >=0.005, nonnegative
VFT AUPR, and >=4/5 positive weak-AUROC seeds; non-VFT logits must be exact.
Failure stops the VFT fusion claim.  Passing isolates evidence beyond MLP
capacity but still does not promote the external-OOD-failed candidate or permit
test-v3 use.
