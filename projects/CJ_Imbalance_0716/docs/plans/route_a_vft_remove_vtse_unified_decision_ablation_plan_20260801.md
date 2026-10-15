# VFT VTSE removal: current unified-decision ablation

## Single mechanism question

Does the aligned strict-weak Ridge-STFT VFT expert make a decision-level
contribution beyond the frozen Q2 base logits and a base-only VFT head when
the same development-v3 calibration and threshold roles are applied anew?

## Frozen inputs and arms

For every identity-isolated seed `2039`--`2043`, the already stored
`all_role_logits.npz` provides all five development-v3 roles. No signal,
backbone, expert, fusion head, optimizer, or hyperparameter is trained or
changed here.

| Arm | VFT decision source | Interpretation |
|---|---|---|
| Q2_NoVTSE | `Q2_H1B2_MFTGuard` VFT logit | VTSE removed |
| Q3_BaseOnlyHead | `Q3_VFTBaseOnlyCapacity` VFT logit | base-only alternative head |
| Q4_AlignedVTSE | `Q4_VFTAlignedRidgeFusion` VFT logit | full aligned strict-weak expert |

Every non-VFT logit must be identical across the three stored variants.
For each arm, Platt calibration is fit only on its calibration role and the
per-label threshold is selected only on its threshold role under FPR `<=0.20`.

## Decision rule

Q4 is credited with a direct VTSE contribution only if it improves early VFT
weak AUROC by at least `+0.010` relative to Q2 and `+0.005` relative to Q3,
while retaining VFT FPR `<=0.20`, non-decreasing VFT AUPR, and at least four
positive weak-AUROC seed differences for each comparison. This is an ablation
only; it cannot promote the external-OOD-failed candidate.

No historical locked test, test-v2, or sealed test-v3 is accessed.
