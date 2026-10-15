# VFT strict-mask versus all-positive supervision: unified decision ablation

## Single mechanism question

Does the strict weak-positive mask remain necessary after its Ridge-STFT score
is integrated into the same current Q4 VFT fusion head, calibration role, and
FPR-constrained threshold protocol? This directly upgrades the earlier
construction-only weak/mixed comparison to the six-label decision level.

## Fixed protocol

- development-v3, identity-isolated seeds `2039`--`2043`;
- frozen Q2 H1+B2+MFT-guard logits, fixed 16x16 STFT, fit-only feature
  standardisation, `RidgeClassifier(alpha=10)`, and the identical four-input
  Q4 VFT logistic head (`C=0.1`, weak-positive head weight 4);
- all non-VFT logits are required to be exactly unchanged;
- Platt calibration and per-label FPR `<=0.20` weak-recall threshold selection
  use only their assigned roles.

## Arms

| Arm | Ridge training positives | Ridge negatives |
|---|---|---|
| StrictWeak_Q4 | frozen historical Q4 strict weak-positive Ridge score | matched VFT-negative score construction |
| AllPositive_NoWeakMask | all composition-matched VFT positives, including strong positives | the corresponding composition-matched negatives |

The all-positive arm removes the weak mask and necessarily admits strong VFT
positives; it is therefore the direct unified-policy realization of core
ablations 2 and 3. The Q4 arm's VFT early logits must reproduce the stored
Q4 logits exactly.

## Gate and action

AllPositive_NoWeakMask could challenge strict supervision only if early VFT
weak AUROC improves by `>=0.015`, weak recall by `>=0.030`, both gains occur
in at least four seeds, VFT FPR remains `<=0.20`, and strong-recall decline is
at most `0.010`. Otherwise strict-mask supervision remains the only supported
choice and the all-positive route stops. This is an ablation, not candidate
selection; no result changes test-v3 eligibility.

No historical locked test, test-v2, or sealed test-v3 data may be read.
