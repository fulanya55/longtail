# VFT core ablation: weighting mechanism and uncertainty gate

## Single mechanism question

Under the *same frozen H1+B2+MFT-guard base logits, the same fixed Ridge-STFT
VFT score, the same four-input VFT logistic head, and the same role protocol*,
does the VFT decision benefit from (a) only weak-positive upweighting, (b)
three-stratum equal mass, or (c) the uncertainty gate itself?  Previous
studies changed more than one of these factors at once and cannot answer this
question directly.

## Fixed protocol

- development-v3 SHA-256 `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`;
  identity-isolated seeds `2039`--`2043` and their frozen fit, validation,
  early, calibration, and threshold roles;
- fixed base logits are `Q2_H1B2_MFTGuard`; the fixed VFT source score is the
  existing 16x16 Ridge-STFT score (`alpha=10`), normalised on strict-weak VFT
  positives plus VFT negatives in the fit role only;
- each arm fits only a VFT `LogisticRegression(C=0.1)` with four inputs
  `[base, z, gate(base)*z, tanh(base)*z]`; it has four coefficients plus one
  intercept.  All non-VFT logits must be exactly unchanged;
- Platt calibration uses the calibration role only; per-label thresholds use
  the threshold role only and maximise weak recall under negative FPR `<=0.20`.

## Pre-registered arms

| Arm | Fit weighting | Gate |
|---|---|---|
| A0_UniformBCE_NoGate | all VFT labels weight 1 | constant 1 |
| A1_WeakWeight_NoGate | weak positives weight 4; all other labels weight 1 | constant 1 |
| A2_ThreeStratum_NoGate | weak positive, strong positive, negative each have total mass 1/3 | constant 1 |
| A3_WeakWeight_SqrtGate | weak positives weight 4; all other labels weight 1 | `sqrt(4p(1-p))` |

`A3` must exactly reproduce the previous Q4 VFT logits.  Thus the comparisons
are A1-A0 (only weak-positive weighting), A2-A0 (three-stratum equal mass),
A2-A1 (stratification beyond weak-only weighting), and A3-A1 (uncertainty
gate under the same weight pattern).

## Decision gates

This is a construction ablation, not candidate selection.  A mechanism is
called supported only when its paired early weak-AUROC delta is at least
`+0.015`, weak-recall delta at least `+0.030`, both are positive in at least
four of five seeds, target FPR remains `<=0.20`, and strong-recall loss is no
worse than `0.010`.  For the gate, the safety condition is instead: no
weak-AUROC loss below `-0.010` while target FPR does not increase by more than
`0.010`.

Failure stops the corresponding mechanism claim.  No hyperparameter, gate
shape, threshold, calibration, base model, or frozen candidate is changed.

## Evidence boundary

This runner never reads historical locked-test, test-v2, or sealed test-v3
data.  It preserves coefficients, normalisation values, calibration,
thresholds, early predictions, per-seed receipts, and all test-use flags.
