# Route A MFT residual-cap removal: threshold-role protocol implementation repair

Date: 2026-08-01

## Reason for this repair

The original cap-removal plan specified a per-arm Platt calibrator followed by
threshold selection on the pre-assigned `threshold` role under FPR <= 0.20.
Its runner instead evaluated the `early`-role observed FPR as the hard
feasibility check.  Since `early` is an independent evaluation role, that is a
reportable generalization diagnostic, not the threshold-selection constraint.
The historical failed run remains immutable evidence; this runner does not
overwrite it or alter a model, fusion rule, threshold rule, or decision.

## Single question

Under the exact registered, role-correct procedure, is the narrow MFT
`cap=1.0` versus `cap=0.75` contrast feasible at the assigned threshold role,
while its independent early-role FPR is transparently reported?

## Frozen replay

- Reuse only stored Q1/Q4 logits and frozen aligned peak-negative evidence.
- Reconstruct cap 0.75 and require <= 1e-6 MFT reconstruction error; cap 1.0
  changes only the residual multiplier.  All non-MFT Q4 logits are exact.
- Refit per-arm Platt only on `calibration`; select thresholds only on
  `threshold` with the unchanged max FPR 0.20 and weak-recall objective.
- Record, separately, MFT threshold-role FPR, threshold, early-role FPR, and
  all existing early weak/strong/ranking metrics.
- Five identity-isolated development-v3 seeds; no training, optimization,
  candidate choice, test-v2, historic locked test, or test-v3 access.

## Decision rule and boundary

Protocol feasibility requires exact reconstruction/non-MFT preservation and
threshold-role MFT FPR <= 0.20 for both arms in all five seeds.  Early-role FPR
is never silently substituted for feasibility: it is a secondary generalization
result and must be reported even if >0.20.  This completes the *mechanism
ablation implementation*, but does not authorize cap=1.0 as a candidate
change, since fresh confirmation and external-OOD gates remain separate.
