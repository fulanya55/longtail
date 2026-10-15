# Route A development-v3 Focal-72 cosine convergence audit

## Falsifiable question

The existing 36-epoch constant-learning-rate Focal baseline still has several
best validation epochs at its budget cap. This audit tests one fixed stronger
baseline, rather than searching loss settings: can a 72-epoch cosine-decayed
Focal run remove or materially reduce the frozen ISRJ periodicity/H1 evidence?

## Fixed protocol

- Existing development-v3 source only, SHA-256
  `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`;
  it is a baseline-convergence audit, not a new candidate-selection source.
- Seeds `2039--2043`, existing role indices, F2 initialization, partial shared
  scope, batch order, RawSaw SawFM routing, Platt calibration role and
  per-label FPR<=0.20 threshold role.
- Train only `Focal(gamma_pos=0,gamma_neg=3)` from F2 initialization for an
  unconditional 72 epochs. AdamW classifier LR is `5e-4`, shared LR is
  `5e-5`, weight decay `1e-4`, batch 64, clipping 5, SawFM loss weight 0.2.
- Cosine learning-rate multiplier decreases from 1.0 to 0.01; final LRs are
  `5e-6` and `5e-7`. No early stopping, loss-parameter, LR, calibration,
  threshold or decision-objective search is allowed.
- Save every epoch history, checkpoint, model/prediction hashes, calibration,
  thresholds and metrics. Compare descriptive paired deltas to Focal-36,
  frozen H1 linear BCE and F3; Focal-72 is not allowed to select a model.

## Decision rule and boundary

The audit reports whether the best validation epoch reaches 72 and the last
ten validation changes. It does not permit extending another schedule after
seeing results. If Focal-72 remains below H1 or fails to erase the bounded
ISRJ evidence, it strengthens the baseline audit but does not promote B2/F3
to a complete six-label candidate. If it improves, it becomes the stronger
ordinary-loss reference and narrows earlier claims.

No locked test, test-v2 or sealed test-v3 may be opened, calibrated, analyzed
or used. All receipts must retain the three corresponding false flags.
