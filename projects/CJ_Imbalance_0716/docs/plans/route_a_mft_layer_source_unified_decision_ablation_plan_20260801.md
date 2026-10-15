# Route A MFT expert layer-source unified-decision ablation

Date: 2026-08-01

## Single hypothesis

If a frozen layer carries MFT weak-component evidence that is useful at the
current decision boundary, then with the same zero-parameter bounded residual
the final-temporal, global, stage2, and stage3 source probes will produce
different MFT outcomes. The layer effect must be tested in the final Q4
decision protocol, not inferred only from representation-probe AUROC.

## Frozen arms

`P0_Q4_NoLayerExpert` is the stored current Q4 decision. Each source arm adds
only `0.50 * sqrt(4p(1-p)) * tanh(z_layer)` to its MFT logit, where `z_layer`
is a fit-only standardized score from the corresponding frozen strict
composition-matched MFT probe:

- `P1_Global` (`G_Global`),
- `P2_FinalTemporal` (`T_FinalTemporal`),
- `P3_Stage2` (`S2_Physics`),
- `P4_Stage3` (`S3_Physics`).

Every arm has zero newly trained parameters and leaves all five non-MFT logits
exact. Probes, C6 checkpoint, Q4 logits, residual cap and uncertainty rule are
fixed before execution; this is a factor ablation, not architecture search.

## Roles and gate

Five development-v3 identity-isolated seeds use fit-only score statistics,
calibration-role Platt scaling, threshold-role FPR <= 0.20 selection, and
early-only reporting. A source is marked as a local evidence source only if
it has MFT weak-negative AUROC gain >=0.005, nonnegative MFT AUPR, >=4/5
positive weak-AUROC seeds, mean early MFT FPR <=0.20, and exact non-MFT
preservation. The table is reported in full regardless of passing sources.

No arm trains a new backbone or probe; no candidate is selected, and no
historic locked test, test-v2 or sealed test-v3 may be accessed.
