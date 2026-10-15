# Route A v23: weak-positive/hard-negative ranking morphology expert

## Single hypothesis

v22 rejected global weak-positive reweighting: equal positive mass did not improve weak VFT ranking. **H-v23:** an explicit pairwise ordering loss between weak VFT positives and VFT-absent/ISRJ-present hard negatives, added to ordinary BCE, improves weak evidence ranking while preserving strong and overall ranking.

## Fresh protocol

- Generate a new train-only source at base seed `640000000`, with 5,040 unique receiver identities, the unchanged 63-combination balanced generator, ISRJ width `[10,30] us`, replicas `[3,4]`, and no test split.
- Use only ISRJ-present examples; split receiver identities by modulo 10 into fit `{0..5}`, early-stop `{6,7}`, and evaluation `{8,9}`. Require at least 20 weak positives, 20 strong positives, and 20 hard negatives in every role.
- Inputs remain the non-learned 268-dimensional v19 STFT morphology feature. Both heads remain exactly `268 -> Linear(64) -> GELU -> Dropout(0.10) -> Linear(1)` with 17,281 parameters.
- `M_BCE` uses ordinary BCE. `M_WPR` uses `BCE + 0.25 * mean(softplus(-(s_weak - s_hard)))` over all weak-positive/hard-negative pairs within each ordinary random mini-batch. The coefficient `0.25` is fixed once here; no sweep is permitted. Strong positives remain supervised by the shared BCE term.
- AdamW (`3e-4`, decay `1e-4`), batch 64, 30 epochs, identical initialization, and the same unweighted three-stratum early-stop loss are used. No sampler, calibration, threshold, Q4 output, width, old v19-v22 data, locked test, test-v2, or test-v3 is accessed.

## Held-out gate

On five model seeds, promotion requires weak AUROC gain `>=+0.015`, paired bootstrap lower CI `>0`, and positive gain in at least 4/5 seeds; weak AUPR gain `>=+0.010`; and no per-seed strong or overall AUROC loss below `-0.010`. The source audit, exact matched capacity, hashes, and test-v3 seal must also pass.

Failure rejects weak-pair ranking as the morphology remedy. Passing permits a separate fresh-source fusion design only; it does not authorize a Q4 modification or test-v3 access.
