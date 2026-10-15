# Route A H0c: frozen predicted-width substitution for the ISRJ delay map

## Single falsifiable hypothesis

H0b demonstrated a strong **oracle** width-normalized repeated-copy map. H0c
tests whether the five already frozen v16 I/Q width probes are accurate enough
to provide that map coordinate on a third fresh identity source. This is the
necessary bridge from physical oracle evidence to a future inference-time
representation.

## New source and frozen dependencies

- Generate a fresh train-only balanced source with seed `720000000`, 5,040
  identities, ISRJ width `[10,30] us`, replicas `[3,8]`, no test split.
- Receiver modulo-10 roles are ISRJ-positive fit/early/evaluation
  `1536/512/512`; source audit must prove zero overlap with all earlier sources.
- For seed `s`, load only frozen v16 `W_real/seed_s/best_model.pth` and the
  frozen v16 fit normalizer. No width probe is refit, selected or calibrated.

## Three matched arms per seed

All three arms share the same 2,497-parameter CNN, optimizer, budget, role
indices and replica target normalization.

1. `O_real`: H0b oracle-width map, true replica target (fidelity reference).
2. `P_real`: map widths are the frozen v16 predicted widths, rounded and
   clamped only to the declared physical `[100,300]` samples; true replica target.
3. `P_perm`: same predicted-width map and initialization as `P_real`, but a
   fixed fit/early replica-label permutation.

`P_real`/`P_perm` are initialized identically and therefore provide the primary
information-control contrast. The oracle arm is not a classifier baseline; it
quantifies representation loss due solely to substituting predicted width.

## Gates

On held-out identities, H0c passes only if:

1. `P_real` mean R2 >= `0.20` and Spearman >= `0.40`;
2. `P_real - P_perm` mean R2 >= `+0.20`, paired 95% lower CI > 0, and 4/5
   positive seed contrasts;
3. `P_real - O_real` mean R2 >= `-0.15` with lower paired CI >= `-0.20`;
4. frozen checkpoint/normalizer hashes, role/source contracts and test seal pass.

Failure stops predicted-width substitution; H0b remains an oracle-only
mechanism result. A pass authorizes a separate H1 source-level ISRJ weak versus
composition-matched-negative test with a zero-initialized local branch and an
information-destroyed map control. It does not authorize a classifier change
or test-v3.
