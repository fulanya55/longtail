# Route A H0: ISRJ delay-block self-similarity identifiability screen

## Single falsifiable hypothesis

In the configured generator, ISRJ retransmits the same captured waveform at
integer multiples of its unknown sample width. The stopped copy-consistency
route reduced that structure to one ACF-selected lag and 20 summary numbers.
**H0-ISRJ** tests a non-overlapping claim: a fixed two-dimensional map of
complex local correlations over the *entire feasible delay band* retains enough
distributed repeated-copy evidence to predict `ISRJ.num_replicas` from I/Q on
held-out identities.

This is a parameter-identifiability screen. It neither recognizes ISRJ nor
changes any six-label logit, calibration, threshold, fusion or test decision.

## Why this is not the stopped copy-consistency mechanism

| Stopped route | H0-ISRJ |
| --- | --- |
| Select one envelope-ACF peak from 40--1200 samples | Scan fixed lags 80--480 samples without sample-specific peak selection |
| Reduce each record to 20 global/window extrema | Retain a `2 x 26 x 12` complex correlation map (real and imaginary coherence) |
| Concatenate with 310D periodicity features and train an HGB classifier | Use only the fixed map in a compact replica-count regressor |
| Tests ISRJ classification increment on viewed development-v3 | Tests physical replica-count recoverability on a fresh train-only source |

## New source and data roles

- Generator base seed: `700000000`; 5,040 records, 80 per each nonempty six-label
  combination; no test split is generated.
- ISRJ width: uniform `[10,30] us`; replica count: integer `[3,8]`.
- Fresh source location: `C:\LJQ_code\干扰数据\复合干扰数据0802_isrj_delay_block_h0`.
- Evaluation uses only ISRJ-present records. Receiver identities are split by
  `receiver_noise_seed mod 10`: fit `{0..5}`, early `{6,7}`, held-out `{8,9}`.
  The source audit must prove 1,536/512/512 eligible identities, zero role
  intersection, finite active replica targets, and zero overlap with all
  historical train-named sources listed by the audit.
- Labels, relative powers, widths and replica counts are never model inputs.
  `num_replicas` is an oracle target solely for this H0 diagnostic.

## Fixed representation and matched probes

For RMS-normalized complex I/Q `z`, fixed lags are `80,96,...,480` samples and
each valid overlap is partitioned into 12 contiguous blocks. For every
lag/block pair, compute its normalized complex cross-correlation. The two map
channels are its real and imaginary parts. The map contains no learned
parameters and no candidate-lag argmax.

Both probes use exactly the same `2x26x12` input map and a 2D CNN regressor:
`Conv2d(2,8,3) -> GroupNorm -> GELU -> Conv2d(8,16,3) -> GroupNorm -> GELU ->
adaptive mean/max pooling -> MLP(32,32,1)`. They have identical initialization,
optimizer, batches, early criterion, training budget and target normalization.

- `R-real`: train against real `ISRJ.num_replicas`.
- `R-perm`: train against one fixed, fit/early-role target permutation; held-out
  evaluation always uses the real target.

Loss is Smooth-L1; AdamW `lr=3e-4`, weight decay `1e-4`, batch `64`, 30 epochs,
five seeds `2039..2043`; select minimum early loss. This label-permutation
control tests whether the new fixed map carries replica-count information; it
does not claim a deployable recognition gain.

## Preregistered metrics and stop rule

Primary held-out metrics are replica-count R2, Spearman and MAE normalized by
the observed target range. H0 passes only if all hold:

1. `R-real` mean R2 >= `0.20`;
2. `R-real` mean Spearman >= `0.40`;
3. mean `R-real - R-perm` R2 >= `+0.20`, paired bootstrap lower CI > 0;
4. at least 4/5 seed R2 contrasts are positive;
5. parameter count and source/role/test-seal contracts pass.

Failure stops this delay-block map; no lag, block-count, network-width,
regularization or epoch search is allowed. A pass permits only a *separate*
fresh-data H1 experiment with an exactly-zero-initialized bounded local branch
and a physical-information-destroyed, equal-capacity map control. It never
authorizes test-v3.

## Required artifacts

Generation receipt, data SHA-256, historical identity-overlap audit, role
indices/hash, source map contract, all checkpoint/history/prediction/metric
files, five run receipts, summary, and `locked_test_used=false`,
`test_v2_used=false`, `test_v3_used=false` fields are mandatory. A smoke is
only an I/O check and is kept under a separate output root.
