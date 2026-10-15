# Route A H0b: ISRJ width-normalized delay-map oracle screen

## Single falsifiable hypothesis

H0 showed that an absolute `80..480`-sample delay grid has only weak replica
information because ISRJ copy delay is a sample-specific multiple of width.
**H0b** tests exactly one different mechanism: when the delay axis is expressed
in *integer multiples of each record's true ISRJ sample width*, a distributed
complex correlation map can recover replica count on held-out identities.

This is an oracle-parameter upper bound only. Width is used exclusively to
construct the diagnostic representation, is not supplied as an input channel,
and is not a candidate classifier feature. A passing H0b would permit a new
fresh transport test that substitutes the already frozen v16 width predictor;
it would not authorize a residual, threshold or test-v3.

## New data and separation

- Generate a new balanced train-only source with base seed `710000000`, 5,040
  identities, 80 samples per nonempty combination, ISRJ width `[10,30] us` and
  replicas `[3,8]`; no test split.
- Use only ISRJ-present identities, with receiver modulo-10 fit/early/evaluation
  roles `60/20/20`. Audit zero overlap with every historical source including
  H0. The H0 evaluation source is never read by H0b code.

## Fixed oracle map and matched probes

For each sample, let `w = round(sample_width_s * fs_Hz)`. Fixed delay rows are
`w, 2w, ..., 8w`; each valid overlap uses 12 blocks. Rows that exceed the
3,000-sample record are represented by a fixed zero/validity convention. Each
cell stores real/imaginary normalized complex correlation, yielding a fixed
`2x8x12` input. No peak selection, width-bin choice, lag sweep or learned
alignment is allowed.

`R-real` and `R-perm` share the H0 CNN topology, initialization, optimizer
(AdamW `3e-4`, `1e-4`), Smooth-L1, batch 64, 30 epochs and five seeds. The only
difference is a fixed fit/early replica-label permutation in `R-perm`.

## Gate and action

The same held-out gates as H0 apply: mean real R2 >= `0.20`, mean real
Spearman >= `0.40`, real-minus-permutation R2 >= `+0.20` with positive paired
bootstrap lower bound, 4/5 positive contrasts, matched count/roles/seal.

Failure stops width-normalized delay-map work with no post-hoc adjustment.
Pass permits only H0c: frozen v16 predicted-width substitution on a third,
new identity source. Historical locked test, test-v2 and test-v3 are forbidden
throughout.
