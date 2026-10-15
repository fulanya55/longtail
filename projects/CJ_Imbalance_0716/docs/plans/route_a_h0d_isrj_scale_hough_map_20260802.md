# Route A H0d: inference-ready ISRJ scale-Hough copy map

## Single falsifiable hypothesis

H0b proves repeated-copy structure only after oracle width alignment, while H0c
proves a single predicted width is not accurate enough for phase-sensitive
alignment. **H0d** removes both dependencies: a fixed bank of all physically
feasible width hypotheses accumulates complex delay-block correlations along
the lines `m*w`, then a compact CNN learns replica count from the full bank.

This is an inference-ready physical representation: it uses I/Q only, contains
no oracle parameter, no predicted-width model, no selected per-record lag and
no classifier score. It is not an H1 recognition result.

## Fixed transform and data

- Fourth fresh train-only source, seed `730000000`, 5,040 identities, ISRJ
  width `[10,30] us`, replicas `[3,8]`, no test split; ISRJ-positive identity
  roles are 60/20/20 by receiver modulo 10 and must be disjoint from H0--H0c.
- Width hypotheses are exactly `100,120,...,300` samples (11 values, the
  physical 10--30 us range at 10 MHz with fixed 2-us spacing). For each
  hypothesis `w`, multiplier `m=1..8`, and one of 12 overlap blocks, store
  real and imaginary normalized complex correlation at lag `m*w`.
- The resulting `2x88x12` map is precomputed from I/Q once; the representation
  has zero learned parameters. A 2,497-parameter CNN head is trained against
  real or fixed-permuted replica targets with the H0b optimizer and budget.

## Gate

Held-out real-map R2 >= `0.20`, Spearman >= `0.40`, real-minus-permutation
R2 >= `+0.20` with positive paired 95% lower CI and 4/5 positive seeds,
matched capacity/roles/seal. Any failure stops this map and prohibits width
grid, spacing, multiplier, pooling or CNN sweeps. A pass permits a separately
generated H1 ISRJ weak-vs-composition-matched-negative expert test with an
information-destroyed map control; it does not authorize classifier fusion or
test-v3.
