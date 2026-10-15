# Route A H0v: natural-visibility-graph observability diagnostic

Status: preregistered; the smoke source is code-path validation only.  No
performance or model-selection conclusion may be drawn from it.

## Rationale and falsifiable hypothesis

The primary source is [Cao et al., Sensors 2019](https://doi.org/10.3390/s19102344),
which defines a visibility-graph mapping for wideband radar time series and
reports a theoretical regular-graph property for LFM.  The candidate observable
is deliberately distinct from the closed MFT delay correlation, LFM matched
filter/dechirp, timestamp coherence, beamspace, ROI, and learned fusion routes:
it only uses the ordinal line-of-sight geometry of a time series.

For a real sequence `x[0..n-1]`, an edge `(i,j)`, `i<j`, is present when

`x[k] < x[i] + (x[j]-x[i]) * (k-i)/(j-i)` for every `i<k<j`.

The one predeclared primary score is normalized degree-distribution entropy
`H = -sum_k p(k) log p(k) / log(n-1)`, computed on the natural visibility graph
of 128 evenly spaced real-I samples.  The direction is fixed before any data:
target-present weak views are expected to have larger structural irregularity
and therefore larger `H` than their target-absent matched views.

H0v asks a narrow question only: in an exact counterfactual pair where one weak
component is added, does this fixed statistic carry target evidence beyond an
otherwise identical fixed temporal-order permutation control?  It does not
train a classifier, select a threshold, tune a network, or claim a six-label
model improvement.

## New source contract

Use the official 0716 closed-set generator discovered through
`create_closedset_6class_config.m`.  The source has 1,800 pairs (600 each for
MFT, VFT, ISRJ), base seed `4100000000`, and only two views per pair:

| View | target label | target power offset | all nuisance components and receiver noise |
| --- | ---: | ---: | --- |
| absent | 0 | absent | identical within pair |
| weak | 1 | -12 dB relative to common 20 dB setting | identical within pair |

Five nuisance compositions per target are fixed exactly in the generator.  A
fresh source is required; no prior H0e/H0f/H1/H2/H3/H5o/development/test source
may be reused.  The source writes I/Q, labels, pair id, target index, pair side,
sample seed, receiver-noise seed and SHA-256 receipt.  It creates no test role.

Before the formal source, run a separately named 30-pair smoke.  The smoke is
only a generator/I-Q layout check and cannot be used to alter `n`, the score,
the direction, composition list, power, seed, gate, or model design.

## Fixed analysis and controls

1. Apply the natural visibility graph to `real(iq_complex)` at 128 evenly
   spaced indices.  Vertical scale/offset do not affect the edge predicate.
2. Compute the degree entropy `H` without fitted parameters.
3. Compute the same score after the fixed index permutation generated from
   `numpy.random.default_rng(20260803).permutation(128)`.  This preserves each
   sample's marginal values but destroys original time order.
4. For each target, evaluate weak vs absent AUROC, AUPR, paired `H_weak >
   H_absent` fraction, and a pair-resampling 95% bootstrap CI.  No orientation,
   feature, threshold, or hyperparameter is selected after observing data.

## One-shot H0v gate

For **each** of MFT, VFT and ISRJ, all of the following are required:

- true-order weak-vs-absent AUROC >= 0.58;
- true-order AUROC minus permuted-order AUROC >= +0.040, with bootstrap 95% CI
  lower bound > 0;
- true-order paired `H_weak > H_absent` fraction >= 0.56;
- no parameters are trained and every test-use field remains false.

Failure for any target closes H0v globally: there is no retry with another
graph size, channel projection, graph feature, permutation, power, loss,
backbone, ROI, or threshold.  If and only if all targets pass, an independent
five-seed, equal-capacity C0/C3/C4 protocol may be designed; no test-v3 action
is authorized by H0v itself.

## Evidence boundary

This is a project-local, identity-isolated physical diagnostic.  It does not
reopen closed multi-PRI, delay-aperture, matched-filter, or learned VFT routes;
it produces no test data and must preserve the sealed test-v3 boundary.
