# Route A: valid target-heterogeneity restart gate

## Status and purpose

This is a corrective research protocol, not a scheduled model run. The v25
joint TS-CLE screen failed its promotion gate and its cyclic-source comparison
has subsequently been shown permutation-equivalent. Therefore there is no
eligible MFT/VFT/ISRJ candidate to train or integrate at this point. The goal
of this document is to state the minimum evidence that a *new*, non-overlapping
architecture must produce before consuming another fresh development source.

## Evidence that remains valid

| Question | Defensible evidence | Consequence |
| --- | --- | --- |
| Does a generic duplicate local branch prove target-specific evidence? | No. The v25 cyclic branch assignment is a parameter permutation under matched initialization and matched BCE. | Do not use its MFT `+0.01877` direction as causal evidence. |
| Does more STFT resolution solve VFT--ISRJ? | No. 32x32 projected STFT did not improve over native 16x16. | Do not search STFT size. |
| Does a low-dimensional raw phase/copy summary solve VFT--ISRJ? | No. It is informative versus phase scrambling but adds too little beyond 16x16 STFT. | Do not append/reweight that summary. |
| Does the tested fixed phase-reset gap statistic expose ISRJ width state? | No. On paired fresh data, AUROC `0.46780`, delta versus scrambling `-0.06451`, and width Spearman `-0.10041`. | Stop this window-state statistic. |
| Does simple VFT weak-positive weighting solve the boundary? | No. All tested weighting arms reduce weak AUROC. | Do not restart with loss weighting. |

## Non-negotiable definition of a valid control

A future target-local candidate is admissible only if its control satisfies all
four properties:

1. **Matched initial function:** candidate and control scores are exactly equal
   at initialization, or their difference is analytically known and zero in
   expectation with a stored proof.
2. **Matched trainable capacity and optimization:** same parameters, update
   count, batches, seeds, optimizer, early criterion and label supervision.
3. **Different physical information or inductive bias:** the candidate must
   encode a declared target cue; the control must replace that cue with an
   information-destroying but dimension- and energy-matched operation. Merely
   renaming or cyclically routing identical learnable branches is invalid.
4. **No selection leakage:** transform type, cue, comparator and gates are
   fixed before generating its source. Evaluation roles cannot be reused to
   select a follow-up transform.

## Required staged evidence

### Stage H0: paired physical identifiability, no classifier fitting

For one declared physical cue, generate a new identity-paired train-only
source in which only that cue is intervened on. Freeze the statistic/encoder
before generation. The raw candidate statistic must (a) recover the intervened
state with preregistered AUROC/correlation, and (b) exceed its
information-destroyed matched control. Failure ends that cue; no threshold,
window, resolution or statistic sweep is allowed.

The cue families must be genuinely different, rather than three copies of a
generic convolution:

- MFT: periodic delay/comb evidence;
- VFT: continuous chirp-ridge orientation or curvature evidence;
- ISRJ: delayed-copy/self-similarity evidence.

These are candidate *physics families*, not authorization to revive the prior
MFT peak/delay guard, VFT Ridge-STFT concatenation/morphology residual, or
ISRJ fixed phase-reset statistic.

### Stage H1: source-level target discrimination with a valid control

Only an H0-passing cue receives a separate newly generated five-seed,
identity-isolated source. The target task is weak positive versus
composition-matched target-absent negative. The candidate must beat the
valid matched control on each declared target: mean weak AUROC at least
`+0.015`, paired 95% lower CI above zero, and at least four of five positive
seed effects, with no strong-positive AUROC loss below `-0.010`.

### Stage H2: six-label integration

Only after H1 passes for all proposed integrated targets may a new six-label
model be designed. It must be compared with its same-capacity nonphysical
control, frozen global safety path, label-isolated calibration and
FPR-constrained threshold roles. No historical locked test, test-v2 or
test-v3 may be read.

### Stage H3: robustness and one-time test

Promoted H2 code, calibration, threshold, fusion and manifests are frozen
before identity, power, parameter and architecture robustness tests. Sealed
test-v3 can be authorized exactly once only after those gates pass.

## Immediate decision

No new model training is authorized by the existing records. The next valid
technical work is an H0 design rooted in a physical cue that is demonstrably
outside the stopped mechanisms, followed by a literature novelty audit and a
new generator contract. This protects the eventual innovation claim from
branch-index, threshold, and sequential-source selection artefacts.
