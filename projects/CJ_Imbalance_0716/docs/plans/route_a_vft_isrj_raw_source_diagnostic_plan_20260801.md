# Route A VFT-ISRJ raw-source diagnostic (construction-v3)

## Falsifiable hypothesis

The VFT failure on long-duration, dense ISRJ parameter OOD is caused by a
source-identification deficit in the existing local STFT score.  VFT should
retain smoother time-frequency ridge evolution, whereas ISRJ hard negatives
show stronger repeated-window phase coherence and less regular reset dynamics.
A fixed raw-I/Q descriptor of those quantities should separate weak VFT
positives from VFT-absent/ISRJ-present hard negatives more reliably than the
existing Ridge-STFT score.  If it does not, this physical-source route stops;
no fusion-head tuning is allowed.

## Data roles and boundary

- Fit/validation/early: the five identity-isolated development-v3 roles
  (`2039`--`2043`).  Fit trains once; validation is reported only; early is the
  primary held-out result.
- External ISRJ parameter-OOD v1: frozen diagnostic application only; it is
  never used for fitting, normalization, selection, calibration, or thresholds.
- `test-v3`: sealed and never opened.

The binary source view is deliberately harder and more specific than ordinary
VFT-vs-all: positives are weak VFT samples; negatives are VFT-absent and
ISRJ-present samples.  They are matched on the remaining four co-labels
(FMB/SawFM/SinFM/MFT), so source discrimination cannot be explained by those
co-occurrences.

## Fixed models

All three arms use fit-only standardisation, class-balanced sample weights, and
`RidgeClassifier(alpha=10)` with no validation selection.

| Arm | Feature | Dimension | Purpose |
|---|---|---:|---|
| `RidgeSTFT` | Existing 16x16 VFT STFT descriptor | 268 | Strong physical baseline under the same hard view |
| `RawSource` | Four phase-copy lags, segment-copy coherence, STFT ridge continuity, instantaneous frequency/amplitude reset statistics | 32 | Proposed source-evidence mechanism |
| `PhaseScrambledControl` | Same 32 operations after deterministic within-record phase permutation while preserving amplitude | 32 | Tests whether the gain depends on phase organisation rather than capacity |

The RawSource descriptor has no trainable encoder, no threshold, no calibrator,
and no parameters selected from validation/early/OOD.

Before raw-feature extraction, each seed must contain at least one exact
context-matched weak VFT positive and one VFT-absent/ISRJ-present negative in
each fit, validation, and early role.  This is an estimability condition, not a
performance gate.  If it fails, the experiment stops with a data-protocol
finding; it must not relax co-label matching after inspecting feature results.

## Advance gate

All conditions must hold before a frozen raw-source expert may be considered
for a new candidate and fresh development-v13:

1. early hard-view AUROC improvement over `RidgeSTFT` >= `0.010`, with at least
   4/5 positive seeds;
2. early hard-view AUROC improvement over `PhaseScrambledControl` >= `0.020`;
3. validation-to-early RawSource AUROC gap < `0.150`;
4. on external ISRJ parameter-OOD, RawSource does not reverse relative to
   `RidgeSTFT` (mean hard-view AUROC difference >= `0` and at least 4/5
   non-negative seeds).

Failure stops this route.  Passing only admits an independently frozen expert;
it does not constitute a unified-model gain or authorize test-v3.
