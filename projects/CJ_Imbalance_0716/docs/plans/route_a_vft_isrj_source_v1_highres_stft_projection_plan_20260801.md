# VFT--ISRJ source-v1 high-resolution STFT projection diagnostic

## Single falsifiable hypothesis

The 16x16 VFT STFT grid may alias long, dense ISRJ time-frequency structure
into a pattern that resembles a VFT ridge. A 32x32 grid should improve strict
weak-VFT versus VFT-absent/ISRJ-present separation even after it is projected
to the exact same classifier input dimension as the 16x16 baseline.

## Fixed source protocol

- Fresh, identity-audited source-v1 data and frozen per-combination fit,
  validation, and early roles `2039`--`2043`.
- Strict weak VFT-positive versus VFT-absent/ISRJ-present negative view,
  matched on the other four non-VFT/non-ISRJ labels.
- Fixed STFT: complex I/Q, Hann, `n_fft=128`, hop 32, full spectrum.
- Each arm uses fit-only standardisation, `RidgeClassifier(alpha=10)`, and
  class-balanced total sample masses. No calibration, threshold, fusion, or
  unified candidate is used.

## Equal-dimension arms

The native 16x16 descriptor has `16*16+12 = 268` dimensions. Every arm
presented to Ridge has exactly 268 dimensions and thus the same classifier
parameter count.

| Arm | Input before fixed projection | Input after projection |
|---|---|---:|
| Native16 | native 16x16 STFT plus 12 track statistics | 268 |
| High32Projected | native 32x32 STFT plus statistics | 268 |
| High32PhaseScrambledProjected | 32x32 descriptor after deterministic phase scrambling | 268 |

The 1036-to-268 Gaussian projection uses seed `2026082501`, is fit-free, and
is saved with SHA-256. The phase-scrambled control has the same resolution,
projection, dimension, classifier, and sample roles but destroys temporal
phase organisation.

## Decision gate

High32Projected must exceed Native16 in early AUROC by at least `+0.020` in
four of five seeds, exceed the phase-scrambled control by at least `+0.020`,
and have validation-to-early AUROC gap below `0.150`. Passing would establish
only a source-level, equal-classifier-capacity representation signal and would
then justify a separately fresh development-v13 candidate test. Failure stops
the resolution route; no grid size, projection seed, alpha, or architecture is
searched.

No historical locked test, test-v2, external OOD dataset, or sealed test-v3 is
permitted.
