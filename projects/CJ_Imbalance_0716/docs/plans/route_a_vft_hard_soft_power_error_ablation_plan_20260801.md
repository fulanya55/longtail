# Current VFT Ridge: hard mask, power-error mask, and soft visibility ablation

## Mechanism question

The current VFT Ridge-STFT expert uses strict weak-positive supervision.  This
experiment separates two claims that must not be conflated: whether that hard
mask is robust to realistic power measurement error, and whether a continuous
visibility weight can preserve weak evidence without discarding all strong
positives.

## Fixed data and models

- development-v3 five identity-isolated roles, seeds `2039`--`2043`;
- fixed 16x16 STFT VFT descriptor, fit-only z-standardisation, and
  `RidgeClassifier(alpha=10)`;
- strict composition-matched negative construction; the evaluation view is
  always the *nominal* 10 dB weak-VFT-positive versus matched-negative view;
- no calibration, threshold, fusion, or OOD data is used.

## Three arms

| Arm | Fit positives | Positive sample mass |
|---|---|---|
| HardNominal | nominal relative power <= -10 dB | equal within hard positives |
| HardPowerError | same rule after independent active-component `N(0,1 dB)` power measurement error | equal within perturbed hard positives |
| SoftVisibility | all composition-matched VFT positives | `sigmoid((-10 - relative_power_dB)/2.5)` |

For every arm, the positive and negative total training masses are separately
normalised to 0.5.  The `2.5 dB` sigmoid temperature is fixed from the already
documented continuous-visibility protocol; it is not searched here.  The
Gaussian measurement error has one fixed seed per receiver-role replication
and is applied only to fit mask construction.

## Interpretation

This is a current-expert mechanism ablation.  The hard-power-error result is a
robustness sensitivity, not a new candidate.  SoftVisibility must improve
early nominal weak AUROC/AUPR over HardNominal to challenge the hard-mask
mechanism; otherwise hard weak supervision remains the justified current
choice.  Any result leaves the frozen unified candidate and test-v3 unchanged.
