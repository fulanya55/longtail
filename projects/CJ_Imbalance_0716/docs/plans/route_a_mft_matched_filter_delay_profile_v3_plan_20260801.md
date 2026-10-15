# Route A MFT matched-filter delay-profile screen (development-v3)

## Research question

Can a physically aligned pulse-compression representation identify weak MFT evidence in high-order compound scenes where the current U2 endpoint produces excessive MFT false positives?

## Evidence boundary

- Selection data: development-v3 only.
- Roles: the existing seed-specific fit, validation, and early roles for seeds 2039--2043.
- development-v7 and development-v8 are diagnostic history only and must not be used for fitting, hyperparameter selection, or threshold selection here.
- test-v3 remains sealed with `access_authorized=false`, `analysis_allowed=false`, and `used=false`.
- If this screen passes, the implementation is frozen before generating a new identity-disjoint development-v9 train split.

## Fixed physical contract

- Sampling rate: 10 MHz.
- PRI: 300 us (3000 samples).
- Active pulse: 100 us (1000 samples).
- Reference waveform: `exp(j*pi*K*t^2)`, `K=5 MHz / 100 us`, with MATLAB-compatible `t=(1:1000)/fs`, followed by 2000 zeros.
- MFT observable delay support: 40--100 us (400--1000 samples), obtained from the configured 10--50 us common arrival plus 30--50 us target-relative delay.
- Correlation is circular because the generator uses `circshift` inside one PRI.

## Frozen variants

1. `MF_AlignedLFM`: circular complex matched filtering against the known transmitted LFM.
2. `MF_WrongChirp`: the same feature capacity and pipeline, but with the chirp slope sign reversed.
3. `Self_ACF`: the same profile summarizer applied to circular self-correlation.

No labels, MFT parameter truth, or co-label predictions enter the feature extractor.

## Feature vector

For each correlation magnitude profile:

- robust log profile normalized by its median absolute level;
- average and maximum pooled summaries over the full PRI (64 bins each);
- average and maximum pooled summaries over the physical 400--1000 sample delay window (64 bins each);
- sorted top-16 peak magnitudes and normalized locations in the physical window;
- fixed global and physical-window quantile, concentration, and peak-ratio statistics.

The aligned and wrong-reference variants have exactly the same feature dimension. The self-ACF control uses the same summarizer.

## Probe and data views

- Probe: one standardized linear classifier, weighted BCE, AdamW.
- Fit standardization is reused unchanged by validation and early roles.
- Training view: weak MFT positives and MFT-absent negatives whose remaining label compositions occur in both classes.
- Primary early view: weak MFT positives with total order >=4 versus MFT-absent negatives with total order >=4.
- Secondary views: all composition-matched samples and order-5 MFT-absent negatives.
- High-order validation AUROC selects the epoch; early is used only once per seed for the preregistered screen.

## Promotion gate

All checks must pass on five seeds:

- aligned early high-order AUROC >= 0.620;
- aligned-minus-wrong-chirp early high-order AUROC >= 0.030;
- aligned-minus-self-ACF early high-order AUROC >= 0.030;
- at least 4/5 seeds positive against each control;
- aligned validation-to-early high-order AUROC drop < 0.100;
- aligned early order-5-negative AUROC >= 0.580.

Passing authorizes only a frozen development-v9 confirmation. It does not authorize fusion into U2 or access to test-v3.

## Stop rules

- If aligned LFM fails the absolute or control-specificity gates, stop this feature mechanism.
- If aligned LFM passes on development-v3 but fails frozen development-v9 confirmation, do not tune on development-v9; return to a new mechanism and a later fresh identity split.
- Classification fusion is considered only after feature identifiability generalizes.
