# Route A MFT pulse-compression morphology audit (development-v3)

## Purpose

The first matched-filter screen showed reference specificity but failed absolute separability and generalization gates. Its 308-dimensional descriptor retained absolute delay-bin locations and selected epochs from only 3--6 high-order weak positives per validation role. This audit asks whether a lower-dimensional, shift-invariant physical statistic contains reproducible MFT evidence before another learned classifier is attempted.

## Fixed audit

- Data: development-v3 early roles for seeds 2039--2043.
- Positive view: weak MFT samples with total order >=4.
- Negative view: MFT-absent samples with total order >=4; order-5 negatives are reported separately.
- References: aligned LFM, wrong-slope LFM, and self-ACF.
- No learned parameters, fitted thresholds, score orientation changes, or per-seed feature selection.
- All scores are defined so that a larger value is hypothesized to indicate multiple complete delayed LFM replicas.

## Frozen scalar scores

The physical delay region is 400--1000 samples. Background is the rest of the PRI. Non-maximum suppression uses an 8-sample exclusion radius.

- ROI maximum divided by background q99.
- Mean top-3 and top-6 ROI samples divided by background q99.
- Mean top-3 and top-6 NMS peaks divided by background q99.
- Count of the first 12 NMS peaks above background q99.
- ROI energy fraction.
- ROI q99 divided by background q99.

This is a mechanism diagnostic, not a promotable endpoint. A subsequent model may be designed only if one or more aligned-reference statistics achieve mean high-order AUROC >=0.60, are positive in at least 4/5 seeds, and exceed the same wrong-reference statistic by >=0.03. Any subsequent learned model requires fresh development-v9 confirmation before U2 fusion.

