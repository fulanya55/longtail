# Frozen MFT pulse-compression peak-count confirmation on development-v9

## Candidate

`MFT_PulseCompression_NMSPeakCount_v1` is frozen by `frozen/20260801_route_a_mft_peak_count_v1/mft_peak_count_frozen_manifest.json`. It has no learned parameters and returns the number of the first 12 non-maximum-suppressed matched-filter peaks in samples 400--1000 that exceed the q99 matched-filter background outside that window. NMS radius is 8 samples.

## Data boundary

- Generate only the train split of fresh development-v9 with base sample seed 94,000,000.
- Expected receiver-noise identities are 292,677,743--292,702,942 and must be disjoint from all earlier development and test contracts.
- No development-v9 test split is generated or opened.
- No candidate value, threshold, reference, ROI, NMS setting, or score orientation may be changed after generation.
- test-v3 remains sealed and unused.

## Frozen evaluation

- Main view: weak MFT positives of order >=4 versus MFT-absent negatives of order >=4.
- Extreme-risk view: the same positives versus order-5 MFT-absent negatives.
- Specificity control: identical peak count from a wrong-slope LFM reference.
- Report overall AUROC/AUPR and five deterministic receiver-identity blocks (`receiver_noise_seed mod 5`).

## Confirmation gate

All checks are required:

- overall high-order AUROC >=0.60;
- at least 4/5 receiver blocks have high-order AUROC >0.50;
- overall aligned-minus-wrong-chirp high-order AUROC >=0.08;
- at least 4/5 receiver blocks favor the aligned reference;
- overall order-5-negative AUROC >=0.58;
- development-v3 to development-v9 high-order AUROC drop <=0.06.

Passing establishes generalizable MFT feature identifiability only. It does not establish safe integration with U2. Fusion must be separately designed on development-v3 and confirmed on another fresh identity split.

