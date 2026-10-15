# U2 bounded MFT peak-count negative guard (development-v3)

## Objective

Test whether the independently confirmed MFT pulse-compression peak count can reduce U2 high-order MFT false positives without materially harming weak-component recall or any non-MFT output.

## Frozen inputs

- Base endpoint: U2 `FullAlpha1p5+RawSaw+aligned periodicity` logits, calibration, and thresholds.
- MFT physical score: `MFT_PulseCompression_NMSPeakCount_v1` from the frozen manifest.
- Data roles: development-v3 fit for the one-dimensional expert calibrator; early for the construction guard. Calibration and threshold roles are not refit.
- test-v3 remains sealed and unused.

## Variants

1. `K0_U2Frozen`: exact U2 output.
2. `K1_AlignedPeakNegativeGuard`: aligned LFM peak-count expert.
3. `K2_WrongChirpNegativeGuard`: same-capacity wrong-slope LFM control with its own fit-only one-dimensional calibrator.
4. `K3_ShuffledAlignedGuard`: aligned expert evidence permuted within each role.

## Expert calibration

- Input: one scalar peak count (0--12).
- Training samples: weak MFT positives versus MFT-absent negatives of total order >=4 from the fit role.
- Model: standardized one-dimensional logistic regression, `C=0.1`, balanced classes.
- The calibrator is not a replacement classifier. It produces a signed evidence logit.

## Bounded integration

Only negative expert evidence can alter U2:

`mft_logit_new = mft_logit_u2 + 0.75 * uncertainty(u2) * min(tanh(expert_logit), 0)`

where `uncertainty(u2)=sqrt(4*p*(1-p))`. Thus confident U2 decisions and positive physical evidence are unchanged. All five non-MFT logits, all U2 Platt calibrators, and all U2 thresholds remain bitwise unchanged.

## Construction guard

Across five seeds, all mean checks must pass:

- macro AUPR decline <=0.002;
- target mean weak recall decline <=0.010;
- target mean weak AUROC decline <=0.002;
- MFT order>=4 negative FPR decreases by at least 0.020;
- MFT weak-positive versus order>=4-negative AUROC increases by at least 0.010;
- aligned guard exceeds wrong-chirp guard by at least 0.005 hard AUROC;
- aligned guard exceeds shuffled guard by at least 0.005 hard AUROC;
- all non-MFT logits are exactly preserved.

Passing authorizes freezing this integration and a fresh development-v10 confirmation. Failure stops the guard without changing U2.

