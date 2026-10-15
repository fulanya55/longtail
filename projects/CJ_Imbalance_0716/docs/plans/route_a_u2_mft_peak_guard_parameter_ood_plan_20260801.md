# Frozen U2 MFT peak guard parameter-OOD validation

## External generator

Use the independent stress generator `C:/LJQ_code/CJ_Inbalance_0724/matlab/generate_p3c_mft_count_ood_data.m` with base seed 460,000,000 and prefix `routea_mft_peak_guard_ood_20260801`. Generate train-only data; no test split is created.

Five regimes are generated at 25, 35, and 50 dB, 630 samples per view:

- `ID_control`: training-range MFT count, spacing, and amplitude.
- `low_count`: 1--2 false targets. This changes the semantic boundary from dense MFT and is a preregistered diagnostic only, not a promotion gate.
- `close_spacing`: 3--8 targets in 30--34 us.
- `ultra_close`: 3--8 targets in 30--31 us.
- `weak_target`: 3--8 targets with relative amplitudes 0.05--1.5.

The three power levels and all regimes reuse receiver identities by design and are paired views, never independent samples. Identity arrays and hashes must be audited before evaluation.

## Frozen endpoint

Reconstruct U2 for seeds 2039--2043 and apply the frozen aligned MFT peak-count guard. Reuse all U2 calibration and thresholds. No model, feature, calibration, threshold, ROI, NMS setting, or residual parameter is fitted on OOD data.

## Metrics

- six-label macro AUPR delta;
- MFT overall recall delta;
- MFT order>=4 negative FPR delta;
- MFT-positive versus order>=4-negative AUROC delta;
- exact preservation of all non-MFT logits;
- per regime, per power level, and per model seed results.

## External-evidence gate

- ID control: macro AUPR delta >=-0.002, MFT recall delta >=-0.030, hard FPR delta <=-0.020.
- Each supported OOD regime (`close_spacing`, `ultra_close`, `weak_target`): macro AUPR delta >=-0.005, MFT recall delta >=-0.050, hard FPR does not increase.
- At least two of three supported OOD regimes improve MFT positive-vs-hard-negative AUROC.
- Worst supported regime-level MFT recall delta >=-0.080.
- All non-MFT logits are exact.

`low_count` is reported as a falsification boundary. A failure there narrows the claim to dense MFT (>=3 replicas); it does not count as evidence for the supported dense-MFT claim.

