# Route A v16: ISRJ sample-width learnability screen (preregistered)

## Hypothesis

The paired physical intervention established a causal **effect** of ISRJ sample width on VFT false positives, while replica count did not support the generic occupancy explanation. The failed no-fit phase-reset statistic does not establish that width is absent from I/Q; it only rejects that low-dimensional handcrafted statistic.

This screen tests one narrower, falsifiable prerequisite: **a compact I/Q-only multiscale probe can predict ISRJ sample width on VFT-absent/ISRJ-present hard negatives better than a same-capacity permutation-label control, on a held-out identity role.** It is not a recognition candidate and cannot modify any classifier, calibration, threshold, fusion rule, or test-v3 authorization.

## New data and roles

- Train-only fresh generator seed: `520000000`; 5,040 records (63 combinations × 80), receiver-noise range `[718677743,718682782]`.
- ISRJ width is explicitly uniform in `[10,30] us`; replicas are in `[3,8]`.
- The probe retains only `VFT=0 AND ISRJ=1` records. A fixed combination-stratified identity split reserves 48/16/16 records per ISRJ-positive combination for optimization/early-stop/held-out evaluation. The held-out role is never used to select epoch or hyperparameters.
- Historical locked test, test-v2, test-v3, v15 data, and any v16 test role are prohibited.

## Matched models and training

Two identical compact multiscale I/Q regressors receive only complex-RMS normalized `[I,Q]` samples and output one standardized scalar.

- W-real: real ISRJ `sample_width_s` target.
- W-perm: the same fit target values permuted once within the fit role using a documented fixed RNG seed.

Architecture: Conv1d `2→16→32→48`, strides `2,2,2`, kernel sizes `9,7,5`, GELU + batch normalization; concatenate temporal mean, standard deviation and top-4 mean; MLP `144→64→1`. Same initialization per model seed, AdamW (`3e-4`, `1e-4` weight decay), Smooth-L1, batch 64, 30 epochs, early-role minimum validation loss checkpoint. Five optimization seeds are used with the same source identity split.

## Pre-registered outcomes and stop rule

Held-out metrics: width R2, normalized MAE (range-normalized), and Spearman correlation. The formal screen passes only if W-real has mean R2 >= 0.20, mean Spearman >= 0.40, W-real minus W-perm R2 mean >= 0.20 with paired bootstrap CI lower bound > 0, and at least 4/5 R2 differences are positive.

If it fails, stop width-conditioned correction entirely: no classifier residual, no threshold rule, no width-bin sweep, and no test-v3 access. If it passes, the next step is a separate new-data proof that a *predicted*, not oracle, width score improves VFT hard-negative control under a frozen FPR constraint.
