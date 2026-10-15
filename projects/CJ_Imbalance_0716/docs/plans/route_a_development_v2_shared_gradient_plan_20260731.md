# Route A development-v2 shared-gradient experiment preregistration

## Research question

The frozen-F2 H5 experiment established that deleting the SawFM output row cannot
change the other five independently trained linear rows.  The next falsifiable
question is therefore: **when a shared representation is allowed to adapt, does
the SawFM classification loss create negative transfer for FMB, SinFM, MFT, VFT
and ISRJ, and can label-selective gradient isolation remove it?**

## Evidence boundary

- Evidence role: fresh `development_v2` only.
- Input file: `closedset_6class_train.mat` under the 0731-v2 data root.
- Historical consumed locked test: forbidden.
- Sealed `closedset_6class_test.mat`: declared for audit only and never opened.
- Five paired seeds: 2039, 2040, 2041, 2042, 2043.
- Every seed uses the same optimization/early-stop/calibration/threshold roles,
  the same 2400-sample label-LT manifest, initialization and minibatch order for
  both variants.

## Controlled variants

Both variants start from the same historical F2 checkpoint.  The complex encoder,
late fusion and feature projection remain frozen.  Only the final temporal
multi-scale block and six-row classifier are trainable.

- `C6_Joint`: common-five loss plus `0.2 * SawFM loss`.  This makes the common
  loss scale identical to M5 while giving SawFM the same per-label weight as one
  of the five common labels.
- `M5_GradientIsolated`: SawFM samples and co-occurrence context are retained,
  but the SawFM output loss is exactly zero.  Common-five labels continue to be
  trained on every sample.

This is a loss/gradient ablation, not a five-class data deletion experiment.

## Optimization and model selection

- Label-LT target counts in code order `[FMB,SawFM,SinFM,MFT,VFT,ISRJ]`:
  `[1600,400,1213,919,696,528]`, exactly 2400 optimization samples.
- Deterministic 80/20 fit/validation split inside the manifest.
- AdamW; classifier LR `5e-4`; final temporal block LR `5e-5`;
  weight decay `1e-4`; 12 epochs; patience 4; gradient clipping 5.
- Checkpoint key: common-five validation macro-AUPR, tie-broken by mean
  MFT/VFT/ISRJ weak-positive-versus-negative AUROC.
- Calibration uses the common-five calibration role only.
- Thresholds use the threshold-selection role with per-label negative FPR cap
  0.20.

## Mechanism diagnostic

Before adaptation, four fixed fit minibatches measure cosine similarity between
the SawFM loss gradient and each of MFT/VFT/ISRJ loss gradients on the trainable
shared temporal block.  Negative cosine is direct evidence of local gradient
conflict, but it is not by itself performance evidence.

## Promotion gate

`M5_GradientIsolated - C6_Joint` must satisfy all of:

1. common-five macro-AUPR >= -0.005;
2. common-five macro-F1 >= +0.010;
3. mean MFT/VFT/ISRJ weak recall >= +0.030;
4. each MFT/VFT/ISRJ AUPR >= -0.010;
5. M5 mean target-label negative FPR <= 0.20;
6. target weak-recall delta positive in at least 4/5 seeds;
7. SawFM-present common-five macro-F1 delta > 0;
8. SawFM-absent common-five macro-F1 delta >= -0.005.

Passing the gate validates label-selective shared-gradient isolation on
development-v2.  It does not authorize opening test-v2; an independent frozen
confirmation protocol must be registered separately.
