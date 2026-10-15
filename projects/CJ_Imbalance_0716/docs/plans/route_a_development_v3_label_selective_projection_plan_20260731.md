# Route A development-v3 label-selective conflict projection preregistration

## 1. Research question

Can SawFM gradient components that conflict with MFT and ISRJ be removed at the
last shared temporal block while retaining compatible SawFM information for VFT,
thereby passing the fixed common-five and weak-component gates without deleting
SawFM-containing samples?

## 2. Evidence boundary

- Primary evidence: a newly generated identity-isolated development-v3 train
  file only.
- Base generator seed: `24000000`.
- v3 train sample_seed interval: `[24000001, 24025200]`.
- development-v2 train interval: `[20260732, 20285931]`.
- sealed test-v2 interval: `[23260732, 23285931]`.
- The generator's receiver-noise seed is a bijective additive uint32 mapping
  from sample_seed, so disjoint sample_seed intervals imply disjoint receiver
  noise identities.
- Historical locked-test and test-v2 are forbidden in this experiment.
- Five paired training seeds: 2039, 2040, 2041, 2042, 2043.

Frozen generated file:

- size: `2549107141` bytes;
- SHA256: `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`;
- observed receiver_noise_seed interval: `[222677743, 222702942]`;
- unique receiver_noise_seed count: `25200`;
- v2-train receiver interval: `[218938474, 218963673]`;
- sealed test-v2 receiver interval from its frozen generator contract:
  `[221938474, 221963673]`.

Both intersections are empty.  test-v2 was not opened for this audit.

## 3. Data protocol

- Build stratified optimization, early-stop, calibration and threshold roles
  independently inside development-v3 for every seed.
- Build a 2400-sample label-LT manifest only inside optimization.
- Target counts in code order `[FMB,SawFM,SinFM,MFT,VFT,ISRJ]` are
  `[1600,400,1213,919,696,528]`.
- Split the manifest deterministically into fit 1920 and validation 480.
- C6, M5 and the projection candidate share exact role IDs, manifest,
  initialization, minibatch order and budget.

## 4. Model and trainable scope

- Initial model: historical `F2_DFMS_Full` checkpoint for the matching seed.
- Frozen: complex encoder, Cartesian/magnitude/phase late fusion, global
  statistics pooling and 128-dimensional feature projection.
- Trainable: final shared temporal multi-scale residual block and six-output
  classifier.
- All variants retain a six-output deployment shape; no data row is removed.

## 5. Fixed variants

1. `C6_Joint`: `L_common + 0.2 * L_SawFM`.
2. `M5_GradientIsolated`: `L_common`; SawFM loss is zero.
3. `GradientContrastFixed1p5`: after C6 and M5 training, use
   `theta_C6 + 1.5 * (theta_M5 - theta_C6)`.  Alpha is frozen from prior
   development-v2 and is not selected on v3.
4. `LSCP_TargetProjected` (primary candidate): retain SawFM loss, but modify
   its gradient only at the final shared temporal block.

## 6. Label-selective projection rule

For each minibatch, calculate shared-block gradients `g_saw`, `g_mft`, `g_vft`
and `g_isrj` from the corresponding label BCE losses.

Use the fixed target order MFT, ISRJ, VFT and two deterministic projection
passes.  For each target gradient `g_t`:

- if `dot(g_saw, g_t) < 0`, set
  `g_saw <- g_saw - dot(g_saw,g_t)/(||g_t||^2+1e-12) * g_t`;
- otherwise leave that component unchanged.

The final shared gradient is `g_common + 0.2 * g_saw_projected`.  Classifier
gradients use ordinary `g_common + 0.2 * g_saw`, because label rows are
independent and the research question concerns shared representation conflict.

Record per epoch and aggregate:

- pre/post cosine for SawFM versus MFT, VFT and ISRJ;
- projection application count and fraction;
- removed SawFM gradient norm fraction;
- common, SawFM and combined gradient norms;
- non-finite logits/gradient count and gradient clipping count.

Any non-finite training state invalidates the candidate.

## 7. Optimization and selection

- AdamW, classifier LR `5e-4`, shared block LR `5e-5`, weight decay `1e-4`.
- 12 epochs, patience 4, batch size 64, gradient clipping 5.
- Paired random seed and minibatch order across trainable variants.
- Checkpoint key: validation common-five macro-AUPR; tie-breaker is mean
  MFT/VFT/ISRJ weak-positive-versus-negative AUROC.
- Common-five temperature is fit on calibration only.
- Per-label thresholds are selected on threshold-selection only with negative
  FPR cap 0.20.
- Early-stop role is used once for final development-v3 reporting.

## 8. Fixed development promotion gate

Each candidate is compared separately with `C6_Joint` and must satisfy all:

1. common-five macro-AUPR delta >= -0.005;
2. common-five macro-F1 delta >= +0.010;
3. mean MFT/VFT/ISRJ weak recall delta >= +0.030;
4. each MFT/VFT/ISRJ AUPR delta >= -0.010;
5. candidate target mean negative FPR <= 0.20;
6. positive target weak-recall delta in at least 4/5 seeds;
7. SawFM-present common-five macro-F1 delta > 0;
8. SawFM-absent common-five macro-F1 delta >= -0.005;
9. no non-finite states, role overlap or identity leakage;
10. `locked_test_used=false` and `test_v2_used=false`.

Primary decision is based on `LSCP_TargetProjected`.  The frozen
`GradientContrastFixed1p5` is reported as a prior-method replication and may
advance only if it independently passes all ten gates; no v3 hyperparameter or
method selection is allowed from early-stop results.

## 9. Next decision

- If a frozen candidate passes every gate, freeze its checkpoints, calibration,
  thresholds and inference code, preregister a one-time test-v2 confirmation,
  and only then access test-v2.
- If no candidate passes but LSCP shows stable mechanism evidence, retain the
  failure and move to a new falsifiable local-evidence mechanism on fresh data.
- Never lower the gates or tune from test-v2.
