# Route A GradientContrastFixed1p5 one-time test-v2 preregistration

## Authorization basis

`GradientContrastFixed1p5` was frozen from development-v2 and then replicated
without alpha selection on identity-isolated development-v3.  On development-v3
it passed all ten preregistered gates versus `C6_Joint`:

- macro-AUPR delta `+0.0062399565`;
- macro-F1 delta `+0.0133747938`;
- mean MFT/VFT/ISRJ weak-recall delta `+0.0328327061`;
- target weak-recall positive in `5/5` seeds;
- SawFM-present macro-F1 delta `+0.0200583707`;
- SawFM-absent macro-F1 delta `+0.0068224661`;
- all target-label AUPR deltas non-negative;
- finite and identity-safe, with locked-test and test-v2 unused.

This authorizes one frozen confirmation on the already generated test-v2.  It
does not authorize fitting, model selection, calibration or threshold search on
test-v2.

## Frozen inputs

- Seeds: 2039, 2040, 2041, 2042, 2043.
- Baseline: each seed's development-v3 `C6_Joint` checkpoint, common-five
  calibration temperature and common-five thresholds.
- Candidate: each seed's development-v3 `GradientContrastFixed1p5` checkpoint,
  common-five calibration temperature and common-five thresholds.
- Candidate definition: `theta_C6 + 1.5*(theta_M5-theta_C6)`; alpha is fixed.
- Inference input: RMS-normalized complex I/Q `[B,2,3000]`.
- Test-v2 expected SHA256:
  `902e636911e76d6915fe04976f59cc46a588b3d98926bb9a48e2200628617947`.

Every checkpoint, calibration file, threshold file, config, development summary,
test plan and inference runner must be hashed into a frozen bundle before access.

## One-time access protocol

1. Run preflight without reading test-v2 contents.
2. Validate every frozen artifact hash and checkpoint/output contract.
3. Refuse execution if an access marker already exists.
4. Write the exclusive access marker before hashing or reading test-v2.
5. Verify test-v2 SHA256 after the marker.
6. Perform inference only.  No optimizer, fit, selection, temperature fit or
   threshold selection is permitted.
7. Save per-seed logits, probabilities, predictions and metrics.
8. Aggregate paired candidate-minus-baseline results and apply the fixed gate.
9. Regardless of result, set `rerun_or_retuning_allowed=false`.

## Frozen independent-test gate

The candidate must satisfy all:

1. common-five macro-AUPR delta >= -0.005;
2. common-five macro-F1 delta >= +0.010;
3. mean MFT/VFT/ISRJ weak recall delta >= +0.030;
4. each MFT/VFT/ISRJ AUPR delta >= -0.010;
5. candidate target mean negative FPR <= 0.20;
6. positive target weak-recall delta in at least 4/5 seeds;
7. SawFM-present common-five macro-F1 delta > 0;
8. SawFM-absent common-five macro-F1 delta >= -0.005;
9. all frozen hashes match and all outputs are finite;
10. `fit_or_selection_on_test_v2=false`.

Passing establishes C-level independent validation.  Failing establishes a
frozen negative independent result; no test-driven repair is allowed.
