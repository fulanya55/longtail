# Final shared-policy candidate: frozen ISRJ parameter-OOD validation

## External evidence

Use the previously generated 5,040-sample train-only external set with unique
receiver identities, ISRJ width `22--30 us`, and replica count `6--8`, outside
the construction ranges `5--20 us` and `1--5`.  The data predate candidate v2,
and no test split exists.

The complete candidate, calibration, thresholds, and shared MFT decision policy
remain frozen.  No external fitting or selection is allowed.

## Gates

1. ISRJ AUPR and weak-vs-negative AUROC improve over H1.
2. ISRJ weak recall improves by at least `0.03`.
3. Absolute ISRJ FPR is at most `0.25`.
4. Macro-AUPR and macro-F1 do not decrease.
5. MFT shared-policy gate retains zero negative-to-positive flips and reduces
   high-order FPR by at least `0.01` on average.
6. VFT aligned expert remains better than base-only and shuffled controls.
7. SawFM probability and decisions remain exact.

This is controlled-generator parameter OOD evidence, not a claim about a new
noise type, new generator, semi-physical hardware, or real measured data.

