# Route A v24: matched-capacity multi-resolution weak-evidence probe

## Single hypothesis

v22 and v23 reject loss-only remedies on the 16x16 morphology representation. **H-v24:** a fixed high-resolution local time-frequency view retains weak VFT evidence that the existing 16x16 feature discards. Any gain must exceed an equally sized input that is a deterministic projection of the old feature and hence adds parameters but no information.

## Fresh source and identity roles

Generate a new train-only six-label source at base seed `660000000` with 5,040 unique receiver identities, 80 examples per nonempty combination, ISRJ width `[10,30] us`, replicas `[3,4]`, and no test split. Retain ISRJ-present examples and split by receiver modulo 10: fit `{0..5}`, early-stop `{6,7}`, evaluation `{8,9}`. Each role must contain at least 20 weak VFT positives, strong VFT positives, and VFT-absent/ISRJ-present negatives. Existing v19-v23 data/predictions and all frozen tests are forbidden.

## Matched representations and network

Both variants receive a 1,292-dimensional input and the exact same `1292 -> Linear(13) -> GELU -> Dropout(0.10) -> Linear(1)` head (16,823 parameters), AdamW (`3e-4`, decay `1e-4`), ordinary BCE, batch 64, 30 epochs, identical initialization, and the unweighted three-stratum early-stop loss.

- `M_ProjControl`: existing 268-dimensional 16x16-STFT-plus-track feature concatenated with a fixed 1,024-dimensional Gaussian projection of that same feature. The matrix is generated once using PRNG seed `240024`; it contains no new signal information.
- `M_MultiRes`: the same 268-dimensional base feature concatenated with a 32x32 log-normalized STFT image (1,024 values), computed from the same complex I/Q but not recoverable from the pooled 16x16 representation.

Thus the input dimensionality, parameter count, optimizer, loss, seed, and checkpoint selection are equal; the sole intervention is access to local multi-resolution evidence. No threshold, calibration, fusion, width statistic, or Q4 output is present.

## Gate

On held-out identities, store features, projection hash, normalizers, histories, checkpoints and predictions. Promotion requires all of: weak VFT versus hard-negative AUROC gain `>=+0.030`, lower paired-bootstrap CI `>0`, and >=4/5 positive seeds; weak AUPR gain `>=+0.020`; no per-seed strong or overall AUROC loss below `-0.010`; exact matched parameter/input contracts; source audit; and sealed test-v3. Passing only permits a new-source fusion design; failure stops multi-resolution morphology augmentation.
