# Route A v16: ISRJ sample-width learnability screen

## Decision

**Pass as a prerequisite mechanism screen, not as a classification result.** A compact I/Q-only multiscale regressor predicts ISRJ sample width on VFT-absent/ISRJ-present held-out identities substantially better than an equal-capacity permutation-label control. This makes a future *predicted-width* VFT hard-negative safety experiment scientifically admissible. It does not authorize an oracle-width classifier, threshold change, fusion update, or test-v3 evaluation.

## Protocol and boundary

- Preregistration: [`route_a_development_v16_isrj_width_identifiability_plan_20260802.md`](../plans/route_a_development_v16_isrj_width_identifiability_plan_20260802.md).
- New train-only data: `C:\LJQ_code\干扰数据\复合干扰数据0802_v16_isrj_width_probe\closedset_6class_train.mat`.
- SHA-256: `ee48e481ecf81479aca6e20a9b2c14e32bdcd54c52e644d935ab1ae4ad66162c`.
- 5,040 unique receiver-noise identities, `[718677743,718682782]`; 2,560 ISRJ positives. The screen uses only the 1,280 records satisfying `VFT=0 AND ISRJ=1`.
- The fixed combination-stratified split contains 768 fit, 256 early-stop and 256 held-out identities (48/16/16 per eligible combination), SHA-256 `a19d29bd1c8bb89d70125cc620c57ba0698f4174e798ee70018a602ac4407b76`.
- Width is `[10,30] us`; replicas are `[3,8]`. Five optimization seeds are 2039–2043. Both probes have 21,377 parameters, width 16, Smooth-L1, AdamW `3e-4`, weight decay `1e-4`, batch 64 and 30 epochs.
- W-real regresses true ISRJ width. W-perm receives identical inputs, architecture, initial seed, target normalization and budget, but its target mapping is fixed-permuted with seed `2026080202`.
- All receipts record `locked_test_used=false`, `test_v2_used=false`, `test_v3_used=false`, and `test_v16_used=false`.

## Held-out result

| Quantity | W-real | W-perm / paired contrast |
|---|---:|---:|
| Mean R² | 0.562 | W-real − W-perm: +0.572, CI `[+0.562,+0.583]` |
| Mean Spearman | 0.756 | W-real − W-perm: +0.779, CI `[+0.668,+0.876]` |
| Mean normalized MAE (observed 20 us range) | 0.130 | contrast: -0.103, CI `[-0.105,-0.100]` |
| Positive R² contrast seeds | — | 5/5 |

Per-seed W-real R² is `0.546, 0.569, 0.547, 0.550, 0.598`; W-perm R² is `-0.022, -0.016, -0.016, -0.009, +0.011`. The real model's MAE is about `2.61 us`, versus `4.66 us` for the permutation control.

## Gate audit

| Gate | Result |
|---|---|
| Equal parameter count in all five pairs | Pass |
| W-real mean R² >= 0.20 | Pass (0.562) |
| W-real mean Spearman >= 0.40 | Pass (0.756) |
| R² gain vs permutation >= 0.20 | Pass (+0.572) |
| R²-gain bootstrap CI lower bound > 0 | Pass (+0.562) |
| Positive R² gain in at least 4/5 seeds | Pass (5/5) |

## Interpretation and next gate

The previous no-fit phase-reset diagnostic rejected one handcrafted window-state statistic; it did **not** show that ISRJ width is absent from I/Q. v16 now provides direct evidence that a compact learned multiscale I/Q representation contains sample-width information under VFT-absent/ISRJ-present hard negatives.

This remains an oracle-positive diagnostic: selecting `ISRJ=1` and reading the width label are permitted only for probe training/evaluation. A deployable correction must use the W-real prediction alone. The next experiment must use a new identity-isolated controlled-width dataset to show that frozen W-real predictions generalize to narrow/wide regimes and align with the frozen Q4 VFT hard-negative risk; only then may a separate, fresh, bounded residual classifier experiment be designed.

## Artifacts

- Summary: `outputs/tapc/route_a_development_v16_isrj_width_probe/route_a_development_v16_isrj_width_probe_summary.json`.
- Formal receipt: `outputs/tapc/route_a_development_v16_isrj_width_probe/run_receipt.json`.
- Identity/parameter audit: `outputs/tapc/route_a_development_v16_isrj_width_probe_generation/identity_parameter_audit.json`.
- Generation receipt: `outputs/tapc/route_a_development_v16_isrj_width_probe_generation/formal_train_generation_receipt.json`.
- Per-seed checkpoints, histories and held-out predictions: `outputs/tapc/route_a_development_v16_isrj_width_probe/runs/`.
- Smoke only: `outputs/tapc/route_a_development_v16_isrj_width_probe_smoke_io_repaired/`.
