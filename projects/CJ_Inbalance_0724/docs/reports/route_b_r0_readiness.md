# Route B R0 readiness

- Status: **verified**
- Project root: `E:\LJQ_code\CJ_Inbalance_0724`
- Data role: development/training MAT only
- Locked test opened: `False`
- Model training: `False`

## Data contract

- MAT: `E:\LJQ_code\干扰数据\复合干扰数据0716\closedset_6class_train.mat`
- File SHA-256: `2e9afdf1999452b6635038c52457f2b44b28f98aec422d87576e5ad76135d50e`
- Input `data_legacy`: `[6000, 25200]`; sample contract `[N, 2, 3000]` after reshape
- Labels: `['FMB', 'SawFM', 'SinFM', 'MFT', 'VFT', 'ISRJ']`
- Samples: `25200`; combinations: `63`; per-combination count range: `400—400`
- Unique `receiver_noise_seed`: `25200/25200`
- Parameter rule: `label=1 AND parameter_mask=1 AND target finite`
- Valid selected parameter entries: `153600`
- Mask-on-negative entries: `0`
- All valid targets finite: `True`

## Executed gates

| Gate | Status |
|---|---|
| Python compile | verified |
| Full pytest | verified |
| GPU/real-data smoke | verified |
| Data shape/label/mask/finite audit | verified |

## Boundary

This receipt proves R0 readiness only. It does not prove R1 baseline performance, R2 conditional-head improvement, multi-seed stability, long-tail generalization, or open-world performance. No test-v2 role has been created or opened.
