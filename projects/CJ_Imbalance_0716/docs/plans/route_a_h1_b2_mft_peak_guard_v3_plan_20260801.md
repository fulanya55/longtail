# Route A H1+B2 plus frozen MFT peak guard: development-v3 preregistration

Date: 2026-08-01

## Scientific question

Can the already frozen pulse-compression peak-count negative guard reduce dense-composition MFT false positives when attached to the stronger H1+B2 classifier, without changing the ISRJ periodicity gain, RawSaw SawFM output, calibration, thresholds, or any non-MFT logit?

## Fixed candidates

- `P0_H1`: frozen Focal representation plus refit linear BCE head.
- `P1_H1B2`: `P0_H1` plus the aligned ISRJ periodicity score.
- `P2_H1B2_MFTPeakGuard`: `P1_H1B2` plus the frozen MFT pulse-compression negative guard.
- `P3_H1B2_WrongChirpGuard`: wrong-chirp physical control.
- `P4_H1B2_ShuffledGuard`: sample-shuffled aligned-evidence control.

Only the MFT logit may change between P1 and P2/P3/P4. The guard is fixed as

`delta = 0.75 * sqrt(4*p*(1-p)) * min(tanh(expert_logit), 0)`.

No model, calibration, threshold, feature, or cap is fitted or selected in this experiment. The aligned, wrong-chirp, and shuffled expert evidence are reused byte-for-byte from the frozen development-v3 MFT guard study.

## Data and evidence boundary

- development-v3 SHA-256: `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`.
- Seeds: 2039--2043.
- Evaluation role: frozen early-evaluation role for each seed.
- test-v3: sealed; access is forbidden.

## Primary construction gates: P2 versus P1

1. Mean macro-AUPR change >= -0.002.
2. Mean target weak-recall change >= -0.010.
3. Mean target weak-vs-negative AUROC change >= -0.002.
4. MFT order>=4 negative FPR reduction >= 0.020.
5. MFT weak-vs-order>=4-negative AUROC gain >= 0.010.
6. Aligned guard hard-AUROC gain exceeds wrong-chirp by >= 0.005.
7. Aligned guard hard-AUROC gain exceeds shuffled aligned evidence by >= 0.005.
8. All non-MFT logits remain exactly equal.
9. All five seeds reduce MFT order>=4 FPR.

## Unified promotion audit: P2 versus P0

The original Goal gates are reported without post-hoc relaxation: macro-AUPR >= +0.005, macro-F1 >= +0.010, mean MFT/VFT/ISRJ weak recall >= +0.030, mean weak AUROC >= +0.015, SawFM exact preservation, each target AUPR decline <= 0.010, target mean FPR <= 0.20, strong-recall decline <= 0.010, at least 4/5 positive weak-recall seeds, no SawFM-present/absent macro-F1 regression, and non-MFT preservation.

Passing the construction gate permits a fresh identity-isolated confirmation. It does not authorize test-v3. Failing stops this integration without changing the frozen evidence or thresholds.
