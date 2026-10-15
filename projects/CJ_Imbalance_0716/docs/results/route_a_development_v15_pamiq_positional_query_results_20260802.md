# Route A v15 PAM-IQ position-aware label query: five-seed development result

## Decision

**Rejected as a unified VTSE-VBF candidate.** Q1 (position-aware query) has exactly the same number of trainable/deployment parameters as Q0 (content-only query), and it has a reproducible ISRJ weak-ranking signal. However, its VFT weak ranking does not improve, target weak recall declines, and the fixed, role-isolated threshold rule yields an unsafe absolute target FPR. The preregistered gate therefore fails; test-v3 must remain untouched and position-encoding hyperparameters must not be tuned on this dataset.

## Protocol and evidence boundary

- Pre-registration: [`route_a_development_v15_pamiq_positional_query_plan_20260802.md`](../plans/route_a_development_v15_pamiq_positional_query_plan_20260802.md).
- New train-only source: `C:\LJQ_code\干扰数据\复合干扰数据0802_v15_pamiq_positional_query\closedset_6class_train.mat`.
- SHA-256: `8cd23d37ad96c4f10dc2911e2da03624572f5d729a0bea713f626df9bd197df8`; 25,200 unique `receiver_noise_seed` identities in `[648677743, 648702942]`.
- Five paired optimization seeds: 2039–2043. Each run used the same label-LT train manifest (2,400 samples, SawFM anchor 400), split, DB loss, 20-epoch budget, calibration role and threshold role.
- Q0: `pam_iq_dual_query`; Q1: `pam_iq_positional_query`. The only change is zero-parameter sinusoidal absolute time code added to label-query keys. Both have 257,819 deployment and 381,851 training parameters.
- Temperature calibration is fit only on the calibration role; shrinkage per-label F1 thresholds are fit only on the threshold role. Weak rank AUROC is threshold-free.
- Identity audit, model receipts and summary all record `locked_test_used=false`, `test_v2_used=false`, `test_v3_used=false`, and `test_v15_used=false`. This is development evidence only.

## Aggregate Q1 minus Q0 result

| Metric | Q0 mean | Q1 mean | Paired difference (Q1-Q0) | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| Macro-AUPR | 0.84490 | 0.85044 | +0.00553 | [-0.00055, +0.01279] |
| Macro-F1 | 0.74080 | 0.75399 | +0.01318 | [+0.00460, +0.02284] |
| Hamming loss | — | — | -0.01611 | [-0.03122, -0.00356] |
| VFT weak-vs-negative AUROC | 0.58548 | 0.57612 | -0.00936 | [-0.03980, +0.03596] |
| ISRJ weak-vs-negative AUROC | 0.53046 | 0.58049 | +0.05004 | [+0.00063, +0.08873] |
| Mean VFT/ISRJ weak AUROC | — | — | +0.02034 | [-0.01490, +0.05006] |
| Mean VFT/ISRJ weak recall | 0.46229 | 0.44643 | -0.01586 | [-0.11147, +0.07975] |
| Mean VFT/ISRJ negative FPR | 0.40142 | 0.32754 | -0.07388 | [-0.22849, +0.08109] |

The average target weak-AUROC increase exceeds +0.015 and is positive in 4/5 seeds, but its confidence interval crosses zero because the effect is not shared by VFT. The apparent pooled gain is therefore **an ISRJ-specific diagnostic result**, not evidence for a unified weak-component mechanism.

## Gate audit

| Preregistered condition | Result |
|---|---|
| Same parameter count for all five pairs | Pass |
| Mean target weak-AUROC gain >= 0.015 | Pass (+0.02034) |
| Weak-AUROC CI lower bound > 0 | **Fail** (-0.01490) |
| At least 4/5 positive weak-AUROC seeds | Pass (4/5) |
| Macro-AUPR non-inferior within -0.005 | Pass |
| VFT and ISRJ AUPR each non-inferior within -0.010 | Pass |
| Target FPR change <= +0.005 | Pass (-0.07388) |
| Q1 absolute mean target FPR <= 0.20 | **Fail** (0.32754) |

## Interpretation and stop rule

1. A learnable backbone plus absolute time keys can make ISRJ local evidence more rankable; the matched-parameter comparison rules out a raw capacity explanation for that narrow result.
2. VFT does not share the effect, so absolute position is not the missing common source of VFT/ISRJ discriminability. This agrees with the previous frozen-query failure and with the earlier physical intervention result: VFT/ISRJ confusion is not repaired by a generic temporal-location statistic.
3. Q1 reduces FPR relative to Q0 but still has an unacceptably high absolute FPR under the frozen, role-isolated F1 threshold contract. It is invalid to retrofit a more permissive/stricter threshold from these early-role results and call it a model gain.
4. Stop this mechanism. Do not sweep sinusoid frequencies, attention floors, query widths, temperatures, epochs or learning rates on v15. Do not connect Q1 to RawSaw or VTSE-VBF, and do not access test-v3.

## Artifacts

- Formal summary and paired bootstrap: `outputs/tapc/route_a_development_v15_pamiq_positional_query/route_a_development_v15_pamiq_positional_query_summary.json`.
- Formal run receipt: `outputs/tapc/route_a_development_v15_pamiq_positional_query/run_receipt.json`.
- Train-only identity audit: `outputs/tapc/route_a_development_v15_pamiq_positional_query_generation/identity_audit.json`.
- Generator receipt: `outputs/tapc/route_a_development_v15_pamiq_positional_query_generation/formal_train_generation_receipt.json`.
- Smoke-only chain: `outputs/tapc/route_a_development_v15_pamiq_positional_query_smoke/`.
