# Route A v21: visibility-balanced morphology-residual selector

## Hypothesis

v20 established that a frozen morphology residual can reduce Q4 VFT hard
negatives, but its coefficient selection constrained aggregate VFT recall
only. The resulting aggregate loss was safe while weak-VFT recall fell.

**H-v21:** selecting the same one-scalar frozen morphology residual under both
aggregate-VFT and weak-VFT recall constraints preserves weak recall while
retaining a meaningful hard-negative FPR reduction. This is a selection-rule
test, not a new backbone, expert, calibrator, or threshold.

## Fresh protocol

- New train-only v21 data: generator seed `600000000`, 5,040 unique receiver
  identities, ISRJ width `[10,30] us`, replicas `[3,4]`; no test role/file.
- Identity roles use `receiver_noise_seed mod 10`: fit `{0..5}`, selection
  `{6,7}`, evaluation `{8,9}`. Each role must contain at least 20 weak VFT
  positives and 20 VFT-absent/ISRJ-present hard negatives.
- Freeze Q4, the v19 morphology M-real model/normalizer, calibration and
  thresholds. No v16 width model, v17/v18/v20 data, threshold selection, or
  source model refit is used.

## Matched rules

Let `p_i` be frozen Q4 ISRJ probability and `p_m` frozen v19 morphology VFT
probability. Both rules modify only Q4 VFT logit with
`z' = z - c * p_i * (1-p_m)` and use the same grid
`c={0.0,0.2,...,2.0}` and original Q4 calibration/threshold.

- **Global-only control:** choose `c` on fit to maximize hard-FPR reduction
  under aggregate VFT-positive recall loss <= 0.010.
- **Visibility-balanced candidate:** same objective/grid, but require both
  aggregate VFT-positive recall loss <= 0.010 and weak-VFT recall loss <=
  0.010. Ties choose smallest c.

Selection/evaluation roles never change `c`.

## Gates

On the held-out evaluation role, candidate must satisfy:

1. Hard-negative FPR delta vs Q4 <= -0.030 with >=4/5 negative seeds.
2. Aggregate and weak VFT recall losses each >= -0.010 in mean and every seed.
3. Candidate weak-VFT recall exceeds global-only control by >= +0.010 in mean
   and at least 4/5 seeds; its FPR trade-off relative to global-only is no more
   than +0.020.
4. VFT AUPR, ISRJ AUPR, SawFM AUPR and SawFM weak recall do not decline more
   than 0.010; Q4 non-VFT logits/thresholds and test-v3 seal remain exact.

Failure stops this selector mechanism. Passing permits a new-source
confirmation of this frozen visibility-balanced residual, not test-v3.

