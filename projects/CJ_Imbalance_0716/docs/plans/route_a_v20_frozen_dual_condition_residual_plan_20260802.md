# Route A v20: frozen width-and-morphology dual-condition Q4 residual

## Single hypothesis

v16/v17 establish predicted ISRJ width as a Q4 risk axis; v18 rejects width
alone; v19 independently establishes nonlinear VFT morphology evidence against
the ISRJ confounder. **H-v20** is that their bounded conjunction, rather than
either alone, safely suppresses Q4 VFT hard-negative false positives.

## Strict cross-source protocol

- Generate fresh train-role-only v20 data (base seed `580000000`, 5,040 unique
  receiver identities, ISRJ width `[10,30] us`, replicas `[3,4]`). No test
  file is generated.
- Use identity roles `fit={0..5}`, `selection={6,7}`, `evaluation={8,9}` by
  `receiver_noise_seed mod 10`; each has balanced VFT-vs-ISRJ hard cases.
- Freeze three sources: matching v16 W-real width checkpoint/normalizer,
  matching v19 M-real morphology checkpoint/normalizer, and matching Q4
  checkpoint/calibration/threshold manifest. No source model is refit.
- v17 and v18 data/predictions are never read. Locked test, test-v2 and sealed
  test-v3 are forbidden.

## Candidate and controls

For Q4 VFT logit `z`, frozen Q4 ISRJ probability `p_i`, predicted width
evidence `w=clip((w_hat-20 us)/10 us,0,1)`, and frozen morphology VFT
probability `p_m`, form bounded evidence:

- width control: `e_w = p_i * w`
- morphology control: `e_m = p_i * (1-p_m)`
- dual candidate: `e_d = p_i * w * (1-p_m)`

Each rule changes only VFT: `z' = z - c e`. `c` is selected separately for
each seed/rule only on the v20 fit identities from `{0.0,0.2,...,2.0}` to
maximize hard-negative FPR reduction subject to VFT-positive recall loss <=
0.010; ties select the smallest c. Original Q4 calibration and thresholds are
applied unchanged. No threshold is selected.

## Prespecified evaluation gates

On held-out evaluation identities, all must hold:

1. Dual minus Q4 VFT hard-negative FPR <= `-0.030`, with >=4/5 negative seeds.
2. Dual minus width-only and dual minus morphology-only hard-negative FPR are
   each <= `-0.010`, with >=4/5 negative seeds.
3. Dual VFT positive and weak-positive recall losses are each >= `-0.010` in
   mean and every seed; VFT AUPR drop is no more than 0.010.
4. ISRJ and SawFM AUPR/weak recall do not decline by more than 0.010.
5. Every run preserves Q4 non-VFT logits and exact threshold vectors; all
   hashes, roles and test-v3 seal validate.

Failure stops dual residual integration. Passing only permits a separately
preregistered fresh confirmation source, not test-v3.

