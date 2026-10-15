# Route A H1t: fresh-identity asymmetric visibility transfer

## Hypothesis and scope

H0t established only a prerequisite: high-visibility frozen Focal scores are
informative for the three target labels. H1t tests one new, falsifiable learned
mechanism: **on identity-paired compound signals, does a fixed high-visibility
posterior improve weak low-visibility target ranking only when it remains
correctly paired to the same identity?**

This is a mechanism H1, not a new unified VTSE-VBF candidate and not a final
test. Knowledge distillation is not claimed as novel. The research question is
the label-internal visibility-tail condition and its pairing-specific evidence.

## New source, roles and boundary

- Generator: `experiments/generate_route_a_h1t_asymmetric_power_source.m`.
- Fresh base sample seed: `160000000`; sample-seed range
  `160000001--160008064`, checked against all recorded live development/test
  ranges, including sealed test-v3 `27000001--27025200`. The official receiver
  mapping is `mod(sample_seed + 2*2246822519, 2^32)`, giving the independent
  `receiver_noise_seed` range `358677743--358685806`.
- Train-only, 63 six-label combinations x 128 = 8,064 identities. Three views
  per identity use 10/30/50 dB for MFT/VFT/ISRJ and fixed 30 dB for FMB/SawFM/
  SinFM. No test split is generated.
- For each of five split seeds, stratify by the full six-label composition into
  fit 50% (4,032), validation 12.5% (1,008), and untouched early 37.5%
  (3,024) identities. Power views of an identity remain in exactly one role.
- The 50 dB teacher view exists only in fit; every primary endpoint uses the
  10 dB early view. No JNR/power metadata is an inference input.
- test-v2, every historical locked test and test-v3 remain unavailable. The
  test-v3 manifest may only be read to verify its sealed fields.

## Frozen architecture and training

The five H1t role seeds 2101--2105 are paired in fixed order with frozen
`E2_Focal_36`/F2 seeds 2039--2043, respectively. The trainable scope is exactly the final temporal stage plus
the six-label classifier (same scope as the Focal-36 fairness baseline);
parameter count is identical across arms. The 50 dB teacher is a separate
frozen copy of the same seed checkpoint and receives no gradient.

All arms use AdamW (classification LR `5e-4`, final-stage LR `5e-5`, weight
decay `1e-4`), batch size 64, gradient clip 5, maximum 36 epochs and patience
8. Validation selection key is the mean target weak-vs-negative AUROC at 10
dB. The base loss is the exact Focal-36 elementwise BCE variant
`gamma_pos=0, gamma_neg=3` on all six observed labels.

| Arm | Additional term | Purpose |
| --- | --- | --- |
| B0 | none | Same initialization/budget supervised Focal baseline at 10 dB. |
| T1 | `0.25 * T^2 * KL(Bernoulli(sigmoid(z50/T)) || Bernoulli(sigmoid(z10/T)))`, `T=2` | Apply only to MFT/VFT/ISRJ entries that are observed positive **and** weak at 10 dB; `z50` is detached. |
| P1 | exactly T1, but `z50` cyclically permuted inside each full-label composition and target-positive stratum | Preserves capacity, teacher score marginals, labels and loss scale while destroying same-identity pairing. |

No teacher-confidence threshold, pseudo-label, all-view feature matching,
temperature search, calibration or deployment threshold is allowed. A stratum
with fewer than two target-positive examples receives no P1 permutation for
that target; the count is recorded.

## Endpoints and fixed gates

Primary endpoints are ranking-only 10 dB early-role target weak-vs-negative
AUROC and AUPR, reported by label and as a three-target mean. Supporting
endpoints: strong-positive AUROC, all-positive AUROC/AUPR, non-target ranking
diagnostics and H0t-style identity/source contracts. No F1/FPR claim is
made because H1t intentionally has no calibration/threshold role.

T1 is admitted only if all conditions hold:

1. T1-B0 mean target weak AUROC >= +0.030, positive in >=4/5 seeds, and its
   paired bootstrap 95% CI lower bound is >0;
2. T1-P1 mean target weak AUROC >= +0.020, positive in >=4/5 seeds, and its
   paired bootstrap 95% CI lower bound is >0;
3. no target weak AUROC mean falls by more than 0.010 versus B0;
4. mean target strong-positive AUROC does not fall by more than 0.010 versus
   B0, and all source/role/test-seal contracts pass.

Failure freezes H1t: no change to lambda, temperature, composition strata,
permutation, power levels, trainable scope, loss or threshold is allowed on
this source. A pass permits a separately designed fresh-source replication;
it does not reconnect the method to C3 or authorize test-v3.

## Smoke-to-formal data-role repair

The first `160000000` source was used for a two-epoch code smoke after the
implementation tests. It is therefore excluded from formal H1t evidence,
including its unused role partitions. The formal R1 source changes only the
base sample seed to `164000000` (sample range `164000001--164008064`; receiver
range is derived by the same official mapping); all levels, 8,064 identities,
role construction, models, optimizer, loss, controls and gates above are
unchanged. This is a pre-formal data-role isolation repair, not a response to
any smoke metric.
