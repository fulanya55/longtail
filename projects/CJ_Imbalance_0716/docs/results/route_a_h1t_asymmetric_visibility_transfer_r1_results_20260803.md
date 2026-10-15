# Route A H1t asymmetric visibility-transfer R1: formal result

Date: 2026-08-03  
Decision: **failed; H1t is frozen and cannot be retuned on this source.**

## Question and evidence boundary

H1t asked whether a fixed 50 dB posterior improves 10 dB weak-component
ranking for MFT, VFT and ISRJ specifically because it remains paired to the
same `receiver_noise_seed`. It is not a new unified VTSE-VBF candidate and it
does not make a final-test claim.

The formal source is
`C:\LJQ_code\干扰数据\复合干扰数据0803_h1t_asymmetric_power_v1_r1`.
It contains 8,064 fresh identities and three train-only power views (10/30/50
dB). Five receiver-identity-grouped role splits (2101--2105) were used. The
first, differently seeded source used for a two-epoch smoke remains excluded
from formal evidence. All runs record `locked_test_used=false`,
`test_v2_used=false`, and `test_v3_used=false`; test-v3 remains sealed.

The model scope, optimizer, initialization and 54,150 trainable parameters
were fixed across arms. B0 is 10 dB supervised Focal-36. T1 adds the frozen,
positive-and-weak-target-only 50-to-10 dB Bernoulli KL term. P1 is identical
to T1 except that teacher scores are cyclically permuted within the same full
six-label composition and target-positive/weak stratum. For every target and
split, P1 reports exact teacher score-marginal preservation (1,792 permuted
examples across 28 composition strata).

## Formal five-seed results

All differences below are candidate minus comparator on the untouched 10 dB
early role. Intervals are paired bootstrap 95% intervals over the five source
splits.

| Comparison | Endpoint | Mean difference | 95% CI | Result |
| --- | --- | ---: | --- | --- |
| T1 - B0 | Mean target weak AUROC | +0.00076 | [-0.00432, +0.00449] | fails +0.030 and positive-CI gates |
| T1 - B0 | Positive mean-weak AUROC seeds | 4/5 | -- | isolated count gate only |
| T1 - B0 | Mean target weak AUPR | +0.01052 | [+0.00698, +0.01312] | supporting only; not admission endpoint |
| T1 - B0 | Mean target strong AUROC | -0.08791 | [-0.09765, -0.07653] | fails safety gate (-0.010) |
| T1 - P1 | Mean target weak AUROC | -0.00137 | [-0.00273, +0.00040] | fails +0.020, 4/5 and positive-CI gates |
| T1 - P1 | Positive mean-weak AUROC seeds | 1/5 | -- | fails |

Per-target T1-B0 weak-AUROC differences are MFT `+0.02030`
(`[+0.01514,+0.02469]`), VFT `-0.01178`
(`[-0.01738,-0.00618]`), and ISRJ `-0.00625`
(`[-0.01960,+0.00436]`). Thus the modest MFT increase neither reaches the
pre-registered global effect nor transfers across target labels.

Per-target T1-P1 weak-AUROC differences are MFT `-0.00137`
(`[-0.00260,-0.00038]`), VFT `-0.00147`
(`[-0.00373,+0.00200]`), and ISRJ `-0.00127`
(`[-0.00349,+0.00057]`). Correct receiver identity pairing is therefore not
better than the composition-stratified score-marginal control in this protocol.

## Interpretation and permitted conclusion

H0t's frozen high-versus-low posterior gap established that a high-power view
contains additional label information. H1t shows that, under the specified
weak-positive KL recipe and final-stage/classifier adaptation, this information
does **not** yield an identity-specific low-power ranking gain. The almost
identical (and slightly worse) T1 result relative to P1 is incompatible with
the claimed same-identity soft-evidence mechanism.

The large strong-positive ranking loss indicates that a KL term restricted by
label mask still changes shared final temporal features and the shared
classifier boundary. A positive AUPR shift alone cannot repair this failure:
the primary ranking endpoint, the identity control, the per-label safety gate,
and the strong-positive safety gate jointly fail.

H1t supplies a negative mechanism result only. It cannot be used to support a
six-label VTSE-VBF improvement, does not authorize a replication or a
test-v3 run, and does not modify the previously frozen C3/C4 conclusion.

## Reproducible artefacts

- Plan: `docs/plans/route_a_h1t_asymmetric_visibility_transfer_preregistration_20260803.md`
- R1 source receipt: `outputs/tapc/route_a_h1t_source_generation_r1/formal_h1t_source_generation_receipt.json`
- R1 source audit: `outputs/tapc/route_a_h1t_source_audit_r1/summary.json`
- Run summary: `outputs/tapc/route_a_h1t_asymmetric_visibility_transfer_r1/summary.json`
- Per-arm histories, checkpoints, predictions and receipts: `outputs/tapc/route_a_h1t_asymmetric_visibility_transfer_r1/`
