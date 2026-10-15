# Route A ISRJ aligned-periodicity evidence ledger

## Executive conclusion

The evidence now supports a **narrow, reproducible ISRJ innovation claim**:
on a frozen strong H1 classifier, a low-capacity aligned-periodicity score adds
ISRJ weak-evidence discrimination that is not explained by retraining,
threshold relaxation, RawSaw routing, or a time-order-destroying control. The
claim is bounded to the controlled generator and the three evaluated ISRJ
parameter regimes. It is not a claim of complete six-label external
robustness, VFT OOD repair, or final locked-test performance.

## Evidence ledger

| Evidence question | Result | Evidence tier | What it supports | What it does not support |
| --- | --- | --- | --- | --- |
| Is B2 better than the frozen H1 parent on development-v3? | ISRJ weak AUROC `+0.07034`, weak recall `+0.09240`, AUPR `+0.02088`; all 5/5 positive | development | A positive ISRJ component increment | External generalization |
| Is temporal ordering necessary? | B2 exceeds same-capacity block-shuffled B3; weak AUROC `+0.06439` on v3 | mechanism control | Gain depends on aligned long-period order rather than extra score capacity | A new learned backbone representation |
| Does the increment survive a first parameter shift? | OOD-v1 (22--30 us, 6--8 replicas): weak AUROC `+0.10261`, weak recall `+0.08619`, AUPR `+0.06706`; 5/5 positive | frozen external | First parameter-range replication | VFT/full-candidate OOD safety |
| Does it survive a farther high-replica shift? | H0g (25--30 us, 9 replicas): weak AUROC `+0.09965`, weak recall `+0.07125`, AUPR `+0.11927`; 5/5 positive | frozen external | Second parameter-range replication | Noise-model, real-data or test-v3 robustness |
| Does it survive a receiver-noise process shift? | H0h, new identities and proper-complex AR(1) noise with rho=0.80: weak AUROC `+0.08037`, weak recall `+0.10270`, AUPR `+0.02242`; 5/5 positive and ISRJ FPR `-0.00468` | frozen external | One preregistered correlated-noise replication | Arbitrary noise-model, VFT or final-test robustness |
| Is the effect just RawSaw/SawFM leakage? | RawSaw SawFM outputs are exactly unchanged in H0g | invariance check | Label-local ISRJ addition | MFT/VFT changes are safe |
| Is the effect just a decision threshold? | H0g ISRJ FPR change `+0.00242`, 95% CI `[-0.00395,+0.00863]`, while ranking/recall improve | decision check | Not a simple threshold relaxation | Worst subgroup FPR improvement |
| Does ISRJ ranking gain occur only under one MFT/VFT co-label? | H0g/H0h frozen-prediction audit: all 9 supported MFT/VFT strata have positive B2−B0 weak AUROC, each seed-bootstrap lower CI > 0 | post-completed descriptive audit | Observed ranking increment is not confined to one reported MFT/VFT condition | Uniform fixed-threshold safety or a new selection result |
| Is the wider VTSE counterfactual representation mechanism viable? | H0f true binding versus B0 weak AUROC `-0.00185`; versus cyclic binding `+0.00274`; binding accuracy `0.54750 < 0.60` | fresh diagnostic | Paired encoder differences are not target-specific enough | A deployable MFT/VFT/ISRJ feature module |

## Current claim wording

Permitted wording:

> A frozen low-capacity aligned-periodicity expert supplies label-local ISRJ
> evidence beyond a strong decoupled classifier. Its benefit is time-order
> dependent and is reproduced across receiver-isolated ISRJ parameter regimes,
> including an unseen high-replica condition, and one preregistered correlated
> receiver-noise process, while the SawFM routing output is unchanged.

Prohibited wording:

- “The complete unified six-label method is externally robust.”
- “The method solves VFT/ISRJ source confusion.”
- “The counterfactual VTSE/delta-prototype mechanism is validated.”
- “The method has passed a final independent test.”

## Completed H0h noise-model stress replication

The next experiment should not change B0/B2, the periodicity feature, fusion,
calibration or threshold. It should generate one **new**, receiver-isolated
train-only six-class source under a preregistered receiver-noise shift, then
perform exactly the H0g frozen B0/B2 comparison. The source must record the
noise process, total noise-power convention, correlation coefficient/bandwidth,
random seed range and source SHA. It must be different from a post-hoc H0g
perturbation: the new generator run requires fresh component and receiver
identities.

Suggested single fixed condition is circular complex AR(1) receiver noise
with normalized unit variance and coefficient magnitude `0.80`, replacing the
default white receiver noise at the generator composition step. This is a
receiver-noise correlation shift, not a new jamming mechanism. It is only
admissible if the reference generator can be extended in a project-local,
versioned wrapper without editing the reference source and if a deterministic
unit test proves variance normalization, proper-complex symmetry, and exact
reproducibility by receiver seed.

Pre-register before generating data:

1. B0/B2 and all saved artifacts remain byte-addressed frozen.
2. Five seeds 2039--2043, no external fitting or threshold changes.
3. The same primary gate as H0g: positive ISRJ AUPR/AUROC, weak-recall delta
   at least `+0.030`, 4/5 positive weak-recall/AUROC, B2 FPR at most `0.25`,
   non-negative macro-AUPR, exact SawFM preservation.
4. Add a stress-specific safety report for all labels and receiver cohorts;
   it reports rather than tunes worst subgroup FPR.
5. Failure freezes the noise source and leaves the current ISRJ claim limited
   to white-noise controlled regimes; it cannot trigger AR coefficient,
   SNR/JNR, fusion or threshold search.

H0h completed under these conditions and passed. Its source, rho=0.80 noise
model and frozen comparison are now closed against tuning. The next admissible
work is a compatible long-tail/mainstream-baseline audit and a complete
six-label ablation ledger; neither H0h nor any earlier external development
result authorizes test-v3. One-time test-v3 authorization can only be
considered after a frozen final bundle and a new complete unified candidate.

## Linked artifacts

- H0g plan: `docs/plans/route_a_h0g_isrj_parameter_ood_v2_frozen_20260802.md`.
- H0g result: `docs/results/route_a_h0g_isrj_parameter_ood_v2_frozen_results_20260802.md`.
- H0f stop: `docs/results/route_a_h0f_counterfactual_delta_prototype_results_20260802.md`.
- H0g machine result: `outputs/tapc/route_a_h0g_isrj_parameter_ood_v2_frozen/route_a_h0g_isrj_parameter_ood_v2_frozen_summary.json`.
- H0g identity/parameter audit: `outputs/tapc/route_a_h0g_isrj_parameter_ood_v2_generation/identity_parameter_audit.json`.
- H0h plan: `docs/plans/route_a_h0h_isrj_ar1_noise_replication_20260802.md`.
- H0h result: `docs/results/route_a_h0h_ar1_noise_frozen_results_20260802.md`.
- H0h machine result: `outputs/tapc/route_a_h0h_ar1_noise_frozen/route_a_h0h_ar1_noise_frozen_summary.json`.
- Frozen co-label audit: `docs/results/route_a_isrj_frozen_colabel_audit_results_20260803.md`.
