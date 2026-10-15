# P3o ISRJ--MFT mechanism-route closure audit

## Purpose and boundary

This audit consolidates completed P3o development-only evidence.  It neither reads nor reuses any consumed independent or locked test, selects no checkpoint or threshold, and changes no model, data role, or protocol.  All referenced experiments report `test_generated=false`, `test_opened=false`, and `locked_test_used=false`.

The question is deliberately narrower than a general six-class improvement claim: can the excess ISRJ false positives induced by MFT be removed in a deployable way while retaining MFT recognition evidence and predicted-gate parameter safety?

## What the evidence establishes

| evidence tier | experiment | admissible finding | limiting finding |
| --- | --- | --- | --- |
| causal oracle | P3o-S | Exact power-calibrated MFT subtraction lowers `MFT+ / ISRJ-` ISRJ FPR by 0.0867--0.1100 in every one of 3 model seeds x 2 fresh identity pools, and exceeds the equal-energy non-MFT control by 0.0909--0.1001. | Macro-AUPR falls by 0.0721--0.0770 (mean -0.0749), driven by MFT AP -0.4414.  Oracle subtraction is diagnostic only, not a deployable preprocessing result. |
| deployable component recovery | P3o-R | A supervised complex STFT-mask separator raises compound-to-MFT waveform cosine from 0.6119 to about 0.944. | Its frozen inverse count estimator collapses to count 8 for all 15,840 views; separated Pearson is 0 in all seeds and no count gate passes.  Global waveform agreement is not component-support identifiability. |
| posterior-conditioned parameter attachment | P3o-L | A detached posterior-selected component table gives a same-direction change in all three seeds while preserving the base classification state and logits bitwise. | Oracle/predicted-gate NMAE changes are only 0.16%--0.18%, below the preregistered 2% effect threshold; the base safe/retrieval conjunction also fails in one seed. |
| direct compound MFT count observables | P3o-M/N/O/P/Q | Sparse peak selection, abstention, adjacent relations, complex coherence, and non-negative support each supply either local or average positive signal under equal-capacity controls. | None satisfies all three-seed whole/compound/low-count/ultra-close gates.  Their small gains cannot support an MFT count expert or predicted-gate full-12 claim. |

## Closed mechanism families

The following are frozen negative evidence rather than tunable candidates:

1. Directly subtracting exact MFT, adjusting its subtraction strength, or recalibrating thresholds to exploit its ISRJ-FPR reduction.
2. Turning P3o-R's waveform loss, mask width, count extractor, or count loss into a sweep on its consumed identities.
3. Continuing MFT-count expert search by changing sparse peaks, peak relations, local geometry, coherence, non-negative support curves, or resolution-abstention scores.
4. Retuning the detached posterior component table, its width, coefficient scale, or training weight.

## V7 three-mechanism stop rule: satisfied on 2026-08-03

The subsequent V7 experiments complete, rather than reopen, this closure.  They
cover three distinct deployable MFT-to-ISRJ hypotheses and all fail their
pre-registered formal-development gates on fresh identity roles:

| independent mechanism family | formal experiment | decisive result | frozen conclusion |
| --- | --- | --- | --- |
| training-time conditional decision geometry | P3n-A: MFT/ISRJ conditional softplus margin, with BCE and hard-negative controls | B3 does not consistently improve over B1/B2; all three worst-cell FPR values exceed 0.10 and 25 dB safety/recall gates fail | A label-conditioned margin changes the boundary but creates no missing low-power evidence. |
| inverse-delay support observable | P3o-T: fixed complex-LFM inverse-delay support residual, with zero-side-input and phase-scrambled controls | T2 gains over T0 are only 0.01250/0.01768/0.02045, while T1 and T3 are indistinguishable from T2 | The support summary does not contribute identifiable information beyond residual capacity plus the predicted MFT posterior. |
| local time-frequency phase dynamics observable | P3o-U: multi-scale phase increments and STFT dynamics, with posterior-only and time-bin-shuffled controls | U2 gains over U0 are only 0.01338/0.01705/0.01503, U2-U3 is approximately zero, and ISRJ recall falls about 3 percentage points in every seed | The observed gain is posterior-gated negative calibration, not ordered local phase dynamics, and it is not safe. |

These are not three random seeds or three hyperparameter variants.  They are,
respectively, a supervised objective/decision mechanism, a delay-support
measurement, and an ordered time-frequency measurement.  The explicit V7
rule is therefore met: **no fourth MFT-to-ISRJ architecture, residual, loss,
calibration, or single-channel compound-I/Q feature variant may be generated
or trained.**  P3n, P3o-T, and P3o-U use distinct identity roles and remain
separate evidence records; their numerical results must not be pooled.

All three records retain `test_generated=false`, `test_opened=false`, and
`locked_test_used=false`.  No confirmation role is generated for P3o-T or
P3o-U.

These closures follow from distinct failure modes, not a lack of training effort: P3o-S shows a real causal confound but loses indispensable MFT evidence; P3o-R fails support recovery even with high waveform cosine; P3o-L's deployable posterior correction is too small; and P3o-M--Q do not create sufficient compound-I/Q count information.

## Required gate before any new ISRJ--MFT architecture

No new router, residual classifier, expert, loss, or calibration may be trained merely because P3o-S has an oracle effect.  A successor requires all of the following before architecture design:

1. A materially independent, physically justified observation contract, rather than a recombination of the closed single-channel compound-I/Q delay features.  Examples require new acquisition authority: an additional receiver/antenna view, controlled PRI/diversity view, or another independently measured MFT reference.  These are examples of a data contract, not permission to synthesize or claim them now.
2. A fresh identity-isolated, train-free physical-sufficiency screen proving that the proposed observable separates `MFT+ / ISRJ-` from true ISRJ while preserving MFT evidence.  Its control must distinguish the new observation from arbitrary energy removal.
3. Only if the screen passes, a separately preregistered, equal-compute model comparison on new train/development identities.  It must improve predicted-gate full-12 parameter risk and source/worst-cell ISRJ FPR without MFT AP or macro-AUPR regression, followed by the planned multi-seed confirmation and cost audit.

Until such an observation exists, the evidence supports a mechanism diagnosis (MFT-induced ISRJ confusion), not a deployable Route-B innovation.  The safe research action is to preserve the frozen P1n/E5 reference and work on a different independent-observable branch or an authorized isolated-GPU cost proof.

## Source artifacts

- `docs/p3os_mft_oracle_subtraction_development_result_20260803.md`
- `docs/p3or_mft_component_separation_results_20260803.md`
- `docs/p3ol_component_conditioned_residual_results_20260803.md`
- `docs/p3om_sparse_peak_ordinal_results_20260803.md`
- `docs/p3on_resolution_abstention_results_20260803.md`
- `docs/p3oo_adjacent_peak_relation_results_20260803.md`
- `docs/p3op_complex_coherence_results_20260803.md`
- `docs/p3oq_nonnegative_support_curve_results_20260803.md`
- `docs/p3n_cross_interference_backbone_results_20260801.md`
- `docs/p3ot_mft_conditional_residual_development_result_20260803.md`
- `docs/p3ou_local_phase_dynamics_development_result_20260803.md`
