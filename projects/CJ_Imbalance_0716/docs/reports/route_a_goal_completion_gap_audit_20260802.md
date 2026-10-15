# Route A VTSE-VBF goal completion gap audit

## Audit decision

The project does **not** currently satisfy the stated publishable VTSE-VBF goal. A six-label engineering composition exists, and RawSaw is a strong frozen SawFM component, but no target-label visibility mechanism has surpassed its matched-capacity control with the required stable effect size. Test-v3 must remain sealed. This is an evidence-gap statement, not a negative claim about all future visibility-aware designs.

## Requirement-by-requirement status

| Goal requirement | Current authoritative evidence | Status |
| --- | --- | --- |
| Six-label unified model | development-v3 C3 composes RawSaw for SawFM with VTSE-VBF for MFT/VFT/ISRJ | implemented, but not promoted |
| Preserve RawSaw SawFM advantage | RawSaw C1/C3 improves SawFM AUPR and weak recall; historical one-shot locked result is frozen | supported within its frozen boundary |
| Target MFT/VFT/ISRJ weak improvement | C3 weak recall gain versus C0 is `+0.00959`; weak AUROC `+0.00264` | below preregistered `+0.030` / `+0.015` gates |
| New VFT physical candidate | H0i passes a single-label intervention, but H1j composite weak-VFT sample bootstrap crosses zero for AUROC/AUPR | directional only; not promotable |
| Rule out parameter-count explanation | C4 same-capacity all-positive control exceeds C3 on macro and target weak metrics | contradicted for current hard-mask VTSE mechanism |
| Complete visibility ablation | Latest hash-audited core-14 matrix records direct current-protocol coverage for all 14 requested questions | coverage complete, but it does not promote the failed unified candidate |
| Fair mainstream baseline comparison | BCE, focal, ASL, DB, class-balanced and logit-adjustment development records exist; latest MLLT code is compatibility-audited | incomplete for a fully compatible recent external MLLT reproduction |
| Robustness/OOD/external evidence | selected frozen components have diagnostics; complete candidate has not passed promotion | incomplete |
| One-time independent test-v3 | sealed manifest says `access_authorized=false`, `analysis_allowed=false`, `used=false` | intentionally unexecuted |
| Complete paper-level innovation evidence | no promoted target visibility mechanism and no final frozen bundle | not established |

## Mechanisms that must not be revived by parameter search

1. Hard-mask, soft-visibility, and continuous weak-positive weighting as the primary VTSE training objective.
2. Complete high/low-power representation invariance.
3. Oracle/ordinal power-anchor calibration as a classification remedy.
4. Frozen F2 label-query, positional-query, or query-derived local-top-k pooling.
5. VFT/ISRJ low-dimensional logit fusion, predicted-width multiplication, or morphology residual-coefficient selection.
6. Weak-positive mass reweighting, pair ranking on the fixed 16x16 morphology feature, and direct 32x32 STFT concatenation.
7. Training-time same-nuisance counterfactual logit-margin supervision: H0e did not exceed its same-target shuffled-pair control and degraded VFT/ISRJ all-positive ranking.
8. Counterfactual delta-prototype target binding: H0f did not identify a target-specific paired difference strongly enough (mean binding accuracy `0.54750`, below `0.60`) and did not beat either BCE or cyclic wrong binding on weak AUROC.

Each is a failed *mechanism*, not merely an unselected hyperparameter. The already inspected sources must not be reused for another selection sweep.

## What a genuinely new candidate must prove first

Any restart has to be architecturally non-overlapping with the rejected routes and must be preregistered on a new identity-isolated train-only source before a six-label integration:

1. It has a target-specific local evidence extractor whose evidence map is not derived from the target classifier query/logit.
2. It keeps an unchanged global safety path and demonstrates, on frozen scores, that its local output is not an implicit threshold or prior shift.
3. It has a parameter- and information-matched control, not only a smaller baseline.
4. It improves weak-positive-versus-composition-matched-negative ranking independently for MFT, VFT, and ISRJ in five seeds.
5. Only then may it enter a fresh six-label end-to-end comparison against C0/C1 and a same-capacity non-visibility control, using role-isolated calibration and thresholds.
6. Only after all development, ablation, baseline, and robustness gates pass may code, calibration, thresholds and fusion be frozen and test-v3 authorized once.

## Immediate research posture

The prior `route_a_innovation_goal_final_decision_20260801.md` stop decision remains authoritative for the current unified candidate. Work may continue as design, code/read-only compatibility auditing, and literature-backed hypothesis formation; it must not reinterpret previously viewed development or sealed test data as selection evidence. A future experimental restart requires an explicitly new architecture/protocol document that demonstrates why it is not a renamed query, residual, weight, or feature-concatenation variant.

### 2026-08-02 H0g frozen ISRJ replication

H0g does not reopen any stopped weak-component mechanism. It froze the H1/B2
ISRJ aligned-periodicity artifacts and evaluated them on a new 5,040-sample,
identity-isolated high-replica source (ISRJ width 25--30 us, exactly 9
replicas). B2 minus B0 achieved ISRJ AUPR `+0.11927` (95% CI
`[+0.09376,+0.14503]`), weak recall `+0.07125`
`[+0.06314,+0.08078]`, and weak AUROC `+0.09965`
`[+0.08981,+0.11069]`, all 5/5 positive, with FPR `+0.00242`
`[-0.00395,+0.00863]`. RawSaw remained exactly unchanged. This is a third
parameter-regime replication of the *ISRJ component mechanism*, not evidence
that the stopped full MFT/VFT/ISRJ candidate is now external-robust.

### 2026-08-02 H0h frozen correlated-noise replication

H0h further freezes the same B0/B2 artifacts and changes only the receiver
noise process on a newly generated, 5,040-identity source: unit-power proper
complex AR(1) noise with rho `0.80`. The source audit preserves all labels and
metadata from its new white-noise base, verifies a distinct I/Q payload and
finds zero receiver overlap with all historical sources. B2 minus B0 achieves
ISRJ AUPR `+0.02242` (95% CI `[+0.01577,+0.02852]`), weak recall `+0.10270`
`[+0.09658,+0.10991]`, weak AUROC `+0.08037`
`[+0.07462,+0.08682]`, all 5/5 positive, and ISRJ FPR `-0.00468`
`[-0.01113,+0.00024]`. The complete all-label/receiver-modulo-five safety
table is descriptive only; its largest mean cohort-label FPR is `0.23387`.
This establishes one correlated-noise replication of the narrow ISRJ
increment, while leaving every full-candidate, MFT/VFT and test-v3 gap open.

### 2026-08-02 fixed ordinary-loss convergence closure

The prior baseline audit left one controlled concern: Focal/ASL/DB at 36
constant-LR epochs had not all reached an apparent plateau. A single
preregistered Focal-72 cosine schedule now completes five full runs with best
epochs `64,71,70,72,71` and near-zero final-ten-epoch changes. It produces no
meaningful gain over Focal-36 (ISRJ weak AUROC `-0.00056`, CI
`[-0.00138,+0.00019]`) and remains below frozen H1 linear BCE by `-0.03539`
weak AUROC and `-0.06744` weak recall, both 0/5 positive. This closes only
the fixed Focal schedule, not every possible loss recipe, and makes H1 the
necessary strong classifier reference for any future visibility mechanism.

### 2026-08-02 physical-ISRJ continuation

Two additional prerequisites narrow, rather than reopen, the ISRJ design
space. H0b showed strong **oracle** width-normalized delay-map replica
identifiability (mean R2 about `0.717`), whereas H0c stopped frozen point-width
substitution and H0d stopped the no-oracle fixed scale-Hough bank (mean real R2
`0.0288`; real-minus-permutation R2 `+0.0422`, 95% CI `[+0.0286,+0.0552]`, far
below its predeclared `+0.20` gate). Thus neither a width prediction nor a
fixed delay-scale bank is presently deployable physical evidence. This adds a
ban on tuning scale-Hough width grids, spacings, multipliers, pooling or CNN
capacity. Any next ISRJ extractor must use a physically different observable
and first pass a new identity-isolated H0 diagnostic.

### 2026-08-02 counterfactual-pair continuation

H0e supplied a stricter training-time test than ordinary weak-positive
reweighting: each pair had identical retained components and receiver noise,
and differed only by the target component. The CVMC logit-margin candidate
still failed against the information/compute-matched shuffled-pair control
(mean three-target weak-AUROC difference `+0.00083`, 95% CI
`[-0.00477,+0.00643]`) and harmed VFT/ISRJ all-positive AUROC. This closes
direct pair-to-logit margin supervision; it must not be tuned or relabeled as
a counterfactual VTSE variant on H0e identities.

### 2026-08-02 representation-level counterfactual continuation

H0f made the counterfactual object more stringent than H0e: it trained a
prototype classifier on the pooled temporal feature difference rather than
directly imposing a target-logit margin. On a fresh, non-overlapping pair
source, C_CDP versus B0 gave mean three-target weak-AUROC `-0.00185` (95% CI
`[-0.00890,+0.00590]`), and versus the cyclic wrong-binding control only
`+0.00274` (95% CI `[-0.00245,+0.00668]`). Its held-out target-binding accuracy
was `0.54750`, below the preregistered `0.60` identifiability gate; VFT/ISRJ
all-positive safety also failed. This closes pair-difference prototype
binding in the current temporal encoder. It must not be revisited by tuning
prototype width, temperature, loss weight, residual amplitude or fusion.

### 2026-08-02 VFT physical-statistic continuation

H0i established that a known-LFM dechirped adjacent-phase-increment statistic
responds to a fresh single-label, paired VFT Doppler intervention. H1j then
held the statistic and the phase-permuted matched control fixed on a separate
5,040-identity six-label composite source. The five fixed optimizer runs had
the same weak-VFT point difference (AUROC `+0.048715`, AUPR `+0.017996`), but
the one-dimensional heads preserve the score ranking, so optimizer-seed
variation is not an independent uncertainty estimate. A post-hoc fixed-score,
10,000-draw class-stratified paired bootstrap gives AUROC
`[-0.032188,+0.132208]` and AUPR `[-0.028179,+0.069921]`; both cross zero.
Consequently the scalar is **not promotable** into a fusion or end-to-end
candidate. H1j identities must not be used for window/reference/frequency/
normalization/training/threshold searches. Any new VFT route must use a
physically different, predeclared local observable and a new identity source
with sample-level paired uncertainty predeclared.

### 2026-08-03 MFT matched-filter effective-support continuation

H0j verified a narrow physical sanity check: a correct-LFM matched-filter
Renyi-2 effective-delay-support statistic separated paired, single-MFT
8-copy from 3-copy records (AUROC `1.00000`; correct-minus-phase-permuted
control `+0.35831`, CI `[+0.33055,+0.38610]`). H1k then kept the statistic,
ROI, phase-permuted control and zero-fit rule fixed on a new 5,040-identity
six-label source. On 113 weak MFT positives versus 480 exact
other-five-label-context matched negatives, the candidate is adverse:
AUROC `-0.16855`, CI `[-0.23240,-0.10387]`, and AUPR `-0.07566`, CI
`[-0.13319,-0.04212]`; all five fixed receiver subfolds are negative.
The statistic measures global matched-filter response spread under composite
background rather than MFT-attributable evidence. It is therefore stopped
against any ROI/entropy/reference/normalization/readout/residual/fusion
search on H0j or H1k identities.

### 2026-08-03 MFT sparse-attribution continuation

H0l replaced the stopped global support statistic by a genuinely different,
fixed three-atom complex OMP reconstruction. It passes a new 640-pair
single-MFT count intervention (correct-minus-permuted AUROC `+0.45844`, CI
`[+0.42975,+0.48740]`). H1m then freezes the three atoms, `[400,999]`
dictionary, candidate/control and zero-fit rule on a new 5,040-identity
six-label source. It is again adverse on 117 weak MFT positives versus 496
context-matched negatives: AUROC `-0.11215`, CI `[-0.16967,-0.05471]`; AUPR
`-0.03372`, CI `[-0.06730,-0.01627]`; four of five receiver subfolds are
negative. Together H1k and H1m show that this is not merely an entropy/peak
summary failure: fixed single-pulse LFM matched-filter dictionaries cannot
attribute MFT under the present composite background. No matched-filter
dictionary/atom-count/regularization/ROI/normalization/readout/fusion search
is allowed on H0j/H1k/H0l/H1m identities.

### 2026-08-03 multi-pulse MFT coherence continuation

H0n first verified a narrow new-observation invariant: when an isolated MFT
component is deliberately given coherent versus independently phase-scrambled
four-PRI copies, the fixed correct-LFM top-8 cross-PRI complex coherence has
AUROC `1.00000`, whereas its phase-invariant noncoherent-amplitude control is
`0.50052`. This H0 alone was not a classification claim. H0o then performed
the required official-generator paired composite test on 640 new receiver
identities: FMB, SawFM, SinFM, VFT and ISRJ, receiver noise, sample seed and
their type-ID substreams are exactly shared; present alone adds weak MFT at
effective JNR 8 dB (-12 dB relative to each 20 dB background component).
The coherence statistic beats its amplitude control by AUROC `+0.09316`
(95% CI `[+0.08206,+0.10612]`) and AUPR `+0.09134`
(`[+0.07828,+0.10787]`), but its absolute MFT-present/absent AUROC is only
`0.58066`, below the preregistered `0.60` requirement. It is therefore not
permitted to enter a multi-pulse network, parameter search, threshold,
calibration or fusion. This closes global top-k cross-PRI complex-coherence
delay images as a weak-MFT attribution route; no existing single-pulse
dev-v3/test-v3 conclusion changes.

### 2026-08-03 single-PRI no-reference MFT autocorrelation continuation

H0p deliberately removed both the known-LFM reference and the multi-PRI
assumption. On a fresh 640-pair single-MFT 3-vs-8 intervention, nonzero-lag
complex autocorrelation Renyi-2 effective support reaches AUROC `0.99815`.
However, a fixed sample-phase-index permutation control run through the same
lag statistic reaches `0.96499`; the candidate-control gap is only `+0.03316`
(95% CI `[+0.02399,+0.04289]`), below the preregistered `+0.10` requirement.
This is a decisive specificity failure, not a lack of raw count sensitivity:
the statistic measures general temporal correlation complexity. H0p closes
global nonzero-lag autocorrelation effective-support readouts for MFT; lag,
Renyi-order, control, peak, normalization, threshold, head and fusion search
on its identities are forbidden.

### 2026-08-03 MFT/VFT architecture admission decision and ISRJ evidence bundle

The proposed blind time-frequency ridge-count continuation was rejected before
source generation: reference-aligned ridge multiplicity is a coordinate change
of the stopped MFT delay-profile/RAAT topology, reference-free ridge morphology
is a prohibited generic STFT-capacity route, and cepstral/correlation variants
are within H0p's stopped delayed-copy complexity family. No MFT/VFT H0,
training, calibration, threshold search or new source was started. The project
therefore has no admissible MFT/VFT candidate under the current evidence
boundary. In parallel, a read-only hash-addressed ISRJ evidence bundle records
the two frozen external replications H0g/H0h and the nonselection co-label
audit. It confirms all artifact hashes and preserves `locked_test_used=false`,
`test_v2_used=false`, `test_v3_used=false` and the sealed test-v3 state. This
strengthens only the narrow ISRJ aligned-periodicity claim; it does not alter
the unified-model gap or authorize test-v3.

### 2026-08-03 protocol identifiability-budget continuation

The 0716 contract itself supplies a physical limit on the remaining targets:
VFT Doppler is 1--7 kHz but each record contains one 100 us active pulse, so
the span is only 0.6 of a 10 kHz single-pulse Fourier scale and accumulates
0.1--0.7 cycles. MFT uses 3--8 delayed copies within a 30--50 us relative
delay band, hence all copies overlap for at least 80% of a 100 us pulse while
their typical per-copy energy decreases under fixed total component JNR. This
does not make recognition impossible, but it explains why H0 single-component
signals have not become target-attributable weak evidence in composite H1.
Any future VFT mechanism requires a separate multi-PRI coherent-aperture
protocol; any future MFT mechanism requires a separately specified delay
aperture/minimum-separation protocol. Neither may be presented as a retuned
result on the sealed single-pulse 0716 test-v3.

### 2026-08-03 H0q staggered-PRI VFT physical continuation

The separate-observation requirement was tested without altering the original
0716 protocol. A new 640-pair source uses eight 100 us LFM pulses with
alternating 300/331 us intervals, continuous VFT absolute-time phase and new
receiver identities. The fixed true-timestamp Doppler consistency contrast
achieves high-versus-low AUROC `0.983069`, while a control with identical
complex coefficients but fixed permuted timestamps achieves `0.335610`; the
paired AUROC difference is `+0.647458`, 95% CI `[+0.622980,+0.672257]`, and
the AUPR difference is `+0.589808`, CI `[+0.575519,+0.600384]`. All H0q gates
pass with no fit, calibration, threshold selection or test usage. This is a
positive **new-observation VFT physical identifiability** result only. It
does not repair the original single-pulse H1j result or promote any six-label
candidate. The only permitted next action is a newly identity-isolated,
composition-matched staggered-PRI VFT present/absent H1.

### 2026-08-03 H1a staggered-PRI MFT+ISRJ VFT attribution

The authorized H1a follow-up now passes on a distinct 640-pair source. Each
pair has exactly shared receiver noise, MFT (20 dB), ISRJ (20 dB), component
substreams and pulse timing; present alone adds VFT at 8 dB. The source is
explicitly a project-local eight-pulse staggered-PRI physics extension, not
official uniform-PRI 0716. The frozen true-timestamp full-grid coherence
candidate yields VFT present/absent AUROC `0.628534`, versus `0.530300` for the
same computation under a fixed timestamp permutation. The paired AUROC gain is
`+0.098234`, 95% CI `[+0.074093,+0.122673]`; AUPR gain is `+0.100520`, CI
`[+0.074643,+0.125485]`; and `0.645313` of paired present scores exceed absent.
All H1a gates pass with zero fit and a sealed test-v3. This promotes only the
new-observation VFT physical route to the required fresh, all-five-background
H1; it does not promote a six-label model or change any original-0716 result.

### 2026-08-03 H1b staggered-PRI all-background VFT attribution

The permitted all-five-background continuation also passes on a third,
identity-isolated 640-pair source. Within each pair receiver noise plus FMB,
SawFM, SinFM, MFT and ISRJ (20 dB each) are exactly shared; present alone adds
VFT at 8 dB. The H1a candidate/control were frozen before source generation.
Candidate AUROC is `0.665583` versus control `0.550112`, a paired difference
of `+0.115471` (95% CI `[+0.093257,+0.138852]`); AUPR difference is
`+0.097358` (`[+0.070951,+0.122757]`) and paired direction is `0.720313`.
The complete source and test-seal audit passes. Thus H0q/H1a/H1b establish a
new multi-pulse VFT physical-observability chain, but still do not repair the
original single-pulse task, establish a neural model, or authorize test-v3.
The next required evidence is a new role-isolated multi-seed learned-model
comparison with the true-timestamp feature versus a same-capacity
timestamp-permuted control.

### 2026-08-03 H3 staggered-PRI three-strata learned-model closure

That required learned-model comparison is now complete on a new three-arm,
identity-isolated source: absent, weak VFT at 8 dB and strong VFT at 20 dB
share the same background and receiver noise within each group. The zero-fit
physical statistic remains strongly discriminative for weak VFT (true-timestamp
AUROC `0.77903` versus permutation `0.57095`, difference `+0.20808`, paired
CI `[+0.18937,+0.22686]`). However, the learned T1 true-timestamp additive
expert is consistently worse than the equal-parameter T2 timestamp-permuted
expert: weak AUROC difference `-0.00137`, CI `[-0.00188,-0.00086]`, and weak
AUPR difference `-0.00106`, CI `[-0.00146,-0.00064]`, both 0/5 positive.
Strong-VFT safety versus T0 passes but does not rescue the target mechanism.
H3 is therefore stopped under its preregistered rule. It establishes neither
a neural VFT contribution nor a unified candidate, and prohibits all H3
timestamp/ROI/capacity/JNR/optimizer/loss/threshold searches. The detailed
decision is `docs/reports/route_a_h3_vft_three_strata_stop_decision_20260803.md`.
All H3 roles are development-only; locked test, test-v2 and sealed test-v3
remain unused.

### 2026-08-03 H4 full delay-frequency beamspace-map P0 closure

H4 was admitted only as an information-path change: retain the fixed
`3 x 25 x 401` normalized complex delay-frequency coherence map rather than
H3's frequency-wise delay maximum. On a new 640-identity absent/weak/strong
counterfactual source, the fixed no-fit five-delay concentration statistic has
weak-VFT true AUROC `0.51769` versus permuted `0.50798`, a difference
`+0.00971` (CI `[+0.00538,+0.01404]`). It therefore passes only the direction
check, not its absolute `0.60` AUROC or `+0.060` control-difference gates.
P0 is false; M1/M2 are not implemented or trained. This closes the proposed
full-map local-concentration route against delay-kernel/ROI/grid/encoder/loss/
JNR/threshold search on H4 identities. It does not contradict the separate
H0q/H1a/H1b physical maximum-coherence evidence, but it supplies no neural
VFT promotion. See `docs/results/route_a_h4_vft_beamspace_p0_results_20260803.md`.

### 2026-08-03 PGL official-code reproduction closure

The new literature refresh identified the authors' public PGL repository for
the AAAI 2023 Probability Guided Loss. A read-only audit pins commit
`d11a6520b0241946672364d974a5ebbd43c554e0` and its `pg_loss.py` blob SHA.
That exact source fails Python AST parsing (missing colon at line 148) and
also contains hard-coded CUDA/image-framework dependencies and references to
undefined `apaf` and `neg_scale1`. It cannot be faithfully ported as a loss
under the PGL name; patching it would be an unregistered custom method, not a
fair baseline. Thus the latest external-method numeric-comparator gap remains
open, but is now supported by a second concrete, pinned-code audit rather than
an unexamined omission. See
`docs/reports/route_a_pgl_official_code_compatibility_audit_20260803.md`.

### 2026-08-03 RAL fair recent-loss baseline closure

The ICCVW 2023 RAL repository contains an auditable logits/labels `Ralloss`,
so it was run under the same five seeds, F2 initialization, RawSaw routing,
36-epoch budget and calibration/threshold roles as Focal-36. RAL has a small
all-five-seed target weak-AUROC increase `+0.00288` (CI
`[+0.00214,+0.00365]`), but macro-AUPR is `-0.00282` (0/5 positive), macro-F1
is `-0.00200`, ISRJ AUPR is `-0.00673` (0/5), ISRJ weak recall is `-0.00509`
(0/5), and ISRJ strong recall is `-0.01780` (0/5). It therefore cannot replace
Focal-36 or explain the aligned-periodicity increment. RAL closes one
formula-traceable recent loss baseline without authorizing parameter/threshold
search or test-v3; see
`docs/results/route_a_development_v3_ral_loss_baseline_results_20260803.md`.

### 2026-08-03 H0r higher-order MFT closure

H0r tested a mathematically non-equivalent third-order route before any
network: fixed segmented bicoherence for new paired 3-copy/8-copy MFT records,
against a DFT random-phase control that preserves every spectral magnitude and
circular second-order autocorrelation to machine precision. The candidate
AUROC is `0.43065`, control AUROC `0.52919`, and candidate-minus-control
`-0.09854` (CI `[-0.14238,-0.05562]`); 8>3 direction is `0.42656`.
The second-order matching contract passes, so H0r is a decisive target-readout
failure rather than a control failure. It is stopped against all bins/segments/
copy-count/JNR/surrogate/network/loss/threshold searches and never enters H1.
See `docs/results/route_a_h0r_mft_higher_order_results_20260803.md`.

### 2026-08-03 H0s joint ISRJ parameter-plus-noise replication

H0s-R1 combines the predeclared high-replica parameter regime (ISRJ width
25--30 us, exactly 9 replicas) with proper-complex AR(1) receiver noise at
rho `0.80`, while keeping B0/B2 artifacts, calibration and thresholds
completely frozen. The source has 5,040 new receiver identities and passes
zero-overlap, metadata, I/Q-change and test-v3-seal audits. B2 minus B0
achieves ISRJ AUPR `+0.11677` (CI `[+0.09305,+0.13667]`), weak recall
`+0.07285` (`[+0.05695,+0.08278]`) and weak AUROC `+0.09709`
(`[+0.08875,+0.10544]`), all 5/5 positive, with ISRJ FPR `+0.00113`
(`[-0.00363,+0.00653]`) and exact RawSaw preservation. An initially reserved
R0 identity interval was rejected before inference because it collided with
a historical source; the retained R1 interval was selected solely from a
read-only identity registry, without changing any physical or model decision.
This is a stronger but still **narrow ISRJ component** replication. It does
not repair the failed MFT/VFT mechanism, promote C3, or authorize test-v3.

## Linked records

- `docs/results/route_a_unified_rawsaw_vtse_vbf_v3_results_20260731.md`
- `docs/results/route_a_innovation_goal_final_decision_20260801.md`
- `docs/results/route_a_vtse_continuous_visibility_v3_results_20260731.md`
- `docs/results/route_a_dfms_iq_label_local_saw_results_20260724.md`
- `docs/results/route_a_vft_isrj_label_query_source_v3_results_20260802.md`
- `docs/reports/route_a_v19_v24_morphology_evidence_ledger_20260802.md`
- `docs/results/route_a_h0e_counterfactual_visibility_margin_results_20260802.md`
- `docs/results/route_a_h0f_counterfactual_delta_prototype_results_20260802.md`
- `docs/results/route_a_h0g_isrj_parameter_ood_v2_frozen_results_20260802.md`
- `docs/results/route_a_h0h_ar1_noise_frozen_results_20260802.md`
- `docs/results/route_a_h0s_isrj_combined_parameter_noise_frozen_results_20260803.md`
- `docs/results/route_a_development_v3_focal72_cosine_convergence_results_20260802.md`
- `docs/results/route_a_h1j_vft_dechirped_composite_results_20260802.md`
- `docs/results/route_a_h0j_mft_matched_filter_support_results_20260802.md`
- `docs/results/route_a_h1k_mft_matched_filter_composite_results_20260803.md`
- `docs/results/route_a_h0l_h1m_mft_omp_evidence_20260803.md`
- `docs/results/route_a_h0n_mft_crosspulse_coherence_results_20260803.md`
- `docs/results/route_a_h0o_mft_crosspulse_composite_results_20260803.md`
- `docs/results/route_a_h0p_mft_delay_autocorrelation_results_20260803.md`
- `docs/results/route_a_h1a_vft_staggered_pri_mft_isrj_results_20260803.md`
- `docs/results/route_a_h1b_vft_staggered_pri_all_backgrounds_results_20260803.md`
- `docs/results/route_a_h3_vft_three_strata_results_20260803.md`
- `docs/reports/route_a_h3_vft_three_strata_stop_decision_20260803.md`
- `docs/reports/route_a_h4_vft_beamspace_map_admission_audit_20260803.md`
