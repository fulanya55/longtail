# Route A H1u: separated-delay MFT learning result

Date: 2026-08-03  
Status: **failed the preregistered development gate; H1u closed.**

## Question and evidence role

H0u established that a fixed correct-LFM readout can distinguish paired
separated-delay MFT records in a new one-PRI physical protocol. H1u tested a
strictly different, falsifiable question: whether injecting three correct-LFM
log peaks through a bounded one-logit residual improves learned weak-MFT
ranking beyond both a zero-input capacity arm and a phase-order-permuted,
equal-parameter control.

This is fresh-source component-learning evidence only. It is neither a C3
result nor evidence for the original 0716 development-v3 task.

## Frozen protocol actually executed

| Item | Value |
|---|---:|
| Fresh source | 5,040 identities; 63 full six-label combinations x 80 |
| Roles per split | fit / validation / early = 3,024 / 1,008 / 1,008 |
| Split/model seeds | 3101--3105 / 4101--4105 |
| Arms | B0 zero residual input; M1 correct-LFM peaks; M2 phase-order control |
| Common model | RMS I/Q, 2->16 stem, 16->32->48 depthwise blocks, mean/std pool, six logits |
| Parameters | 3,802 in every arm |
| Optimizer | AdamW, lr 3e-4, weight decay 1e-4, batch 32, clip 5 |
| Epoch rule | max 24, patience 6; select only validation weak-MFT AUROC |
| Objective | non-MFT BCE + equal mean BCE over MFT absent/weak/strong strata |
| Calibration / threshold | none / none; ranking endpoints only |

The source audit passed all payload, role, receiver-identity, historic-train
disjointness, H0u disjointness, and test-v3-seal contracts. The synthetic
smoke and 5 unit tests passed. All checkpoints, histories, role files, early
predictions, metrics, run receipt and source audit are preserved below.

## Early-role paired outcomes

| Paired comparison | Mean delta | 95% bootstrap CI | Positive seeds |
|---|---:|---:|---:|
| M1 - M2 weak MFT AUROC | -0.07933 | [-0.19184, +0.00001] | 2 / 5 |
| M1 - M2 weak MFT AUPR | -0.12362 | [-0.28807, -0.00004] | 2 / 5 |
| M1 - B0 weak MFT AUROC | -0.07970 | [-0.19169, -0.00049] | 0 / 5 |
| M1 - B0 strong MFT AUROC | -0.02057 | [-0.05709, -0.00063] | 0 / 5 |
| M1 - B0 non-MFT mean AUPR | -0.13006 | [-0.26344, +0.00256] | 1 / 5 |

All pre-registered admission checks are false except source provenance and
the test-v3 seal. In particular, M1 neither exceeds M2 or B0 by the required
weak-MFT AUROC/AUPR margins, nor meets strong-MFT/non-MFT safety limits.

## Failure diagnosis

The diagnostic is explanatory only and did not select a model or change the
H1u decision.

1. This is not an absent-physics failure. The learned M1 residual alone has
   mean weak-vs-absent AUROC 0.9271 and AUPR 0.9249, while M2 has 0.4502 and
   0.3253. H0u's correct-LFM observability result remains valid.
2. It is not residual saturation. Across all M1 weak positives, the absolute
   bounded-residual saturation rate (>= 0.49 of its 0.5 bound) is zero.
3. It is a downstream incremental-value failure. B0 already reaches early
   weak-MFT AUROC 0.9944--0.9998 in four of five role/model seeds. A
   low-dimensional statistic can be physically discriminative yet provide no
   reliable remaining ranking margin to a raw-I/Q classifier.
4. Two M1 seeds stop after epoch 7 with their best validation checkpoint at
   epoch 1 and exhibit broad non-MFT degradation. The fixed physics residual
   changes the shared multi-label optimization trajectory; bounded magnitude
   alone does not isolate that gradient interaction.
5. There is genuine complementary information, but too little error mass in
   this source for an unconditional six-label residual. B0 weak-vs-absent
   pair inversions are only 0.00020--0.00558 per seed (25--708 of 126,976
   pairs); M1's residual orders 0.671--1.000 of those rare inverted pairs
   correctly. This is a post-hoc diagnostic, not a license to add a gate or
   rerun H1u.

## Decision and boundary

Per the preregistration, **H1u is closed**. Do not retry it by modifying the
source, jitter, delay bands, replica geometry, JNR, backbone, residual size,
loss, optimizer, epochs, calibration or threshold. The result cannot be used
to promote an original-0716 unified candidate or to authorize test-v3.

No historical locked test, test-v2, or test-v3 data were opened or used.

## Artifacts

- [Preregistration](../plans/route_a_h1u_mft_delay_aperture_learning_20260803.md)
- [Source audit](../../outputs/tapc/route_a_h1u_mft_delay_aperture_generation/source_audit.json)
- [Formal summary](../../outputs/tapc/route_a_h1u_mft_delay_aperture_learning/summary.json)
- [Run receipt](../../outputs/tapc/route_a_h1u_mft_delay_aperture_learning/run_receipt.json)
- [Failure diagnostic](../../outputs/tapc/route_a_h1u_mft_delay_aperture_learning/failure_diagnostic.json)
- [Runner](../../experiments/run_route_a_h1u_mft_delay_aperture_learning.py)
