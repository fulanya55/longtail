# Route A H1u: separated-delay MFT learning preregistration

Date: 2026-08-03  
Status: preregistered before H1u source generation or model training.

## Question carried from H0u

H0u passed only the physical-information prerequisite. H1u asks one new,
falsifiable question: **does a fixed correct-LFM three-band evidence readout
improve weak MFT ranking in fresh six-label composite records beyond an
equal-parameter phase-order-permuted evidence control?**

It is a component H1 on a new 1 ms external protocol. It is not C3, does not
modify F2/RawSaw/VTSE-VBF on original development-v3, and cannot authorize
test-v3.

## Fresh source and roles

- New source: `复合干扰数据0803_h1u_mft_delay_aperture_r1`, created by
  `experiments/generate_route_a_h1u_mft_delay_aperture.m` after a fresh-ID
  audit. It contains all 63 non-empty six-label combinations x 80 = 5,040
  identities under the 1 ms contract.
- A present MFT component uses the same separated three-replica geometry as
  H0u. Within every MFT-containing composition, 40 identities are weak at 8
  dB and 40 are strong at 20 dB. All non-MFT components use the official
  generator at 20 dB. For each identity no-metadata inference input is allowed.
- For each of five receiver-grouped split seeds 3101--3105, stratify by full
  six-label composition and MFT state (absent/weak/strong) into fit 60%,
  validation 20% and untouched early 20%. No identity is shared between roles.
- H1u source identities must be disjoint from H0u and all readable historic
  train sources. No test source is generated. `locked_test_used=false`,
  `test_v2_used=false`, `test_v3_used=false` are mandatory.

## Frozen models and fair controls

All arms consume only RMS-normalized complex I/Q `[B,2,10000]`. A common
compact backbone has `2->16` strided Conv1d stem, two depthwise-separable
downsampling blocks `16->32->48`, global mean/std pooling, and a six-logit
linear classifier. Each arm also has the same 3-to-1 MFT residual linear
layer; it is bounded by `0.5*tanh(.)` and affects only the MFT logit.
Parameter counts and initialization are identical arm-wise.

| Arm | Fixed three-number residual input | Purpose |
| --- | --- | --- |
| B0 | all zeros | Capacity and optimization baseline |
| M1 | correct-LFM log peaks in fixed bands [160,300), [350,500), [540,700) us | Test physical evidence use |
| M2 | phase-order-permuted LFM log peaks in the same bands | Exact information-destroying control |

The feature is recomputed from I/Q inside the model with no metadata, no fit
and no threshold. M1/M2 have exactly the same input dimension, residual head,
backbone, budget and initialization.

## Objective and optimization

Each arm uses identical AdamW (`lr=3e-4`, weight decay `1e-4`, batch 32,
gradient clip 5, maximum 24 epochs, patience 6). The five deterministic model
seeds are 4101--4105. Checkpoint selection is only the validation MFT
weak-vs-negative AUROC.

The base loss is BCE for all non-MFT labels. MFT uses the mean of three
separately averaged BCE strata (negative, weak positive, strong positive), so
none dominates due to its count. Every arm uses this same loss; H1u tests the
evidence input, not a loss change. No calibration, threshold, F1 or FPR is
used in selection or primary endpoints.

## Endpoints and admission gates

Primary: early-role MFT weak-vs-MFT-absent AUROC. Secondary: weak AUPR,
strong-vs-negative AUROC, six-label macro-AUPR and non-MFT mean AUPR. Every
comparison is paired across the five same-role seeds with a 5,000-draw paired
bootstrap interval.

M1 may pass only if all conditions hold:

1. M1-M2 weak AUROC >= +0.030, >=4/5 positive seeds, CI lower bound >0;
2. M1-M2 weak AUPR >= +0.025 with CI lower bound >0;
3. M1-B0 weak AUROC >= +0.030, >=4/5 positive seeds and CI lower bound >0;
4. M1-B0 strong MFT AUROC is no worse than -0.010;
5. M1-B0 non-MFT mean AUPR is no worse than -0.010; and
6. all source/role/test-seal contracts pass.

Failure closes H1u against changes to the source, jitter, delay bands,
replica geometry, JNR, backbone, residual size, loss, optimizer, epoch budget,
calibration or threshold. A pass permits only a fresh-source replication;
it does not reconnect H1u to original development-v3 or test-v3.
