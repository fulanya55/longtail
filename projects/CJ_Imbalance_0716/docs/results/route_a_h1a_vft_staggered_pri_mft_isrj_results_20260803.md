# Route A H1a: staggered-PRI weak VFT in MFT+ISRJ backgrounds

## Decision

**H1a passes its preregistered restricted-composite gate.** A zero-fit, fixed
absolute-time Doppler-coherence statistic retains discriminative information
for an 8 dB VFT component in a counterfactually matched 20 dB MFT plus 20 dB
ISRJ background. This is evidence for the new staggered-PRI observation
protocol, not a result on the original single-pulse 0716 six-label task.

## Frozen protocol

- 640 new receiver-noise identities (`2898677742--2898678381`), distinct from
  history and from H0q; each pair shares receiver noise, MFT, ISRJ, their
  substreams, physical parameters, and pulse starts.
- Absent arm is MFT + ISRJ + receiver noise. Present arm changes only by adding
  VFT at effective JNR 8 dB; MFT and ISRJ are each calibrated to 20 dB.
- Eight 100 us LFM pulses use alternating 300/331 us intervals. The source is
  explicitly `project_local_staggered_pri_physics_extension`, not official
  uniform-PRI 0716 data.
- Candidate: maximum normalized LFM-delay coherence over a fixed
  `1000:250:7000 Hz` grid with true pulse timestamps. Matched control: the
  identical coefficients, grid, delay range and computation with timestamps
  permuted as `[0,5,2,7,4,1,6,3]`.
- No trainable parameter, refit, calibration, decision threshold, or test data
  was used. Uncertainty is a 5,000-draw paired bootstrap.

## Formal result

| Metric | Value |
| --- | ---: |
| candidate AUROC | `0.628534` |
| permuted-timestamp control AUROC | `0.530300` |
| AUROC difference | `+0.098234`, 95% CI `[+0.074093,+0.122673]` |
| candidate AUPR | `0.634593` |
| control AUPR | `0.534074` |
| AUPR difference | `+0.100520`, 95% CI `[+0.074643,+0.125485]` |
| paired present score > absent score | `0.645313` |

All preregistered gates pass: candidate AUROC >= 0.62, AUROC and AUPR
candidate-control differences meet their magnitude gates with positive lower
CI limits, and paired direction is >= 0.60. The source audit also passes all
eleven contracts, including the VFT-only intervention, historical/H0q identity
isolation, and the intact test-v3 seal.

## Boundary and next gate

H1a does not prove weak VFT recovery in the original one-pulse source, a
six-label classifier improvement, model-network novelty, OOD robustness, or
any test-v3 result. It authorizes exactly one next physical step: a fresh,
composition-matched five-background (FMB/SawFM/SinFM/MFT/ISRJ) staggered-PRI
VFT H1. That source must use new identities, a fixed statistic/control, and a
separate audit; it must not reuse H1a identities or tune the score.

## Artifacts

- Generator: `experiments/generate_route_a_h1a_vft_staggered_pri_mft_isrj.m`
- Source audit: `outputs/tapc/route_a_h1a_vft_staggered_pri_generation/paired_identity_audit.json`
- Formal metrics: `outputs/tapc/route_a_h1a_vft_staggered_pri/route_a_h1a_vft_staggered_pri_summary.json`
- Source SHA-256: absent `4ac7c7bb1a29496357e3bc1eca208ca53422af293cafa29ce060cb59dce18fe7`; present `a5da13c3292feceefc4f39378a8cc0c2ffe928b75dab07586d4f06c882273257`.

Every artifact records `locked_test_used=false`, `test_v2_used=false`, and
`test_v3_used=false`.
