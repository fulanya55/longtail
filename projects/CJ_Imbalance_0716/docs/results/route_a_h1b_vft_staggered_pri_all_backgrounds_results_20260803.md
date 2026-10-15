# Route A H1b: all-five-background staggered-PRI weak VFT attribution

## Decision

**H1b passes.** The fixed absolute-time coherence observable remains specific
to VFT presence after the background is expanded from H1a's MFT+ISRJ to all
five non-VFT types (FMB, SawFM, SinFM, MFT and ISRJ). This closes the required
physical attribution gate for a changed multi-pulse VFT observation protocol.
It does not establish a learned model or an original-0716 six-label gain.

## Frozen protocol

- 640 new present/absent pairs use eight 100 us LFM pulses with alternating
  300/331 us PRI. Both arms share receiver noise and all five background I/Q
  components, their substreams and physical parameters; each background is
  individually 20 dB. Present alone adds VFT at effective 8 dB.
- FMB/SawFM/SinFM ranges and formulas reproduce the read-only official
  generator; nonuniform pulse timing makes this a `project_local` extension,
  not official uniform-PRI 0716.
- Candidate/control remain H1a's fixed normalized LFM-delay coherence on
  `1000:250:7000 Hz`, true timestamps versus `[0,5,2,7,4,1,6,3]`, with no
  trainable parameter, refit, calibration or threshold. Uncertainty is a
  5,000-draw paired bootstrap.

## Formal results

| Metric | Value |
| --- | ---: |
| candidate AUROC | `0.665583` |
| permuted-timestamp control AUROC | `0.550112` |
| AUROC difference | `+0.115471`, 95% CI `[+0.093257,+0.138852]` |
| candidate AUPR | `0.651108` |
| control AUPR | `0.553750` |
| AUPR difference | `+0.097358`, 95% CI `[+0.070951,+0.122757]` |
| paired present score > absent score | `0.720313` |

All preregistered gates pass. The source audit passes payload/paired metadata,
five-background calibration, VFT-only intervention, historical/H0q/H1a
identity isolation, and the intact test-v3 seal.

## Boundary and next gate

The H0q -> H1a -> H1b chain supports a physical VFT observability claim for a
new staggered multi-pulse protocol. It is not a raw-grid innovation claim, a
neural-network result, original 0716 evidence, OOD evidence, or test-v3
result. The admissible next stage is a role-isolated multi-seed learned model:
compare a shared I/Q backbone, the same-capacity backbone with the fixed
true-timestamp side feature, and the same feature under timestamp permutation;
preserve the RawSaw path and evaluate VFT weak ranking plus all-label safety.

## Artifacts

- Generator: `experiments/generate_route_a_h1b_vft_staggered_pri_all_backgrounds.m`
- Plan: `docs/plans/route_a_h1b_vft_staggered_pri_all_backgrounds_20260803.md`
- Audit: `outputs/tapc/route_a_h1b_vft_staggered_pri_generation/paired_identity_audit.json`
- Formal metrics: `outputs/tapc/route_a_h1b_vft_staggered_pri/route_a_h1b_vft_staggered_pri_summary.json`
- SHA-256: absent `5a9a40ad7b3210d4e09d4af511d116c4d35340483ae770df2bdfeb1d32fc4838`; present `18bb93af7e852cb5790aed1649575887756adcf74a2532b9b21f37ed377e7891`.

Every artifact records `locked_test_used=false`, `test_v2_used=false`, and
`test_v3_used=false`.
