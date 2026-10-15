# Route A v24: matched-capacity multi-resolution weak-evidence probe result

## Decision

**Stop direct high-resolution STFT feature concatenation.** When compared with an equal-input-dimension, equal-parameter projection control, adding the 32x32 local STFT image materially degrades weak VFT ranking and also violates overall safety.

## Protocol and evidence boundary

v24 used a fifth new train-only source (base seed `660000000`; SHA-256 `513b7b0d9da2bace17ad277ebbc8d01854e572b92dadb86fc09efbf9569fdff8`) with 5,040 unique receiver identities. ISRJ-present fit/early-stop/evaluation roles were 1,536/512/512 identities. Evaluation contained 69 weak VFT positives, 187 strong VFT positives, and 256 matched VFT-absent/ISRJ-present hard negatives.

Both variants used a 1,292-dimensional input and the same 16,823-parameter `1292 -> 13 -> 1` GELU head, initialization, AdamW, BCE, batch size, training budget, and unweighted three-stratum early stop. `M_ProjControl` concatenated the original 268-dimensional feature with a fixed 1,024-dimensional Gaussian projection of itself. `M_MultiRes` instead concatenated the original feature with a 32x32 STFT image. The projection hash was exact across all seeds. No Q4 signal, calibration, threshold, historical data, locked test, test-v2, or test-v3 was accessed.

## Five-seed held-out evaluation

| Metric: M_MultiRes minus M_ProjControl | Mean delta | 95% paired bootstrap CI | Decision |
| --- | ---: | ---: | --- |
| weak VFT versus hard-negative AUROC | `-0.04457` | `[-0.05847,-0.03295]` | materially adverse |
| weak VFT versus hard-negative AUPR | `-0.08713` | recorded in summary | materially adverse |
| strong VFT versus hard-negative AUROC | `-0.00738` | recorded in summary | one seed fails safety |
| overall VFT AUROC | `-0.01740` | recorded in summary | safety fails |

All five weak-AUROC deltas were negative. Input/parameter matching, projection-hash consistency, source audit, and test-v3 sealing passed; every efficacy gate failed. The preregistered decision is `stop_multiresolution_morphology`.

## Mechanism conclusion

The original 16x16 feature is not simply missing recoverable information that can be restored by appending a larger STFT image to a small shared head. The local image competes with the stable global/track statistics and produces worse ordering than a parameter-matched no-new-information control. This rejects direct feature concatenation, not the broader need for label-conditioned local evidence.

Together with v21-v23, the narrow morphology-residual line has rejected coefficient selection, weak-mass reweighting, pairwise ranking, and naive multi-resolution concatenation. The next work item is a literature and existing-code audit before any further architecture change: a new candidate must separate local evidence extraction from global safe statistics (for example, a label-conditioned local branch with an explicit bypass), rather than concatenate another feature map into the same classifier.

## Artifacts

- Plan: `docs/plans/route_a_v24_multiresolution_weak_evidence_probe_plan_20260802.md`
- Generator/audit/runner: `experiments/generate_route_a_v24_multiresolution_weak_evidence_probe.m`, `experiments/audit_route_a_v24_multiresolution_weak_evidence_probe.py`, `experiments/run_route_a_v24_multiresolution_weak_evidence_probe.py`
- Formal artifacts: `outputs/tapc/route_a_v24_multiresolution_weak_evidence_probe/`
