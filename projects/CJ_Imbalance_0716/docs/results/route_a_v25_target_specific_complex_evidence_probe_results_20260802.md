# Route A v25: target-specific complex local-evidence probe result

> **2026-08-03 evidence correction.** The cyclic-source comparator was later
> proved permutation-equivalent when initialized correctly. The formal runner
> did not apply that required parameter permutation, so the historical
> candidate-minus-control deltas below can contain branch-initialization
> assignment effects. Preserve them as an exploratory architecture screen only:
> they are **not** admissible source-controlled causal evidence for MFT, VFT,
> or ISRJ and cannot justify a confirmation run. The executable correction is
> [the cyclic-control validity audit](../reports/route_a_v25_cyclic_control_validity_audit_20260802.md).

## Decision

**Stop TS-CLE as a joint MFT/VFT/ISRJ expert.** Its joint promotion gate failed
independently of the later comparator correction. The historical raw-complex
branch differences also harm VFT weak ranking and violate strong-stratum
safety, but their target-specific causal interpretation is invalid because the
cyclic control was not permutation-matched. It cannot be integrated into the
six-label model or evaluated on test-v3.

## Fresh protocol

v25 used a new train-only source at base seed `680000000`, SHA-256 `4cdb94e7e58b28f37306bcd508f6f70800efcaac99b40a7c1a53569aa33711bf`, with 5,040 unique receiver identities. Fit/early/evaluation roles were 3,024/1,008/1,008 identities with exactly 512 positives per label in each held-out role.

The 5,133-parameter TS-CLE model first trained a six-label global complex-I/Q path. The stem, normalization, and global classifier were then frozen. Only 4,743 target-local parameters were trained for MFT/VFT/ISRJ. The cyclic-source control had identical parameter count, samples, batch ordering, optimizer, epochs and early criterion, but the historical runner used independently assigned branch states rather than the required cyclic parameter permutation. FMB/SawFM/SinFM outputs were exactly equal in every seed. Weak masks were evaluation-only metadata.

## Historical five-seed exploratory differences (not causal evidence)

| Target | weak AUROC delta: TS-CLE minus historical cyclic run | 95% paired bootstrap CI | Permitted interpretation |
| --- | ---: | ---: | --- |
| MFT | `+0.01877` | `[+0.00229,+0.04067]` | exploratory only; not target-specific proof |
| VFT | `-0.07334` | `[-0.13831,-0.01178]` | adverse exploratory screen, not a valid source-control effect |
| ISRJ | `+0.03992` | `[-0.01874,+0.09921]` | exploratory only; not target-specific proof |
| three-target mean | `-0.00488` | `[-0.01672,+0.00603]` | failed joint gate; no integration |

The target mean failed magnitude, confidence-interval and 4-of-5 gates; VFT
made the per-target nonnegative gate fail; at least one target/seed strong-AUROC
delta fell below the `-0.010` safety floor. Parameter/trainable contracts,
non-target exact safety, identity audit, and test-v3 seal passed. The formal
decision is `stop_target_specific_complex_evidence`. The later equivalence audit
adds a stricter disposition: none of the displayed target deltas may be cited
as valid causal evidence.

## Mechanism inference

This is not evidence that a raw-complex local cue exists for MFT, VFT or ISRJ:
the cyclic-routing comparator does not remove target supervision and becomes
exactly the candidate under the correct parameter permutation. What remains
valid is only the negative joint disposition under this exploratory screen;
there is no eligible MFT-only confirmation or unified integration. A future
raw-local hypothesis requires a control that destroys declared physical
information (or substitutes a genuinely target-distinct inductive bias) while
matching the initial function, capacity, optimizer and supervision. It is
invalid to rescue v25 by changing its routing cycle, residual scale, loss
weights, top-k, epochs or thresholds.

## Evidence boundary and artifacts

All formal records state `locked_test_used=false`, `test_v2_used=false`, `test_v3_used=false`, and `test_v25_used=false`.

- Plan: `docs/plans/route_a_v25_target_specific_complex_evidence_probe_plan_20260802.md`
- Model and unit tests: `cjlt/models/target_specific_complex_evidence.py`, `tests/test_target_specific_complex_evidence.py`
- Generator/audit/runner: `experiments/generate_route_a_v25_target_specific_complex_evidence_probe.m`, `experiments/audit_route_a_v25_target_specific_complex_evidence_probe.py`, `experiments/run_route_a_v25_target_specific_complex_evidence_probe.py`
- Formal artifacts: `outputs/tapc/route_a_v25_target_specific_complex_evidence_probe/`
