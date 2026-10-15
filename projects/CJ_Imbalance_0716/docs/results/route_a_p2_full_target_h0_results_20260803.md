# Route A P2: full-target external H0 result

Date: 2026-08-03  
Status: **P2 failed its preregistered all-target H0 gate and is closed.**

## Protocol boundary

P2 was a new external project-local protocol, not a P1 rerun. It retained the
P1 MFT/VFT aperture conditions but instantiated ISRJ with nine replicas and
25--30 us slices, the condition independently supported by earlier H0g/H0h/H0s
frozen aligned-periodicity evidence. It used a new 900-identity seed block and
2,700 exact absent/weak/strong views. All score parameters and controls were
written before source generation; there are no fitted parameters, networks,
calibrators or thresholds.

## Formal H0 results

| Target | true AUROC | control AUROC | delta | weak paired direction | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| MFT | 0.8111 | 0.5459 | +0.2653 | 0.7433 | pass |
| VFT | 0.9983 | 0.5374 | +0.4608 | 0.9967 | pass |
| ISRJ | 0.5551 | 0.4990 | +0.0561 | 0.5133 | fail |

MFT and VFT pass every predeclared condition; their delta-AUROC 95% bootstrap
CIs are `[+0.2361,+0.2947]` and `[+0.4162,+0.5049]`. ISRJ fails absolute weak
ranking (required 0.75), the required +0.10 control advantage, and the 0.65
paired direction threshold. Its delta CI `[+0.0393,+0.0739]` is positive but
too small to satisfy the protocol's absolute-evidence standard.

## Interpretation and stop rule

This result rejects a tempting but invalid inference: H0s's frozen B2
high-replica ISRJ expert does not make an untrained scalar periodicity statistic
automatically transferable to a new multi-pulse compound protocol. That prior
evidence remains a narrow, frozen ISRJ result; it is not a license to claim
shared-protocol physical observability.

P2 is closed. Do not modify its replica count, slice width, lag interval,
block size, block permutation, MFT/VFT settings, score, loss, model,
calibration or threshold, and do not train C0/C3/C4 from P2. No locked test,
test-v2 or test-v3 data were accessed.

## Artifacts

- P2 preregistration: `docs/plans/route_a_p2_full_target_observability_protocol_20260803.md`
- Generator: `experiments/generate_route_a_p2_full_target_pairs.py`
- Fixed H0: `experiments/run_route_a_p2_full_target_h0.py`
- Source receipt: `outputs/tapc/route_a_p2_full_target_generation/formal_receipt_r1.json`
- H0 receipt: `outputs/tapc/route_a_p2_full_target_h0/formal_summary.json`
- Smoke and implementation tests before the formal run: `8 passed`.
