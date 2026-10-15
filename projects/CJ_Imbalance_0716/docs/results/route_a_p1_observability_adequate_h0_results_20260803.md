# Route A P1: observability-adequate external H0 result

Date: 2026-08-03  
Status: **P1 does not pass its all-target gate and cannot enter model training.**

## Scope and boundary

P1 is a new project-local external physics protocol, not an original-0716
result. It retains 10 MHz complex I/Q and 100 us LFM pulses, but supplies six
uniform 500 us PRI pulses, non-overlapping MFT copies at 120/230/340 us, and
20--60 kHz VFT. The source contains 900 new identities and 2,700 exact
absent/weak/strong views (300 identities per MFT/VFT/ISRJ target). Within each
identity, nuisance labels, their powers, sample seed and receiver noise are
identical; only the target arm changes from absent to 8 dB weak to 20 dB
strong.

This H0 has zero fitted parameters. It evaluates three frozen target-specific
physical scores against information-destruction controls. It neither trains a
model nor calibrates/selects a threshold. `locked_test_used`, `test_v2_used`,
and `test_v3_used` are all false.

## Fixed H0 results

| Target | true AUROC | control AUROC | delta | weak paired direction | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| MFT | 0.8027 | 0.5501 | +0.2526 | 0.7000 | pass |
| VFT | 0.9981 | 0.5592 | +0.4390 | 1.0000 | pass |
| ISRJ | 0.5775 | 0.4953 | +0.0822 | 0.4400 | fail |

The MFT delta-AUROC bootstrap 95% CI is `[+0.2272,+0.2784]`; VFT's is
`[+0.3931,+0.4834]`. Both also pass the predeclared AUROC >= 0.75 and strong-
positive safety gates. They establish only that the P1 physical conditions
make their fixed target statistics observable beyond their matched controls.

ISRJ fails three fixed checks: true AUROC is below 0.75, delta is below +0.10,
and only 44% of matched weak views exceed their absent counterpart. Its delta
CI `[+0.0367,+0.1277]` being positive cannot repair the absolute weak-ranking
or paired-direction failure.

## Decision

The P1 all-target H0 gate is false. Per the P1 preregistration, do **not**
train C0/C3/C4, tune a score, change a JNR/delay/PRI/grid/ROI, add a network
branch, or use P1 to support test-v3. The MFT/VFT passes are retained as
narrow external physical-observability evidence; they do not license a
six-label VTSE-VBF efficacy or causality claim because the full shared protocol
failed for ISRJ.

The next valid research question, if pursued, must be separately pre-registered
and must rely on the already documented ISRJ aligned-periodicity physical
condition rather than retuning P1's failed generic envelope-autocorrelation
score. It must use a new identity block and make clear that it is a new
protocol, not a P1 rerun.

## Artifacts

- P1 protocol: `docs/plans/route_a_p1_observability_adequate_protocol_20260803.md`
- Generator: `experiments/generate_route_a_p1_observability_adequate_pairs.py`
- H0 implementation: `experiments/run_route_a_p1_observability_adequate_h0.py`
- Formal source receipt: `outputs/tapc/route_a_p1_observability_adequate_generation/formal_receipt_r1.json`
- Formal H0 receipt: `outputs/tapc/route_a_p1_observability_adequate_h0/formal_summary.json`
- Implementation test suite before formal H0: `7 passed`.
