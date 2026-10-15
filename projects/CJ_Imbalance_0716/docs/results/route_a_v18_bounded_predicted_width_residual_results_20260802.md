# Route A v18: bounded predicted-width residual result

## Decision

**Stop this one-dimensional bounded-width residual mechanism.** It passed all
frozen-contract and non-target-label safety checks, but failed both primary
pre-registered efficacy gates on fresh identity-isolated evaluation data. The
gate is not relaxed after observing these results.

## Protocol and evidence boundary

v18 used a newly generated, train-role-only source (base seed `540000000`):
5,040 unique receiver identities, ISRJ width `[10,30] us`, and replica count
`[3,4]`. Identity roles were fixed before inference: fit 3,024, selection
1,008, evaluation 1,008; the evaluation role contains 256 `VFT=0, ISRJ=1`
hard negatives. The data audit SHA-256 is
`1a87562963ee3e0504afc890543538cec8e23474ca2ba8a8f90727a61a5043a5`.

For every frozen Q4 seed, the matching frozen v16 width probe produced
`w_hat`. The candidate changed only the VFT logit:

`z_vft' = z_vft - c * clip((w_hat - 20 us) / 10 us, 0, 1) * p_isrj`.

The `20 us` centre and `10 us` scale come from v16's width support. A
same-capacity control used `z_vft' = z_vft - c * p_isrj`. Both choose one
scalar `c` only in the fit role over `{0.0, 0.1, ..., 1.0}`, maximizing hard
FPR reduction subject to a VFT-positive recall loss of at most 0.010. Q4
checkpoints, calibration, thresholds, all non-VFT logits, and threshold vectors
were exact and frozen. No locked test, test-v2, test-v3, or v18 test role was
used.

## Held-out five-seed result

| Comparison on the evaluation role | Mean delta | Bootstrap 95% CI | Per-seed direction | Gate |
| --- | ---: | ---: | ---: | --- |
| Candidate minus Q4 hard-negative VFT FPR | -0.0211 | [-0.0250, -0.0180] | 5/5 lower | required <= -0.030: fail |
| Candidate minus ISRJ-only-control hard-negative VFT FPR | -0.0063 | [-0.0141, +0.0016] | 4/5 lower | required <= -0.010: fail |
| Candidate minus Q4 VFT positive recall | -0.0070 | [-0.0078, -0.0055] | all >= -0.010 | safety: pass |
| Candidate minus Q4 VFT AUPR | +0.0053 | [+0.0037, +0.0067] | 5/5 higher | safety: pass |
| Candidate minus Q4 ISRJ AUPR | 0.0000 | [0, 0] | unchanged | safety: pass |
| Candidate minus Q4 SawFM AUPR / weak recall | 0.0000 / 0.0000 | [0,0] / [0,0] | unchanged | safety: pass |

## Failure mechanism

The v17 result correctly identified a risk-associated physical axis, but v18
shows that a *single monotonic scalar* is not sufficient to yield the required
operational improvement.

- The candidate is more selective than the ISRJ-only gate, and does preserve
  more VFT positive recall. Its extra FPR reduction is nevertheless only
  0.0063 on average and is not stable against the same-capacity control.
- The safety constraint binds the scalar. Stronger downward shifts can reduce
  hard FPR, but reduce VFT positive and weak-positive recall; fit-selected
  candidate coefficients were bounded by that constraint.
- Predicted width explains an important risk *ordering*, not the whole false
  positive mechanism. Samples with comparable width can differ in VFT score
  because of waveform morphology, co-occurring components, and the frozen
  expert's score geometry. A global width correction cannot separate those
  cases.

Therefore this does not invalidate v16/v17 physical evidence. It rejects the
specific claim that predicted width alone, through a single bounded residual,
is enough to improve Q4 beyond a non-width ISRJ gate under the specified recall
budget.

## Read-only residual-error audit

This post-run diagnosis used only the saved v18 evaluation predictions; it did
not alter `c`, the grid, or the gate. Across seeds, the candidate corrected
only 4--7 of 256 hard negatives, while 43--65 VFT false positives remained.
Corrected cases had Q4 VFT probabilities around `0.50--0.57`; remaining false
positives were much more confident (`0.65--0.74`). Their predicted widths and
width evidence overlapped substantially with corrected cases. The observed
ceiling is therefore a score-margin problem: the bounded scalar can move
borderline false positives below the fixed threshold, but cannot safely move
the higher-confidence false positives without violating VFT recall safety.

## Next admissible direction

Do not add another width-bin, coefficient, or threshold sweep to v18. A new
mechanism must supply an independently measurable second condition with a
same-capacity control and a fresh development source. A read-only check of the
already existing linear STFT-ridge score found hard-negative Q4-probability
Spearman between `-0.062` and `+0.087`, and FP-vs-hard-non-FP pairwise ordering
between `0.475` and `0.568` across five seeds. Thus simply reusing that linear
score as another residual term is rejected. The next hypothesis must test a
new hard-negative-conditional, nonlinear time-frequency morphology expert
before it can be attached to Q4. test-v3 remains sealed.

## Artifacts

- Plan: `docs/plans/route_a_v18_bounded_predicted_width_residual_plan_20260802.md`
- Generator/audit/runner: `experiments/generate_route_a_v18_bounded_predicted_width_residual.m`, `experiments/audit_route_a_v18_bounded_predicted_width_residual.py`, and `experiments/run_route_a_v18_bounded_predicted_width_residual.py`
- Formal output, selected coefficients, grids, predictions and receipt: `outputs/tapc/route_a_v18_bounded_predicted_width_residual/`
- Smoke is separate and non-evidential: `outputs/tapc/route_a_v18_bounded_predicted_width_residual_smoke/`
