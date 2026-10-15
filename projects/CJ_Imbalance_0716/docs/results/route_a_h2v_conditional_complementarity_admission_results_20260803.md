# Route A H2v: conditional-complementarity admission result

Date: 2026-08-03  
Status: **admission failed; no correction candidate was trained.**

## Purpose and boundary

H2v did not retry H1u. It pre-registered a fresh 0/4/8 dB MFT visibility
ladder and asked a narrower prerequisite question: before training a
conditional correction, is there stable enough frozen-B0 weak-MFT error mass
for a fixed correct-LFM statistic to complement it?

The source contains 5,040 fresh identities, all 63 full six-label
combinations x 80, and uses one 1 ms PRI with three separated replicas. Its
formal audit passed all shape, source hash, role, receiver-identity,
visibility-ladder, historic-train-disjointness and test-v3-seal contracts.

## B0-only five-seed admission evidence

| Quantity | Result |
|---|---:|
| Early weak-MFT AUROC, mean | 0.87361 |
| Per-seed weak-MFT AUROC | 0.86311, 0.95926, 0.83662, 0.92169, 0.78735 |
| B0 weak-vs-absent inversion rate | 13.69%, 4.07%, 16.34%, 7.83%, 21.26% |
| Union B0 inversion pairs | 80,244 |
| Correct LFM ordering inside those pairs | 88.49% |

The mean weak-MFT-AUROC gate and the physical complementarity gate pass.
However, the preregistered requirement that **every** seed have at least 5%
B0 inversion pairs fails in split/model seed 3102 (4.07%). The study therefore
does not establish stable error mass across seeds.

## Decision

H2v is closed before candidate training. No conditional gate, calibration,
threshold, residual, M1/M2 arm, candidate fusion model or test evaluation was
trained or selected. The 0/4/8 dB source may not be regenerated or reweighted
to rescue the gate.

This negative result is still useful: physical evidence can be strongly
complementary **conditioned on a baseline error set**, yet the size of that
set is source- and seed-dependent. An innovation claim must therefore be
validated on a primary unified six-label protocol with explicit seed-stability
evidence rather than on an MFT-only external admission screen.

No historical locked test, test-v2 or sealed test-v3 data were accessed.

## Artifacts

- [H2v preregistration](../plans/route_a_h2v_conditional_complementarity_admission_20260803.md)
- [Source audit](../../outputs/tapc/route_a_h2v_visibility_ladder_generation/source_audit.json)
- [B0 admission summary](../../outputs/tapc/route_a_h2v_b0_admission/summary.json)
- [B0 run receipt](../../outputs/tapc/route_a_h2v_b0_admission/run_receipt.json)
- [Runner](../../experiments/run_route_a_h2v_b0_admission.py)
