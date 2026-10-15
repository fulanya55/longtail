# Route A H0u: separated-delay MFT physical-identifiability result

Date: 2026-08-03  
Decision: **passed H0; eligible only for a separately preregistered, fresh-source H1.**

## Scope and boundary

H0u is a zero-fit physical screen on a new external 1 ms delay-aperture
protocol. It is not an original-0716 development result, a six-label unified
model, a calibration/threshold result, or a test-v3 result.

The source contains 640 present/absent counterfactual pairs. Each pair shares
FMB, SawFM, SinFM, VFT, ISRJ, all background substreams and receiver noise;
the present arm alone adds a total-8-dB MFT component with three full LFM
replicas in delay bands centered at 200, 400 and 600 us. The observed minimum
inter-replica spacing is 1608 samples / 160.8 us, greater than the 100 us
LFM pulse width. The source uses receiver identities not present in any
readable historical train source.

## Fixed readout result

The candidate is the mean log peak of a correct-LFM circular matched-filter
response in three pre-registered, separated delay bands. The control uses the
same unit-amplitude LFM reference with a fixed phase-order permutation and the
same bands and aggregation. Both have zero trained parameters.

| Endpoint | Correct LFM | Phase-order control | Difference |
| --- | ---: | ---: | ---: |
| MFT present/absent AUROC | 0.99786 | 0.59005 | +0.40781 |
| MFT present/absent AUPR | 0.99720 | 0.57128 | +0.42592 |
| AUROC paired bootstrap 95% CI | -- | -- | [+0.39555, +0.41967] |
| AUPR paired bootstrap 95% CI | -- | -- | [+0.41141, +0.43553] |
| Correct-score present > absent fraction | -- | -- | 1.00000 |

All nine H0 gates pass, including identity/source contracts and sealed-test
checks. `locked_test_used=false`, `test_v2_used=false`, and
`test_v3_used=false`.

## What this proves and does not prove

It proves that, when a 1 ms observation supplies three non-overlapping weak
MFT LFM replicas, their separated matched-filter evidence is physically
attributable beyond a phase-order control. This is a distinct observation
protocol from the original 100 us/300 us single-PRI 0716 contract and from the
closed H0j/H1k/H1m/H0o readouts.

It does not prove a neural head can use this information, that it survives a
six-label learning setting, that it improves MFT weak recognition safely, or
that it generalizes to the original 0716 data. Those are the H1 obligations.
No calibration, threshold, fusion, C3 promotion or test-v3 action is allowed
from H0u alone.

## Artefacts

- Pre-registration: `docs/plans/route_a_h0u_mft_delay_aperture_identifiability_20260803.md`
- Source receipt: `outputs/tapc/route_a_h0u_mft_delay_aperture_generation/formal_generation_receipt.json`
- Identity/source audit: `outputs/tapc/route_a_h0u_mft_delay_aperture_generation/paired_identity_audit.json`
- Machine summary: `outputs/tapc/route_a_h0u_mft_delay_aperture/summary.json`
