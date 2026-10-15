# Route A H0i: VFT dechirped-Doppler physical identifiability result

## Decision

**H0i passes as a no-fit physical identifiability screen.** In a fresh paired
VFT-only intervention, correct known-LFM dechirping followed by an
adjacent-sample phase increment recovers the low/high Doppler intervention;
the dimension-, energy- and computation-matched phase-permuted-reference
control does not. This is not a classifier, a composite-jamming recognition
result, a fusion result, or a test-v3 result.

## Fresh paired intervention

- New train-only low/high sources each contain 640 VFT-only records, with the
  same 640 unique receiver identities (`988677743--988678382`), component
  seeds, labels, masks, JNR and arrival-delay values in the two arms.
- The sole intervention is VFT Doppler (and its deterministic velocity
  derivative): low `1003.89--2491.25 Hz`; high `5503.89--6991.25 Hz`.
- Both source hashes, pairing, VFT-only labels, unaltered non-intervened
  parameters and fixed-column I/Q change are audited. Every live historical
  train source has zero receiver overlap. All test flags, including test-v3,
  are false.

## Fixed statistic and control

The zero-parameter candidate selects the highest-energy 1,000-sample pulse,
removes the known 0716 LFM phase, and converts the mean adjacent complex phase
increment into Hz. It has no labels, fitted normalizer, threshold or learned
weights.

The control performs exactly the same pulse selection and score operation but
uses one deterministic permutation of the unit-modulus LFM phase. Thus it
preserves reference magnitude, dimension and compute while destroying the
coherent dechirping relation; it is not a branch-index or relabeling control.

| Endpoint | Result | Gate |
| --- | ---: | --- |
| Candidate low/high AUROC | 0.999995 | >= 0.65 |
| Phase-permuted control AUROC | 0.498718 | descriptive control |
| Candidate minus control AUROC | +0.501277 | >= +0.10 |
| Paired bootstrap 95% CI of difference | [+0.473187, +0.529335] | lower > 0 |
| Paired high-score > low-score rate | 1.0000 | >= 0.65 |

All preregistered H0i gates pass. Candidate mean estimates are `1773.99 Hz`
(low) and `6271.74 Hz` (high), close to their imposed bands.

## Boundary and next gate

H0i establishes that an untested, generator-grounded VFT physical cue exists
outside the stopped STFT-concatenation, phase-reset-window, query, residual
and loss-weighting routes. It says nothing yet about weak VFT recognition
inside compound interference. The only admissible next experiment is H1 on a
**new** composite, identity-isolated source: a same-capacity source classifier
using the correct versus phase-permuted dechirped statistic must improve weak
VFT versus composition-matched VFT-absent negatives without a strong-positive
safety loss. No six-label integration is authorized before that result.

Artifacts:

- `docs/plans/route_a_h0i_vft_dechirped_doppler_identifiability_20260802.md`
- `outputs/tapc/route_a_h0i_vft_dechirped_doppler_generation/paired_identity_audit.json`
- `outputs/tapc/route_a_h0i_vft_dechirped_doppler/route_a_h0i_vft_dechirped_doppler_summary.json`
