# Route A P1: observability-adequate external physics protocol

Status: preregistered. P1 is a new project-local generator protocol for an
external mechanism-evidence stage. It is not a replacement for the original
0716 single-pulse protocol, does not alter its closed-route decisions, and can
never be used to retune or evaluate test-v3.

## Single falsifiable hypothesis

The original 0716 MFT/VFT weak-evidence failure is driven in part by inadequate
physical aperture: 100 us LFM copies overlap by at least 80% and the 1--7 kHz
VFT range contributes only 0.1--0.7 phase cycles per pulse. If a *new*
counterfactual protocol instead supplies non-overlapping delayed copies and a
coherent multi-pulse Doppler aperture, fixed physics scores should distinguish
weak target-present views from their same-identity, composition-matched target-
absent views more strongly than matched information-destruction controls.

This is a protocol observability claim only. It does not claim that the
original 0716 data contain this information and does not by itself establish a
neural-network improvement.

## P1 generator contract

| Property | P1 value | Difference from original 0716 / failed multi-PRI route |
| --- | --- | --- |
| Sampling | 10 MHz, complex I/Q | preserves the original LFM sampling scale |
| LFM pulse | 100 us / 1,000 samples, 5 MHz sweep | unchanged base waveform |
| Aperture | six uniform-PRI pulses, PRI 500 us | coherent 3 ms observation aperture |
| MFT copies | delays 120, 230 and 340 us | each 100 us copy is separated by at least 110 us; unlike original 30--50 us overlapping copies |
| VFT | 20--60 kHz fixed per identity across all six pulses | 2--6 phase cycles per pulse; unlike original 1--7 kHz range |
| Arms | absent, weak 8 dB, strong 20 dB | same identity/noise/nuisance components within a triplet |
| Labels | FMB, SawFM, SinFM, MFT, VFT, ISRJ | six-label compound background retained |
| Formal count | 900 identities / 2,700 views (300 per target) | sufficient for a no-fit H0 only; no train/test role is generated |
| Formal identity block | base `4200000000` | separate from P1 smoke `4250000000` and prior documented blocks |

For every pair seed, non-target components, their powers, target parameters,
and receiver noise are identical across absent/weak/strong views. Only the
target component and its fixed arm power differ. Pairs cannot be split across
future fit, calibration, threshold, or evaluation roles.

## H0-P1, fixed before source generation

No classifier is trained. H0 computes one target-specific physical score and a
matched control from the generated I/Q:

| Target | true statistic | information-destruction control | fixed direction |
| --- | --- | --- | --- |
| MFT | known-LFM dechirped multi-tone peakiness in the three alias-resolved bands 1.25--1.75, 2.75--3.25 and 3.75--4.25 MHz | identical score with chirp rate `0.83K` | high |
| VFT | six-pulse coherent Doppler maximum over 15--65 kHz | identical statistic with pulse-order permutation `[0,3,1,5,2,4]` | high |
| ISRJ | envelope periodicity peak over lags 50--1,000 samples | same statistic after fixed sample-order permutation | high |

The implementation must not read target delay, Doppler, replica width, arm,
label, power, seed or any existing model logit while calculating a score. Those
fields are used only after scoring to form the predeclared evaluation groups.

For each target's weak-vs-absent comparison, all one-shot conditions are
required: true AUROC >= 0.75; true minus control AUROC >= +0.10 with the
pair-bootstrap 95% CI lower bound > 0; paired weak > absent fraction >= 0.65;
strong AUROC is not lower than weak AUROC by more than 0.02. Any failure closes
P1 for that target; no parameter/JNR/PRI/delay/grid/ROI/score retry is allowed.

## Escalation and evidence boundary

Only if all three H0-P1 target gates pass may a new pre-registered P1 five-seed
equal-capacity C0/C3/C4 experiment be *designed*. That later experiment must
include a physics-score permutation control, raw non-target safety, and fresh
identity-separated roles. It remains external evidence; it cannot authorize
the original test-v3. A failed H0-P1 will be reported as a protocol diagnostic
and closes this exact P1 design.
