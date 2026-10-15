# Route A H0u: separated-delay MFT physical-identifiability preregistration

Date: 2026-08-03  
Status: preregistered before source generation, scoring, model training,
calibration, threshold selection, or test access.

## Single falsifiable hypothesis

The original 0716 MFT contract places 3--8 replicas in a 30--50 us relative
delay interval while the LFM pulse width is 100 us. Its replicas consequently
overlap strongly in the raw observation. In a **new external sensing
contract** with a 1 ms PRI and three weak MFT replicas in three fixed,
non-overlapping delay bands, the correct-LFM separated-peak statistic should
separate MFT-present from composition-matched MFT-absent records better than a
same-amplitude, phase-order-permuted reference statistic.

This asks only whether a new delay-aperture observation contains target
attributable information. It does not repair original single-pulse 0716,
does not train a neural network, and cannot authorize test-v3.

## New source and counterfactual contract

- Generator: `experiments/generate_route_a_h0u_mft_delay_aperture_composite.m`.
- New fresh source: 640 paired identities, sample seeds
  `3900000000--3900000639`; receiver-noise IDs use the official deterministic
  mapping and must be checked against every readable historic train source
  before generation and again after generation.
- Radar: 10 MHz sampling, 100 us / 1000-sample LFM, 5 MHz bandwidth, one
  1 ms PRI / 10,000 samples, unit-power proper-complex receiver noise.
- Both arms contain exactly FMB, SawFM, SinFM, VFT and ISRJ from the official
  configurable generator at common 20 dB component JNR, sharing all component
  seeds, sample seeds and receiver noise. No MFT signal is generated through
  the ordinary narrow-delay branch.
- The present arm additionally has one custom, normalized 3-replica LFM MFT
  component at total 8 dB JNR (12 dB below the common background component
  level). Replica starts are independently jittered by at most 20 us around
  200, 400 and 600 us. Thus every pairwise separation is at least 160 us,
  greater than the 100 us pulse width. The absent arm adds nothing.
- The present/absent arms therefore differ only in that declared MFT component
  and its I/Q contribution. Both remain train-only H0 artefacts; no test split
  is created.

This is deliberately a new physical protocol, not an extra model capacity or
an ROI retry on H0j/H1k/H1m/H0o. It may only produce an external physical H0
statement for its own delay-aperture regime.

## Fixed zero-fit readout and control

For each RMS-normalized record, circularly correlate the complete 1 ms record
with the known 100 us LFM. Retain three pre-specified start-delay bands
`[160,300)`, `[350,500)`, `[540,700)` us and average the natural logarithm of
the maximum magnitude in each band. The bands cover the allowed jittered
replicas and are separated by exclusion gaps.

The control uses the identical unit-modulus reference with a fixed
permutation of its phase order. It has equal length and amplitude but destroys
the LFM chirp relation. It uses exactly the same bands, score formula, source
and no fitted parameter. There is no learned parameter, normalization fit,
classifier head, calibration, threshold, fusion or F1 endpoint.

## Endpoints and gates

Primary endpoint: MFT-present versus MFT-absent AUROC of the correct-LFM
statistic. Primary attribution endpoint: correct-LFM AUROC minus permuted-LFM
AUROC, with a pair-preserving 5,000-draw bootstrap interval. Supporting
endpoints are AUPR differences and the paired present-greater-than-absent rate.

H0u passes only if all of the following hold:

1. correct-LFM AUROC >= 0.70;
2. correct-minus-permuted AUROC >= +0.10 and its paired 95% CI lower bound > 0;
3. correct-minus-permuted AUPR >= +0.08 and its paired 95% CI lower bound > 0;
4. correct score is greater on >= 0.65 of paired present/absent identities;
5. source identity, metadata, non-target background and minimum-separation
   contracts all pass; and
6. `locked_test_used=false`, `test_v2_used=false`, `test_v3_used=false`.

## Stop and progression rule

Any failed gate stops H0u permanently: no delay-band, gap, jitter, JNR,
replica count, score aggregation, reference, source-size, model, loss,
calibration or threshold search is allowed on these identities. A pass permits
only a separately preregistered fresh-source composite H1 with an
equal-parameter physical-information-destroying control. It still does not
permit fusion with original development-v3 or test-v3.

## Required artefacts

Save dry-run and formal generation receipts, fresh-ID selection, source SHA-256,
metadata/identity audit, zero-fit summary, score output and this plan hash.
The source audit must verify all declarations before the scorer reads I/Q.
