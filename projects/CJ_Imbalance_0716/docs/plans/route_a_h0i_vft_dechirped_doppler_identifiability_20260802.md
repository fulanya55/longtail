# Route A H0i: VFT dechirped-Doppler physical identifiability screen

## Hypothesis and non-overlap

The 0716 VFT generator creates a delayed known LFM reference multiplied by a
constant Doppler phasor. Its new physical hypothesis is that, after locating
the active pulse by energy, **known-LFM dechirping followed by adjacent-sample
phase-increment estimation** recovers a controlled Doppler-band intervention.

This is not the stopped STFT-ridge concatenation, low-dimensional RawSource
replacement, phase-reset-window statistic, query pooling, residual, threshold
or loss-weighting mechanism. It is a no-fit H0 screen; it cannot be fused into
H1 or a six-label system regardless of outcome.

The reference-generator contract is the primary factual source: VFT is
`circshift(reference.train, delay) * exp(j 2 pi f_d t)` with a 100-us known
LFM pulse and Doppler as its VFT-specific physical parameter. Literature
retrieval was attempted but the available search endpoint returned HTTP 404;
no unverified literature-specific novelty claim is made in this plan.

## Fresh paired intervention

- Base seed `790000000`; VFT is the only enabled component; 640 train-only
  receiver identities are generated in each arm, with the same manifest,
  component seed and receiver-noise seed for every paired record.
- Low arm: `doppler_Hz=[1000,2500]`. High arm:
  `doppler_Hz=[5500,7000]`. The non-overlapping bands, source count, statistic,
  comparator, seed and gates are fixed before generation.
- The radar, VFT arrival-delay/JNR ranges, reference LFM, total noise power,
  all component seeds and receiver-noise realizations remain the standard
  0716 configuration. Only the VFT Doppler range changes.
- The audit must establish paired receiver/component identities, labels and
  non-Doppler scalar metadata; I/Q must differ; all live historical sources
  must have zero receiver overlap; no test file may be opened.

## Fixed statistic and valid matched control

For each record, energy sliding-window argmax locates the 1,000-sample active
pulse. The selected segment is multiplied by the conjugate known LFM phase;
the score is the angle of its mean adjacent complex product, converted to Hz.
There are no trained parameters, fitted normalizers, thresholds or labels in
the estimator.

The control uses the same location, segment, score operation and unit-modulus
reference, but applies one fixed deterministic permutation to the LFM phase.
It preserves vector length, reference magnitude, output dimension and compute
budget while destroying correct LFM phase cancellation. It is not a branch
permutation or a relabeling control.

## Outcomes and one-shot gates

Evaluate high versus low intervention labels using AUROC, paired high-score
greater-than-low-score rate, and 2,000 fixed-seed paired bootstraps.

1. candidate AUROC >= `0.65`;
2. candidate minus phase-permuted-control AUROC >= `+0.10` and its 95% paired
   bootstrap lower bound is above zero;
3. paired high-greater-than-low rate >= `0.65`;
4. provenance, paired identity, fixed-control and sealed-test checks pass.

Failure closes this exact dechirped-increment statistic and its pulse-window,
permutation and band choices. Passing is only H0 physical identifiability. A
fresh H1 source with composite matched negatives and a new valid control would
still be required before classifier training.
