# Route A H1j: composite VFT dechirped-Doppler source validation

## Entry condition

H0i passed the fresh paired low/high-Doppler physical intervention with a
zero-parameter correct-dechirp AUROC of `0.999995` and a phase-permuted
control AUROC of `0.498718`. H1j is therefore authorized as a **separate new
source-level** test, not as a six-label integration.

## Hypothesis

On new six-label composite I/Q, a fixed dechirped-Doppler statistic carries
VFT-specific information beyond a dimension-, energy- and computation-matched
phase-permuted-reference statistic. A matched one-dimensional BCE readout of
the correct statistic must improve weak VFT ranking against
composition-matched VFT-absent negatives without harming strong VFT ranking.

## New source and role protocol

- Fresh balanced train-only source, base seed `800000000`, 63 combinations x
  80 = 5,040 samples. Standard 0716 physical ranges, six labels, white unit
  receiver noise and normal VFT Doppler `[1,7] kHz` apply. The expected
  receiver range is `998677743--998682782`.
- Source audit requires zero intersection with every live earlier train source,
  all labels and metadata, and test-v3 remains sealed.
- Receiver identities are split fit/validation/evaluation by modulo 10 as
  60/20/20. Validation is recorded but unused for selection. Five fixed
  optimizer seeds `2039--2043` differ only in minibatch order; candidate and
  control use identical role records and updates for each seed.
- For each role, VFT positives are compared only with VFT-absent negatives
  sharing the exact other-five-label context. Weakness is evaluation metadata
  derived from measured component power; it never enters fitting, calibration
  or thresholding.

## Matched candidate/control readout

For every record, both arms locate the same maximum-energy 1,000-sample
window and emit one dechirped adjacent-phase-increment scalar. Candidate uses
the known LFM reference; control uses the fixed phase-permuted, unit-modulus
reference. Each scalar is fit-standardized on the same fit identities and
passed through an identically zero-initialized `Linear(1,1)` BCE head for 24
fixed epochs (AdamW `1e-2`, batch 64, no early stopping). At initialization
both heads output exactly zero. No threshold, calibration or fusion is fit.

## Gates

On evaluation identities, candidate minus control must satisfy:

1. weak-VFT versus matched-negative AUROC mean >= `+0.015`, bootstrap 95% CI
   lower bound > 0, and at least 4/5 positive optimizer seeds;
2. weak-VFT AUPR mean is positive, with at least 4/5 positive seeds;
3. no seed has strong-VFT AUROC delta below `-0.010`;
4. source/identity/matched-context/frozen-control checks and sealed-test flags
   pass.

Failure closes the correct-dechirp scalar readout on this composite protocol;
neither LFM phase permutation nor the score can be tuned, widened, fused or
thresholded. Passing is only H1 source evidence. A further fresh H2 six-label
integration, same-capacity physical control, calibration/threshold isolation,
mainstream baselines and external robustness would still be required.
