# Route A H0h: Frozen ISRJ periodicity under AR(1) receiver-noise stress

## Single falsifiable hypothesis

The aligned-periodicity increment is not an artifact of independent white
receiver noise. If its ISRJ evidence is structural, then the fully frozen B2
model should improve over its exact frozen B0 parent when the *only* changed
data-generating factor is a proper-complex, unit-power circular AR(1) receiver
noise process.

This is an external-replication question. It is not a new VTSE architecture,
a noise-augmentation training study, a threshold adjustment, or a retry of
any stopped MFT/VFT counterfactual mechanism.

## Fresh source and noise wrapper

- Generate a six-label, balanced, train-only white-noise base source with
  base seed `770000000`, 5,040 samples (63 combinations x 80), and no test
  split. Expected receiver identities are `968677743--968682782`.
- Preserve the reference generator output unchanged as an upstream artifact.
  A project-local wrapper reproduces the default white noise from each saved
  `receiver_noise_seed`, subtracts it exactly, and adds AR(1) noise:

  `n[0]=e[0]`, `n[t]=0.80*n[t-1]+sqrt(1-0.80^2)*e[t]`,
  `e ~ CN(0,1)`, `E|n|^2=1`.

  The output is multiplied by `sqrt(noise_power)`, so the total receiver
  noise-power convention remains one. Components, labels, parameter values,
  parameter masks, sample seeds, receiver seeds and combination metadata are
  copied unchanged. The reference generator is read-only and is not edited.
- Before use, the wrapper must prove deterministic equality for repeated
  seed calls, zero pseudo-covariance within numerical tolerance, unit-power
  normalization and a positive lag-one complex correlation near `0.80` in a
  large deterministic probe. The formal receipt records both source hashes,
  transform hash, AR coefficient, sample count and reconstruction error.
- A train-only audit checks the expected identity range/count, zero receiver
  intersections with live historical train data and recorded H0e/H0f ranges,
  exact equality of all non-I/Q data fields between base and AR source, source
  hash provenance, the sealed test-v3 manifest, and the frozen noise receipt.

## Frozen comparison

Evaluate seeds 2039--2043 using exactly the saved B0_H1 and
B2_H1AlignedPeriodicity artifacts used in H0g: Focal-36 backbone, H1 head,
RawSaw, periodicity expert, B2 fusion, calibrations and thresholds. No
training, normalization refit, calibration, threshold choice, score fitting,
or artifact selection is permitted on the AR source.

## Outcomes and gates

Primary endpoint: paired B2-minus-B0 ISRJ weak-positive-versus-negative
AUROC. Supporting endpoints: ISRJ AUPR, weak/strong recall, FPR, macro-AUPR,
macro-F1, per-label metrics, receiver-cohort strata and exact SawFM outputs.

All gates must pass:

1. mean ISRJ AUPR and weak-AUROC deltas are positive;
2. mean ISRJ weak-recall delta is at least `+0.030`;
3. weak-recall and weak-AUROC deltas are positive in at least 4/5 seeds;
4. B2 mean ISRJ FPR is at most `0.25`, macro-AUPR delta is non-negative, and
   all-label / receiver-cohort FPR extrema are reported without selection;
5. SawFM AUPR and weak recall are exactly unchanged;
6. data provenance, frozen artifact hashes and test-v3 seal checks pass.

Pass supports a bounded noise-shift replication of the ISRJ periodicity
mechanism. It does not promote the stopped complete VFT/MFT candidate or
authorize test-v3. Failure freezes coefficient `0.80`, this source, noise
wrapper and all B0/B2 decisions: no AR coefficient, source, threshold,
calibration, fusion or periodicity-feature search is permitted.

## Non-negotiable boundaries

Only the new train-only base and derived AR MAT may be opened. Historical
locked test, test-v2 and sealed test-v3 must not be opened, analyzed or used.
All receipts include `locked_test_used=false`, `test_v2_used=false` and
`test_v3_used=false`.
