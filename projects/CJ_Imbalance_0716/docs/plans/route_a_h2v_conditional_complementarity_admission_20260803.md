# Route A H2v: conditional-complementarity admission study

Date: 2026-08-03  
Status: preregistered before H2v source generation.

## Why this is not an H1u retry

H1u is closed: an unconditional three-number MFT residual did not improve a
joint learned classifier on its fixed 8/20 dB source. Its post-hoc diagnostic
showed a distinct fact: the physical residual correctly orders most of the
very small B0 error set. H2v tests a new claim about **where** complementary
evidence may be useful; it does not change any H1u code, data, bands, JNR,
loss, roles, or results.

## Single falsifiable hypothesis

On a once-generated, fresh identity-isolated visibility-ladder source, a
frozen raw-I/Q baseline must have a non-trivial weak-MFT error set before a
conditional evidence model may be trained. If the fixed M1 physics score
orders the baseline's weak-vs-absent inversion pairs better than chance, then
an evidence correction restricted to an uncertainty region can be evaluated
in a separate H2v development stage. This document implements only the
**admission stage**; it does not train or select a candidate correction.

## Fresh source protocol

- New base sample seeds: 3,920,000,000--3,920,005,039; no reuse of H0u,
  H1u, historical training, test-v2, locked test, or sealed test-v3
  identities.
- Six labels and all 63 nonempty combinations x 80 identities are retained.
- One PRI of 1 ms and three physically separated LFM replicas remain fixed.
- MFT-present records comprise 40 strong records at 20 dB and 40 weak
  records drawn deterministically, without post-generation selection, from a
  three-level visibility ladder: 0, 4, and 8 dB in counts 14, 13, 13.
- Non-MFT active components remain at the official 20 dB setting.
- Only a source audit and B0 admission experiment are permitted after source
  generation. No test split is generated.

## Admission experiment

The B0 raw-I/Q compact baseline is trained five times using the H1u backbone,
the same three-stratum MFT loss and 60/20/20 composition-by-MFT-state roles.
No fixed-statistic feature is injected. Checkpoint selection is validation
weak-vs-absent AUROC; early roles are untouched.

H2v proceeds to a separately preregistered correction study only if all hold:

1. mean early weak-MFT AUROC is in [0.65, 0.93];
2. each seed has at least 5% weak-vs-absent B0 pair inversions;
3. the fixed correct-LFM score is above chance on the union of those B0
   inversion pairs, with paired-correct-order rate >= 0.60; and
4. all identity, role and test-v3-seal audits pass.

The interval is an admission definition, not a tuned performance target: its
lower edge prevents a no-signal task; its upper edge prevents a near-ceiling
task such as H1u. If any check fails, H2v closes without training a candidate,
and no source, JNR or threshold may be changed.

## Boundaries

No calibration, threshold selection, historical test, test-v2 or test-v3 may
be used. A pass only authorizes a new, separately preregistered H2v
conditional-correction experiment on a fresh source; it cannot promote C3 or
authorize the final sealed test.
