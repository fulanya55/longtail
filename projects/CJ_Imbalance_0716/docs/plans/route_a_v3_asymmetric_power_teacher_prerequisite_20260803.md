# Route A H0t: asymmetric high-to-low power posterior-transfer prerequisite

## Question and evidence boundary

This H0t is a **frozen-inference feasibility audit**, not a new neural model.
It asks a single falsifiable question: do the existing five Focal-36 models,
trained at the 30 dB development-v3 reference condition, provide substantially
more reliable ranking evidence for MFT, VFT and ISRJ at 50 dB than at the
matched 10 dB view of the same receiver identity?

The candidate mechanism, if admitted later, is asymmetric: a fixed
high-visibility teacher may supply a stop-gradient soft target to a
low-visibility student while supervised labels remain active.  It is neither
hard pseudo-labeling nor high/low feature invariance, both of which are
historically closed as primary mechanisms.  H0t contains no student, no loss,
no gate, no temperature selection, no calibration and no threshold selection.

## Frozen inputs

- Development-v3 base MAT SHA-256: `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`.
- Same-identity train-only power views: level 01 (10 dB) and level 09 (50 dB)
  from `power_sweep_9level_v3`.  The manifest fixes all three target labels
  MFT/VFT/ISRJ to each sweep level, while the reference condition is 30 dB.
- Five existing, frozen `E2_Focal_36` checkpoints for seeds 2039--2043,
  reconstructed with their matching F2 initialization.
- Only each seed's pre-existing `early` development role (2,520 identities) is
  forwarded.  The views must preserve identity order and labels exactly.

The script rejects any test path, verifies the sealed test-v3 fields only, and
records all input/checkpoint/role hashes.  It must not open test-v2, a historic
locked test, or test-v3 I/Q/labels.

## Fixed endpoints and admission rule

For each target and seed, report AUROC and AUPR at 50 dB and 10 dB, their
paired difference, fixed top-decile precision lift at 50 dB, and descriptive
same-positive score changes.  The top-decile quantity is a fixed ranking probe,
not a deployment threshold.  The `0.75/0.50` rescue count is descriptive only.

The candidate is admitted only if all three targets satisfy all of the
following across five seeds:

1. mean 50 dB AUROC >= 0.60;
2. mean (50 dB minus 10 dB) AUROC >= +0.030;
3. at least 4/5 seed AUROC differences are positive;
4. mean 50 dB top-decile precision lift over prevalence >= 1.50.

Failure closes asymmetric posterior transfer on these views: no adjustment of
teacher confidence, temperature, divergence, loss weight, pairing level or
threshold is allowed.  A pass is only a prerequisite.  It authorizes design
of a **fresh-identity** H1 source and a new preregistered model comparison;
it does not establish a six-label improvement or authorize test-v3.

## Outputs

`outputs/tapc/route_a_v3_asymmetric_power_teacher_prerequisite/` contains the
per-seed target logits, hash receipts and immutable summary.  Every receipt
records `locked_test_used=false`, `test_v2_used=false`, and `test_v3_used=false`.
