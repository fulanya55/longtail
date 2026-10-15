# Route A H0g: Frozen ISRJ parameter-OOD-v2 replication gate

## Purpose and scope

H0g is an **external mechanism replication**, not a new model-selection
experiment. The strongest remaining Route-A evidence is that the low-capacity
aligned-periodicity score improves ISRJ when attached to the frozen H1
classifier (B2). It already replicated on development-v7 and on one
parameter-OOD source, but the latter covers only ISRJ width 22--30 us and
replicas 6--8. H0g asks a single falsifiable question:

> With all learned artifacts and decisions frozen, does B2 still improve ISRJ
> weak evidence over its exact B0 parent under an identity-isolated,
> previously untested **high-replica** ISRJ regime (25--30 us; exactly 9
> replicas)?

It does **not** test the stopped complete VFT/MFT unified candidate, does not
repair VFT OOD, and does not select a new architecture. A failure is useful
boundary evidence and cannot trigger retuning.

## Data contract

- Generator: the established six-class closed-set generator, train split only.
- New base seed: `760000000`; expected 5,040 samples and unique
  `receiver_noise_seed` range `958677743--958682782`.
- Each of 63 label combinations has 80 samples. ISRJ width is sampled in
  `[25e-6, 30e-6]` seconds and replica count is fixed at `9`.
- The ordinary development range is width 5--20 us / replicas 1--5; OOD-v1
  used width 22--30 us / replicas 6--8. H0g therefore makes a feasible
  one-coordinate extrapolation in replica count while retaining a high-width
  regime. A dry-run rejected the originally proposed 31--36 us / 9--11 range
  because the generator correctly enforces `(replicas+1)*width <= 3000`
  samples; it was not generated or evaluated.
- Before evaluation, a train-only audit must verify source SHA-256, 5,040
  unique receiver identities, active finite ISRJ parameter ranges, and zero
  receiver intersection with all live historical training MATs plus preserved
  H0e/H0f recorded ranges. It may read only the test-v3 seal manifest, whose
  `access_authorized`, `analysis_allowed` and `used` values must remain false.

## Frozen arms

For each fixed training seed 2039--2043, evaluate the exact saved artifacts
used by `RouteA_ISRJ_ParameterOOD_v1_Frozen`:

| Arm | Difference |
| --- | --- |
| B0_H1_Frozen | Focal-36 encoder, frozen linear H1 head, RawSaw SawFM route, saved B0 calibration and thresholds |
| B2_H1AlignedPeriodicity_Frozen | B0 plus the saved aligned 310-D periodicity expert and saved B2 low-capacity fusion head; saved B2 calibration and thresholds |

No checkpoint, normalization, expert, fusion model, Platt parameter, threshold,
data-role index, loss, optimizer, training epoch or seed is fitted or selected
on H0g. Inference uses the one new view only. B0/B2 source checksums are
recorded per seed.

## Metrics and pre-registered decision

Primary paired endpoint is B2 minus B0 ISRJ weak-positive-versus-negative
AUROC. Supporting endpoints are ISRJ AUPR, weak recall, strong recall, FPR,
six-label macro-AUPR/macro-F1, and exact SawFM preservation.

H0g passes only if all hold:

1. mean ISRJ AUPR delta > 0 and mean weak-AUROC delta > 0;
2. mean ISRJ weak-recall delta >= +0.030;
3. weak-recall and weak-AUROC deltas are positive in at least 4/5 seeds;
4. B2 mean ISRJ FPR <= 0.25 and macro-AUPR delta is non-negative;
5. RawSaw SawFM AUPR and weak recall are exactly unchanged;
6. all data, artifact and test-seal contracts pass.

Pass supports a narrowly worded third external replication of the **ISRJ
periodicity mechanism**. It does not promote any full six-label candidate or
authorize test-v3. Failure freezes the H0g source and forbids any adjustment of
periodicity features, fusion coefficients, calibration, thresholds, OOD ranges
or data identities; the evidence claim stays limited to the prior supported
ranges.

## Artifacts

The experiment must preserve the generator receipt, formal source SHA and
identity audit, five seed summaries, per-seed B0/B2 predictions and metrics,
aggregate summary, model/calibration/threshold hashes, and a result report.
Every receipt must include `locked_test_used=false`, `test_v2_used=false` and
`test_v3_used=false`.
