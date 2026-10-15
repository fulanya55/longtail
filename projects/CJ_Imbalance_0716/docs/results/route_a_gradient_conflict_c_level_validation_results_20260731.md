# Route A SawFM gradient-conflict innovation: C-level validation result

Date: 2026-07-31

## 1. Current conclusion

`GradientContrastFixed1p5` has passed the preregistered one-time test-v2 gate and
therefore reaches **C-level independent validation**.

The validated innovation is:

> Keep every SawFM-containing compound-jamming sample, isolate the SawFM-driven
> shared-representation update through a paired C6/M5 contrast direction, and
> use the frozen extrapolation `theta_C6 + 1.5*(theta_M5-theta_C6)` to improve the
> other five labels under a target-label FPR cap.

The result does not validate the new instantaneous LSCP projection candidate;
LSCP is retained as a negative development-v3 result.  The successful candidate
was frozen before test-v2, used no test fitting or selection, and may not be
rerun or repaired from test outcomes.

## 2. Most recent three rounds

| Round | Evidence | Candidate and hypothesis | Main result | Gate |
|---|---|---|---|---|
| development-v2 gradient contrast | development-v2, repeatedly developed | Extrapolate the paired C6-to-M5 update direction | macro-F1 `+0.01233`, target weak recall `+0.02597` | Failed 1/8: weak recall below `+0.030` |
| development-v3 replication and LSCP | fresh identity-isolated development-v3 | Replicate fixed alpha=1.5 and test label-selective instantaneous projection | fixed alpha: macro-F1 `+0.01337`, weak recall `+0.03283`; LSCP weak recall only `+0.00360` | Fixed contrast passed 10/10; LSCP failed |
| one-time test-v2 | independent frozen test-v2 | Apply frozen models, temperatures and thresholds without fitting | macro-F1 `+0.01390`, weak recall `+0.03048`, 5/5 positive seeds | Passed 10/10, C level |

All three rounds recorded `locked_test_used=false`.  test-v2 was used only in
the final one-time round.

## 3. Success reasons

1. The training data were not reduced to samples without SawFM.  SawFM-present
   compound backgrounds remained in every C6 and M5 fit role.
2. C6 and M5 shared initialization, long-tail manifest, minibatch order,
   optimization budget, calibration roles and threshold rules.
3. Frozen-head H5 had already shown that deleting an independent output row is
   ineffective; the useful direction appears only when a shared temporal block
   is allowed to adapt.
4. The C6-to-M5 paired update direction was stable: alpha=1.5 was selected in all
   development-v2 seeds and was then frozen, not reselected, on development-v3.
5. development-v3 used 25,200 new receiver-noise identities.  Its receiver seed
   interval `[222677743,222702942]` was disjoint from development-v2 train and
   test-v2.
6. The development-v3 effect replicated the earlier direction and crossed the
   frozen weak-recall gate before test-v2 access.
7. Test-v2 used frozen temperature and thresholds from development-v3; no
   optimizer, calibration fit, threshold search or candidate selection touched
   test-v2.

## 4. Failure analysis retained

### Frozen H5 output deletion

With F2 frozen, common BCE rows are independent.  H5-minus-C6 was effectively
zero, so output competition was rejected as the mechanism.

### Plain M5 shared-gradient isolation

The direction was consistent, but ordinary M5 on development-v2 did not reach
the required effect size.  It established A-level mechanism evidence rather
than a promotable method.

### LSCP_TargetProjected

LSCP successfully modified the intended gradient geometry on development-v3:

- SawFM-to-MFT mean cosine: `0.03609 -> 0.06526`;
- SawFM-to-ISRJ mean cosine: `-0.00713 -> 0.05602`;
- SawFM-to-VFT mean cosine: `0.05230 -> 0.06899`;
- removed SawFM gradient norm fraction: `0.09289`;
- non-finite states: `0`.

However, its downstream effect was too small:

- macro-AUPR delta: `-0.000043`;
- macro-F1 delta: `+0.000468`;
- target weak-recall delta: `+0.003598`.

The failure shows that correcting instantaneous batch-level cosine is not
sufficient.  Likely factors include high minibatch variance, interaction with
AdamW momentum, projection only at the last block, and the fact that a zero or
positive local cosine does not guarantee a better long-horizon representation.
No LSCP hyperparameter was tuned from early-stop or test-v2.

## 5. Innovation evidence level

| Level | Requirement | Status |
|---|---|---|
| A: mechanism | Shared SawFM isolation produces consistent multi-seed improvements | Passed |
| B: development | All fixed gates pass on fresh identity-isolated development-v3 | Passed by `GradientContrastFixed1p5` |
| C: independent | Frozen candidate passes one-time test-v2 without fitting/selection | Passed |

The strongest permitted wording is now:

> The paired SawFM-gradient contrast method obtained independent validation on
> test-v2 for improving common-five compound-jamming recognition under the
> preregistered target-label FPR constraint.

It is not permitted to claim that LSCP itself was validated.

## 6. Final preregistered candidate

- Baseline: `C6_Joint`.
- Supporting endpoint: `M5_GradientIsolated`.
- Candidate: `GradientContrastFixed1p5`.
- Formula: `theta_C6 + 1.5*(theta_M5-theta_C6)`.
- Trainable development scope: final shared temporal multi-scale block plus
  six-output classifier.
- Frozen inference scope: full F2 model, common-five calibration temperature and
  common-five thresholds for each seed.
- Seeds: 2039-2043.
- Test size: 25,200 samples.
- Test SHA256:
  `902e636911e76d6915fe04976f59cc46a588b3d98926bb9a48e2200628617947`.

## 7. Implementation and verification

- development-v3 generator uses base seed `24000000` and train-only generation.
- development-v3 file SHA256:
  `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`.
- LSCP and evidence-boundary focused tests: passed.
- Frozen-test focused suite: `11 passed`.
- Frozen bundle: 35 hashed artifacts.
- Preflight: ten model contracts and all frozen hashes passed without opening
  test-v2.
- Access marker was written before test-v2 hashing and reading.

## 8. One-time test-v2 results

| Metric | C6_Joint | GradientContrastFixed1p5 | Paired delta |
|---|---:|---:|---:|
| common-five macro-AUPR | 0.86724 | 0.87372 | **+0.00648** |
| common-five macro-F1 | 0.75332 | 0.76722 | **+0.01390** |
| MFT/VFT/ISRJ mean weak recall | 0.29495 | 0.32543 | **+0.03048** |
| target mean negative FPR | 0.19338 | 0.19541 | +0.00204 |
| SawFM-present macro-F1 | 0.70647 | 0.72607 | **+0.01959** |
| SawFM-absent macro-F1 | 0.80298 | 0.81123 | **+0.00824** |

Target-label AUPR deltas:

- MFT: `+0.00763`;
- VFT: `+0.01215`;
- ISRJ: `+0.00676`.

Target weak-recall delta by seed:

| Seed | Delta |
|---:|---:|
| 2039 | +0.03427 |
| 2040 | +0.02678 |
| 2041 | +0.03774 |
| 2042 | +0.02379 |
| 2043 | +0.02984 |

The mean target weak-recall delta 95% bootstrap interval was
`[0.02619,0.03485]`.

## 9. Independent gate audit

| Frozen check | Result |
|---|---|
| macro-AUPR delta >= -0.005 | Pass |
| macro-F1 delta >= +0.010 | Pass |
| target weak recall delta >= +0.030 | Pass |
| each target AUPR delta >= -0.010 | Pass |
| candidate target FPR <= 0.20 | Pass: 0.19541 |
| positive weak-recall delta in at least 4/5 seeds | Pass: 5/5 |
| SawFM-present macro-F1 delta > 0 | Pass |
| SawFM-absent macro-F1 delta >= -0.005 | Pass |
| hashes match and outputs finite | Pass |
| no fit or selection on test-v2 | Pass |

Final decision: `promotion_gate_passed=true`.

## 10. Evidence boundary

- Historical locked-test was not accessed.
- development-v2 was not reused for final selection after development-v3 began.
- development-v3 selected no alpha; alpha=1.5 was already frozen.
- test-v2 was used once after bundle freeze and preflight.
- `fit_or_selection_on_test_v2=false`.
- `rerun_or_retuning_allowed=false`.
- The result cannot be used to modify the candidate, calibration or thresholds.

## 11. Formal artifacts

- v3 plan:
  `docs/plans/route_a_development_v3_label_selective_projection_plan_20260731.md`
- v3 generator:
  `experiments/generate_route_a_v3_dataset.m`
- v3 identity audit:
  `outputs/tapc/route_a_development_v3_generation/identity_audit.json`
- v3 experiment runner:
  `experiments/run_route_a_development_v3_label_selective_projection.py`
- v3 summary:
  `outputs/tapc/route_a_development_v3_label_selective_projection/route_a_development_v3_label_selective_projection_summary.json`
- one-time test plan:
  `docs/plans/route_a_gradient_contrast_test_v2_once_plan_20260731.md`
- frozen bundle:
  `frozen/20260731_route_a_gradient_contrast_test_v2/frozen_bundle.json`
- one-time runner:
  `experiments/run_route_a_gradient_contrast_test_v2_once.py`
- access marker:
  `outputs/tapc/route_a_gradient_contrast_test_v2_once_20260731/test_v2_access_started.json`
- final summary:
  `outputs/tapc/route_a_gradient_contrast_test_v2_once_20260731/route_a_gradient_contrast_test_v2_summary.json`
- final receipt:
  `outputs/tapc/route_a_gradient_contrast_test_v2_once_20260731/run_receipt.json`

## 12. Next action

The research goal terminates successfully at C level.  No further experiment may
use this test-v2 for tuning or confirmation.  The next legitimate work is result
packaging, ablation presentation and collection of a future external dataset;
it is not another test-v2 run.
