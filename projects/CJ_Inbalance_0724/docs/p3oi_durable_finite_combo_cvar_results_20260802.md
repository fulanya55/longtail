# P3o-I: durable finite-valid bounded compound-CVaR residual

## Decision

**Frozen: formal development gate failure.** All three pre-registered model seeds completed in independent Python processes and produced a successful receipt. Nevertheless, I1 (equal-weight compound-label-group CVaR) did not meet the all-seed promotion rule. Therefore the reserved 1350M/1360M confirmation identities were not generated, and no test or locked test was generated, opened, or used.

This is valid three-seed *development* evidence about the frozen I1-vs-I0 comparison; it is not a test result or a claim of a deployable method.

## Protocol and evidence boundary

- Train/development identity seeds: 1330000000 / 1340000000; 2,079 unique identities per role.
- Identity audit: train-development overlap = 0; maximum overlap with 343 historical identity files = 0.
- Model seeds: 2081, 2082, 2083. Each ran alone, wrote either a success or exception receipt, and released CUDA cache on exit. All three wrote success receipts; none wrote an exception receipt.
- Loss contract: only finite entries satisfying `label=1 AND parameter_mask=1` enter Smooth-L1. The P3o-I unit test also checks finite gradients and one finite AdamW step with inactive NaN targets.
- Before data generation: `267 passed, 9 warnings`.
- F backbone, F calibration, thresholds, checkpoint rule, epochs (20), optimizer settings, bounded residual (0.25), and parameter-loss weight (0.1) were frozen. I0 is mean residual; I1 is exactly I0 plus equal-weight upper-quartile compound-label-group CVaR.
- `test_generated=false`, `test_opened=false`, `locked_test_used=false`.

## Gate result

| Gate | Result |
|---|---:|
| Zero identity overlap; finite-valid implementation; F state/logits bitwise preserved | Pass |
| F safety/retrieval gates for every seed | Pass |
| I1 miss penalty non-worse than F-base for every seed | Pass |
| I1 worst-combination predicted-gate NMAE / F-base <= 0.98 for every seed | Pass |
| I1 oracle and predicted-gate NMAE / F-base <= 0.98 for every seed | **Fail** |
| I1 worst-combination predicted-gate NMAE / I0 <= 0.99 for every seed | **Fail** |

## Fixed comparison results

| Seed | I1 oracle / F | I1 predicted-gate / F | I1 miss delta vs F | I1 worst / F | I1 worst / I0 |
|---:|---:|---:|---:|---:|---:|
| 2081 | 0.9897 | 0.9894 | -0.00446 | 0.9669 | 1.0150 |
| 2082 | 0.9772 | 0.9758 | -0.01038 | 0.9677 | 1.0161 |
| 2083 | 0.9909 | 0.9905 | -0.00410 | 0.9590 | 1.0146 |

I1 consistently improved the selected worst-combination metric relative to F-base (3.31%, 3.23%, and 4.10%) and did not increase the miss penalty. This is insufficient for promotion: its global oracle/predicted-gate improvement relative to F-base exceeded the required 2% only for seed 2082, and I1 was consistently 1.46%--1.61% worse than the simpler I0 method on the same worst-combination metric.

## Interpretation and next design constraint

The finite-valid correction and durable runner worked: this failure is not the prior inactive-NaN defect, an incomplete process, a classification-state leak, or an unsafe F calibration. It falsifies this particular mechanism: adding an equal-weight group-tail CVaR term to the bounded residual does not improve the mean-residual comparator on the hard group, despite improving that group versus F-base.

The next successor must not tune the CVaR fraction or its weight on these identities. It should use fresh identities and a materially different falsifiable mechanism, for example a diagnostic-first component-conditioned residual that targets the actual worst parameter/component group and compares against both F and the retained I0 baseline under the same state-preservation and predicted-gate rules.

## Artifacts

- [Pre-registration](../configs/p3oi_durable_finite_combo_cvar_preregister_20260802.json)
- [Method freeze](../configs/p3oi_durable_finite_combo_cvar_method_freeze_20260802.json)
- [Merged development receipt](../outputs/p3oi_durable_finite_combo_cvar_development_20260802/summary.json)
- [Outcome freeze](../configs/p3oi_durable_finite_combo_cvar_outcome_freeze_20260802.json)
