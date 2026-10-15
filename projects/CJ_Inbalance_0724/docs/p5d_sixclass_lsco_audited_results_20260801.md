# P5d: audited six-class LSCO heterogeneous-route screen

## Protocol and evidence boundary

P5d is a fresh-identity, three-fold leave-set-of-compositions (LSCO) development screen.  The only changed axis from P4c is that every complete label combination assigned to the held-out cardinality band is absent from train, early stop, and any selection role.  Both models use the frozen P4c profile: 20 epochs, seeds 15041/15042/15043, width 24, BCE plus 0.1 masked SmoothL1, fixed 0.5 classification threshold, and parameter-NMAE early selection.  No calibration, threshold selection, test generation, test opening, or locked-test use occurred.

The independent role audit verified that all 6,300 fresh identities are partitioned in every fold, all role pairs are disjoint, training/early/seen-confirm roles exclude the held-out combinations, and the receipt hashes match.

## Complete rerun result

The first P5d execution was intentionally not promotable because it lacked fixed-threshold per-class Recall.  This instrumented rerun changed no model, data, role, seed, optimizer, epoch budget, early-selection rule, or threshold.  It only persisted checkpoints, logits hashes, and fixed-threshold Recall/FPR.  All 18 runs and all mandatory audit fields are present.

| frozen comparison, unseen combinations | result |
|---|---:|
| Mean heterogeneous-route parameter relative NMAE reduction | 7.43% |
| Positive parameter reduction | 9/9 pairs; all three seeds in each fold |
| Mean macro-AUPR delta | -0.235 percentage points |
| Mean subset-accuracy delta | +0.198 percentage points |
| Per-target all-three-seed degradation over 0.03 NMAE in any fold | none |

## Pre-registered safety-gate failures

The screen is **not promoted**.  Two gate families fail:

1. Macro-AUPR is worse than the allowed -0.005 in fold 0, seed 15042 (-0.00632) and seed 15043 (-0.00598).
2. Fixed-threshold recall gain is coupled with FPR inflation above 0.02 in three pair/class observations:
   - fold 0, seed 15042: ISRJ Recall +0.0100, FPR +0.0400;
   - fold 1, seed 15042: VFT Recall +0.0270, FPR +0.0420;
   - fold 1, seed 15042: ISRJ Recall +0.0150, FPR +0.0230.

This is a response-amplification failure under composition shift: the mean parameter loss improves, but the classification posterior does not safely separate difficult unseen-combination negatives.  It cannot support a claim of composition-disjoint classification improvement, nor authorize a confirmation run.

## Stop decision

The P5d preregistered stop rule applies.  Do not add composition prototypes, relation losses, margins, threshold fitting, or new experts using these unseen combinations.  P4c/P5b3/P5b4 remain evidence for the known closed-set distribution only; P5c remains a separate parameter-OOD boundary characterization.  The next experiment must be a separately preregistered evidence-completion study for an already-passing mechanism, rather than a P5d-derived rescue.

## Artifacts

- `outputs/p5d_sixclass_lsco_instrumented_rerun_20260801/results.json`
- `outputs/p5d_sixclass_lsco_instrumented_rerun_20260801/audit.json`
- `outputs/p5d_sixclass_lsco_instrumented_rerun_20260801/role_audit.json`
- `configs/p5d_sixclass_lsco_heterogeneous_preregister_20260801.json`
