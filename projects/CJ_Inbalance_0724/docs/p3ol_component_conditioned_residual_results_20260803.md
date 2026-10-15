# P3o-L: durable D-drive posterior-component-conditioned residual

## Decision

**Frozen: formal three-seed development gate failure.** P3o-L repaired the preceding generation infrastructure without changing the P3o-J scientific method. All three independent seeds completed, but L1's posterior-selected 6x12 correction did not meet the pre-registered magnitude requirement against L0. Consequently, 1550M/1560M confirmation was not generated; no test or locked test was generated, opened, or used.

## Evidence boundary and protocol

- Train/development identities: 1530M/1540M, respectively. Each role contains 2,079 identities; train-development overlap is zero and the maximum overlap with 363 historical role files is zero.
- Raw data were generated and rechunked on D:. Each role had 5 regimes x 3 powers = 15 committed train views with no partial file. Three powers were identity-paired and train/development label contracts matched.
- Before generation: `271 passed, 9 warnings`; targeted P3o-L tests passed after the D-drive audit integration.
- Seeds 2101/2102/2103 each ran in a separate process, with an individual success/exception receipt. All wrote success receipts; none wrote an exception receipt.
- L0 is the finite-valid shared delay residual. L1 changes only the trainability of a posterior-selected 6x12 component table; classification base parameters, buffers and logits are bitwise preserved, and F-only calibration is reused unchanged.

## Gates

| Gate | Result |
|---|---:|
| Identity/power/label audit, finite-valid loss, and F-state/logit invariance | Pass |
| L1 miss penalty non-worse than L0, all seeds | Pass |
| L1 worst-combination predicted-gate NMAE / F-base <= 0.98, all seeds | Pass |
| F-base safe and retrieval gate, all seeds | **Fail** |
| L1 oracle and predicted-gate NMAE / L0 <= 0.98, all seeds | **Fail** |
| L1 worst-combination predicted-gate NMAE / L0 <= 0.98, all seeds | **Fail** |

## Fixed seed-level comparison

| Seed | L1 oracle / L0 | L1 predicted / L0 | L1 miss delta | L1 worst / L0 | L1 worst / F |
|---:|---:|---:|---:|---:|---:|
| 2101 | 0.99836 | 0.99826 | -0.000711 | 0.99646 | 0.94422 |
| 2102 | 0.99823 | 0.99812 | -0.000746 | 0.99674 | 0.94030 |
| 2103 | 0.99842 | 0.99832 | -0.000696 | 0.99704 | 0.95425 |

L1 changes every seed in the favorable direction, but only by about 0.16%--0.18% for oracle/predicted-gate NMAE and 0.30%--0.35% for worst-combination NMAE relative to the strong L0 control. This is far below the pre-registered 2% effect threshold. It is therefore a diagnostic of weak signal, not mechanism evidence.

The base safe/retrieval conjunction also failed because seed 2102 has ISRJ recall 0.82323 and 25 dB recall 0.62027, below the required 0.85 and 0.78. This is a base generalization failure even though all observed development maximum cell FPRs were below 0.10.

## Interpretation

P3o-L rules out the simple detached-posterior linear component table as a sufficient cross-identity parameter-risk mechanism. The result is not explained by the former MATLAB failure: data generation, D-drive storage, identity isolation, finite-valid loss, state preservation and durable execution all passed. A successor must introduce a different observable information source (for example component-local analytic/temporal evidence with a frozen conditional attachment), use new identities, and compare against L0 under the same predicted-gate and worst-risk constraints. It must not sweep the 6x12 table width, coefficient scale, or related training weights on 1530M/1540M.

## Artifacts

- [Pre-registration](../configs/p3ol_component_conditioned_residual_preregister_20260802.json)
- [Method freeze](../configs/p3ol_component_conditioned_residual_method_freeze_20260802.json)
- [Data audit](../outputs/p3ol_durable_d_drive_development_20260802/data_audit.json)
- [Merged development receipt](../outputs/p3ol_durable_d_drive_development_20260802/summary.json)
- [Outcome freeze](../configs/p3ol_component_conditioned_outcome_freeze_20260803.json)
