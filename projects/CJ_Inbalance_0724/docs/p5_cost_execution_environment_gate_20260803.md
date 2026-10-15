# P5 computation-cost execution environment gate

## Observed environment

The workstation exposes one CUDA device: NVIDIA GeForce RTX 5080 (`GPU-cc711249-e009-a430-7c74-90b363ad873a`).  Its compute mode is `Default`; no second device is available for an isolated timing workload.

## Repeated validated interruption mechanism

Three independently pre-registered shared-forward timing attempts encountered external CUDA work after their own preflight:

| attempt | external CUDA workload | admissible cost result |
| --- | --- | --- |
| P5g | P3o-L training process already shared the device | none; stopped before a result |
| P5k | `p3on_resolution_abstention` (PID 69120) | none; guard stopped at a real CUDA Python PID |
| P5l | `run_route_a_h1b_vft_staggered_pri_all_backgrounds.py` (PID 71888) | none; guard stopped during the first reference loop |

P5h, P5i and P5j are separate pre-result implementation/guard-definition freezes.  P5f remains a numerical non-equivalence stop for selected-row lazy evaluation.  Thus there is no valid end-to-end shared-forward latency estimate in this series.

## Mandatory gate before another fresh cost experiment

Do not create P5m data, choose new seeds, relax the guard, reduce repetitions, or reuse P5g--P5l identities merely because the device happens to be idle at preflight.  A further fresh cost experiment is allowed only after one of the following is explicitly established and recorded:

1. a user-authorized exclusive GPU time window in which no other CUDA job will be launched for the complete measured run; or
2. user authorization to switch the sole GPU to an exclusive-process isolation mode and to restore the prior `Default` mode afterward; or
3. an independently available second GPU assigned to the experiment.

This gate changes no model, data role, threshold, metric or scientific claim.  It preserves `test_generated=false`, `test_opened=false` and `locked_test_used=false`.

## Latest non-mutating recheck

On 2026-08-03, a read-only `nvidia-smi` recheck reported the sole RTX 5080 in `Default` compute mode, 4% utilization and 1,550 MiB / 16,303 MiB memory already in use.  The process listing included multiple `[Insufficient Permissions]` entries, so it could not establish that all CUDA work was absent.  No P5 workload, timing loop, data generation, model execution, or GPU-mode change was started from this recheck.  The mandatory gate therefore remains unmet.
