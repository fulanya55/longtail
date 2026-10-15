# Route A research restart: target-specific complex local-evidence branch

> **Superseded for execution.** This early design predates the v25 cyclic
> comparator validity audit. Its three identical branches with cyclic routing
> are permutation-equivalent under correctly matched initialization and BCE;
> this document does not authorize a rerun, MFT confirmation or six-label
> integration. See [the corrective gate](route_a_valid_target_heterogeneity_restart_gate_20260802.md).

## Why this is non-overlapping

The candidate is not a reweighted positive loss, a Q4 residual coefficient, a frozen-F2 label query, query-derived top-k pooling, a low-dimensional logit fusion, or a concatenated STFT feature. It reads the raw complex I/Q signal with target-specific local filters before a classifier score exists. Its local evidence maps are exposed as a first-class output, while the global path remains separately observable and can remain unchanged in a later integration.

The design is motivated by the 0716 literature audit: multi-label long-tail losses such as DB/ASL/PGL act on label frequency or probability imbalance, while the target claim requires a physical within-label visibility mechanism. The latest audited GEDFD metadata was refreshed from Crossref on 2026-08-02, but no auditable official implementation was found, so it is retained as a documented comparator rather than a numeric baseline.

## Architecture: TS-CLE

Input is RMS-normalized I/Q `[B,2,3000]`.

1. A shared complex convolution (`1 -> 8`, kernel 17, stride 4) produces real/imaginary maps; their magnitude is the shared local carrier `[B,8,750]`.
2. The **global safety path** uses unweighted carrier mean and standard deviation (`16` values) and a linear six-label classifier. It never consumes a local label score.
3. Each target in `{MFT,VFT,ISRJ}` has an independent depthwise local filter bank with kernels `{9,33,129}`, 1x1 mixing, and a direct one-channel evidence map. A fixed top-k mean (`k=16`) and full-time mean are concatenated to form the target-local scalar. This map is generated before, and not from, the global classifier/logit/query.
4. The final target logit is `global_logit + tanh(scale_t) * local_logit_t`. The three `scale_t` parameters initialize at zero. FMB/SawFM/SinFM are exactly global-path outputs inside this module.

The same-capacity source control cyclically routes the three independently trained local branches to the wrong target output (`MFT <- VFT`, `VFT <- ISRJ`, `ISRJ <- MFT`) while keeping every convolution, classifier, global path, and scalar count identical. It tests target-specific physical source rather than parameter count.

## Required experiment before integration

This document deliberately does **not** authorize integration or test-v3. First, a fresh train-only identity source must assess raw expert evidence only:

- all six labels use ordinary presence BCE on fit identities; no power, weak mask, oracle width, threshold or calibration enters the model;
- weak masks are evaluation-only strata; the primary contrast is weak positive versus composition-matched target-absent negative for MFT/VFT/ISRJ;
- candidate and cyclic-source control share initialization, all training samples, batch order, optimizer, epochs, early role and parameter count;
- expose local-map entropy, map top-k coverage, global/local logit correlations, and strong/weak/negative ranks to rule out a hidden prior shift;
- preregister a gain threshold before generating the new source. Only an all-target five-seed expert-level pass permits a separate six-label C0/C1/C4 integration protocol.

## Safety constraints

- The global path must remain recoverable without local maps; zero local scales reproduce it exactly.
- FMB/SawFM/SinFM outputs must be bitwise equal under candidate/control routing when the shared state is equal.
- No result can be promoted by an F1 threshold: ranking and fixed-FPR safety are primary.
- Current development/test sources, historical locked test, test-v2 and sealed test-v3 are forbidden for selection.
