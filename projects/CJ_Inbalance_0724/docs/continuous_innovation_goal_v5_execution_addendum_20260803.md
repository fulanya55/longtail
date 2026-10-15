# V5 execution addendum: P3o-R closure and next admissible question

## Evidence update

P3o-R is complete and frozen. It closes the MFT-count component-separation branch under the tested contract:

1. P3c proves that exact MFT component count is physically recoverable (`MAE=0`, `Pearson=1`).
2. P3o-M/N/O/P/Q showed that direct compound-IQ count features only yield marginal or non-reproducible benefit.
3. P3o-R tested the missing supervised separation link on fresh `1880M/1890M` identities. A waveform-only separator raised raw-to-oracle cosine from `0.6119` to about `0.944`, yet the frozen inverse count saturated to eight for all 15,840 development views and worsened MAE in all three seeds.

Therefore no further local peak, phase, support-curve, ordinal head, generic waveform-separation, or threshold tuning on these MFT count identities is admissible. `1900M/1910M` remain ungenerated and unused.

## Next admissible research question

The main Level-4 claim concerns MFT-to-ISRJ false positives, not count error. The next distinct falsifiable question is:

> If the *exact, correctly power-scaled* MFT component is removed from compound I/Q before a frozen classifier, does ISRJ false-positive risk fall across fresh identities without unacceptable loss of ISRJ recall?

This is an oracle causal diagnosis, not a deployable method and not a parameter-head experiment. It decides whether a future support-aware subtractor has a physically meaningful classification target. It is distinct from P3o-R because P3o-R asked whether normalized MFT shape preserves `num_targets`; the new question asks whether the actual scaled MFT contribution causes the classification confusion.

## Required frozen design before any new data

- C0: frozen classifier on raw normalized compound I/Q.
- C1: same frozen classifier on compound I/Q minus exact, correctly power-calibrated MFT component, followed by the same input normalization.
- C2: energy-matched non-MFT subtraction control, to distinguish removal of MFT structure from arbitrary energy reduction.
- No classifier retraining, threshold change, router, parameter loss, MFT count head, or feature selection.
- Fresh train-free diagnostic identities only; `1920M/1930M` are candidates, subject to a recorded cross-history identity audit.
- Report per regime/power ISRJ FPR, recall, macro-AUPR, and paired receiver-noise-seed bootstrap; oracle access must be prominently labelled non-deployable.
- Promotion is allowed only if C1 improves the predefined worst-cell FPR and is superior to C2 while retaining ISRJ recall; otherwise the MFT component is not a demonstrated causal source of the Level-4 confusion.

## Storage prerequisite

P3o-R raw data plus sample-chunked caches leave only 7.34 GiB free on D:, while a new 30-view paired round requires roughly 8 GiB including caches. No historical data will be deleted or moved without user authorization. Before generating the next fresh identities, choose an approved storage action: archive/move specific P3o-R artifacts, or approve a preflighted high-capacity output root on C: (currently 373.6 GiB free) with the MATLAB write probe and capacity guard retained.

`locked_test_used=false`.

## P3o-S readiness update (2026-08-03)

The next question has now been implemented and frozen as **P3o-S oracle MFT subtraction causal diagnosis**:

- Preregister: `configs/p3os_mft_oracle_subtraction_preregister_20260803.json`.
- Frozen code: `cj0724/p3os_mft_oracle_subtraction.py`; the corresponding method freeze records the pre-generation SHA-256 values.
- Atomic MATLAB generator: `matlab/generate_p3os_mft_oracle_subtraction_level.m`.
- Conditions are C0 raw normalized compound I/Q, C1 exact generator-power-scaled MFT subtraction, and C2 deterministic equal-energy FFT-phase-scrambled subtraction.  The frozen main `mlcnn_iq_grouped_task_routed` checkpoints and P2c `safe_thresholds` are used unchanged.
- Formal data will be two train-free, identity-paired diagnostic seeds `1920000000/1930000000`, five regimes and 25/35/50 dB.  It will report source-subset (`MFT=1, ISRJ=0`) ISRJ FPR, recall, macro-AUPR, per-cell worst risk, and identity-clustered paired bootstrap for C0-C1 and C2-C1.
- Gates require the C1 source-FPR effect to exceed both C0 and C2 while retaining ISRJ recall, with all six data/model replications direction-consistent.  A failure stops a support-aware subtractor line rather than inviting threshold or oracle-control tuning.

Readiness-only verification passed: P3o-S plus P3o-R targeted tests `6 passed`; full repository regression `301 passed, 9 pre-existing warnings`; MATLAB `executeGeneration=false` preflight passed and did not generate a view.  There is still no P3o-S result, no generated test, and no locked-test access.

The remaining prerequisite is unchanged: do not set `executeGeneration=true` until an output location is authorized.  A full two-seed round needs approximately 8 GiB including rechunked caches; D: remains below this capacity, while C: has sufficient capacity but is a deliberate departure from the repaired default output root.
