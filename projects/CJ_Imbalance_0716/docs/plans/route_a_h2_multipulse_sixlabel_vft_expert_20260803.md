# Route A H2: role-isolated multi-pulse six-label VFT-expert model

Status: preregistered after H0q/H1a/H1b. H2 role-isolated sources, smoke and
the fixed five-seed formal queue have been launched under this protocol. No
individual mode/seed receipt is a selection result: only the post-queue,
hash-audited five-seed summary may decide the promotion gate. This is a
project-local staggered-PRI physics protocol, not a replacement for the
sealed original-0716 development-v3/test-v3 evidence chain.

## Research question

Can a true-timestamp VFT-coherence expert improve weak VFT recognition in a
multi-pulse six-label compound setting, beyond an information- and
parameter-matched timestamp-permuted expert, while preserving RawSaw's SawFM
advantage and all-label safety?

This is deliberately not a retry of C3/C4. The input observation contract is
different (eight staggered pulses); the target side feature is independently
validated by H0q/H1a/H1b; and the key control has identical LFM coefficients,
grid, delay window, feature dimension, projection, optimizer and loss but
permuted timestamps.

## Data roles and isolation

Generate three project-local, eight-pulse staggered-PRI multi-label sources:

| Role | Samples | Identity block | Permitted use |
| --- | ---: | --- | --- |
| train-mp-v1 | 5,040 | new block A | fitting only |
| development-mp-v1 | 2,520 | new block B | early stop, calibration and threshold selection |
| external-mp-v1 | 2,520 | new block C | one post-freeze external evaluation |

Each sample has labels `[FMB,SawFM,SinFM,MFT,VFT,ISRJ]`; sample combinations
are hash-recorded and stratified so each target has enough all-positive, weak
positive (VFT 8 dB) and exact composition-matched negatives. Components use
the H1b local staggered-PRI family. Within a counterfactual VFT subcohort,
present/absent pairs share all other components and receiver noise. No source
may overlap H0q/H1a/H1b, historic train identities, another role, or test-v3.

The train source is not a final external test; no original 0716 test-v2 or
test-v3 file is opened. The new local protocol must be named as such.

## Frozen candidate family

Input is split into eight 3,000-sample pulse windows. The shared I/Q branch
uses the existing multi-scale temporal encoder per pulse followed by temporal
mean/std pooling across pulses. Historical RawSaw weights must **not** be
silently transferred across observation protocols. C1 instead preserves the
fixed RawSaw *architecture role* (a SawFM-only raw local expert) and fits it
only on `train-mp-v1`; its success is evaluated independently on the new
protocol. The historical RawSaw result remains a separate, original-0716
evidence boundary.

For VFT only, compute the H1b fixed 25-bin true-timestamp normalized
coherence map over the `[100,500]` delay bank. A fixed small projection maps
that map to the VFT head. It is not reused by MFT/ISRJ heads. The matched
control replaces true timestamps with `[0,5,2,7,4,1,6,3]`; it retains exactly
the same shape and trainable projection. Both VFT heads receive the same
global feature and use the same binary loss.

| Required arm | SawFM role | VFT role | Purpose |
| --- | --- | --- | --- |
| C0-mp | none | none | F2+C6 shared-backbone baseline |
| C1-mp | RawSaw-architecture local expert | none | SawFM-only expert reference |
| C2-mp | none | true-timestamp VFT expert | VFT-expert attribution |
| C3-mp | RawSaw-architecture local expert | true-timestamp VFT expert | unified candidate |
| C4-mp | RawSaw-architecture local expert | timestamp-permuted expert | matched-capacity/information control |

All arms use the same batch schedule, augmentation, data order per seed,
parameter count outside the fixed expert, optimizer, loss and stopping rule.
C2/C3/C4 have exactly equal learned expert projection dimensions. The model
must expose a detached expert contribution and a global-path VFT logit for
safety analysis.

The frozen initial implementation uses pulse-backbone width `32`, VFT expert
projection `25 -> 16 -> 1`, and produces parameter counts C0 `26,246`, C1
`27,671`, C2 `26,679`, C3 `28,104`, C4 `28,104`. C3/C4 equality is an
executable test, rather than an estimate. The SawFM expert is target-only;
the VFT expert affects only the VFT logit.

## Training, calibration, and thresholds

Use five declared training seeds `20260803, 20260811, 20260819, 20260827,
20260904`. Fit on train-mp only using AdamW (LR `3e-4`, weight decay `1e-4`),
batch `16`, BCEWithLogitsLoss, global gradient clip `5`, and at most `24`
epochs. Choose epoch by raw macro-AUPR on development-mp only, without VFT
threshold use. Split complete development pairs by `pair_seed mod 2`: fit one
per-label temperature on even pairs and select thresholds on odd pairs. The
VFT threshold maximizes weak-VFT recall subject to FPR <= `0.15`; other labels
maximize F1. Freeze model epoch, calibrator and threshold before external-mp
evaluation. Report raw ranking and calibrated threshold metrics separately.

The initial loss is the existing fixed DB/BCE baseline loss, not a new
visibility weight. That isolates the contribution of the observation and
expert. Any loss change requires a separate, pre-registered same-capacity
ablation after the architecture itself passes.

## Promotion gates

On five seeds, C3-mp must beat both C0-mp and C4-mp on development-mp by:

1. VFT weak-positive AUROC mean >= `+0.030` with 5/5 positive seed effects;
2. VFT weak-positive AUPR mean >= `+0.025` with paired seed CI lower > 0;
3. VFT weak recall at a development-fixed FPR constraint >= `+0.030`, with no
   VFT FPR increase above `+0.015`;
4. C3-mp SawFM AUPR and weak recall are noninferior to C1-mp within `-0.010`;
   non-VFT macro-AUPR drop is <= `0.010`;
5. C2-mp/C3-mp true expert beats C4-mp's permutation expert, so neither
   capacity nor a generic LFM score explains the result.

Only after all gates pass may the frozen winning arm run once on external-mp.
External-mp must reproduce directional VFT ranking improvement with a positive
paired confidence lower bound, before any paper-level claim. Test-v3 remains
sealed until the complete six-label evidence chain, standard baselines,
ablations, robustness source and external-mp have all passed.

## Failure rule

If B1-true does not beat B1-perm under these gates, the side expert is not
target-specific enough for learned fusion. Do not tune grid, delay ROI,
timestamp permutation, expert width, loss weight, threshold or data identity;
record a negative H2 result and return to protocol-level analysis instead.
