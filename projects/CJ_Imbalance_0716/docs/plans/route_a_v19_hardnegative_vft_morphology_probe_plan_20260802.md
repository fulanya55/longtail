# Route A v19: nonlinear VFT morphology hard-negative separability probe

## Hypothesis

v18 showed that predicted ISRJ width can identify risk but cannot safely
correct Q4's high-confidence VFT false positives. A read-only v18 audit also
rejected reuse of the existing *linear* STFT ridge score: it was nearly
uncorrelated with Q4 hard-negative risk and had near-chance FP ordering.

**H-v19:** a compact nonlinear expert over the same raw I/Q time-frequency
morphology can separate `VFT=1, ISRJ=1` positives from the matched
`VFT=0, ISRJ=1` hard negatives on held-out receiver identities, beyond an
identical permutation-label capacity control. This is an identifiability probe
only; it does not attach to Q4 or alter calibration/thresholds.

## Fresh source and roles

- Generate a fresh train-role-only six-label file with base seed `560000000`,
  5,040 unique receiver identities, ISRJ width `[10,30] us`, replicas `[3,4]`,
  and no test file.
- Retain only ISRJ-present samples. The target is the VFT label, yielding a
  physically matched VFT-vs-ISRJ hard-negative task.
- Split by `receiver_noise_seed mod 10`: fit `{0..5}`, early-stop `{6,7}`, and
  held-out evaluation `{8,9}`. All roles are identity-disjoint; each must
  contain both target states.
- v17/v18 data and predictions are not read by the runner. Locked test,
  test-v2 and sealed test-v3 are forbidden.

## Matched probes

Each sample is complex-RMS normalized I/Q. The input is a frozen,
non-learned 16x16 log-normalized STFT distribution plus 12 frequency-track
statistics (`268` features). The learnable head is
`268 -> Linear(64) -> GELU -> Dropout(0.10) -> Linear(1)` (17,281 parameters).

- **M-real:** real fit labels.
- **M-perm:** fit labels permuted once within the fit role using a fixed seed;
  early-stop labels are independently permuted only to make its checkpoint
  choice equally label-uninformative. Evaluation always uses real labels.

Both start from identical seed-specific initialization; AdamW (`3e-4`, weight
decay `1e-4`), BCEWithLogits, batch 64, maximum 30 epochs, and minimum
early-stop-role loss checkpoint. No width input, Q4 score, calibration, or
threshold is available to either model.

## Outcomes and gates

On held-out identities, report AUROC, AUPR, weak-VFT AUROC where defined,
per-seed values, paired bootstrap CIs and stored predictions/checkpoints.
The probe passes only if:

1. M-real mean AUROC >= `0.65` and mean AUPR >= `0.65`.
2. M-real minus M-perm mean AUROC >= `0.10`, lower paired-bootstrap CI > `0`,
   and at least 4/5 seed deltas are positive.
3. M-real weak-VFT-vs-hard-negative AUROC >= `0.55` when the held-out weak
   stratum has at least 20 examples per seed.
4. Identity audit, artifact hashes and test-v3 seal all pass.

Failure rejects morphology evidence as a second condition and stops an
integrated residual. Passing permits only a newly preregistered Q4 integration
experiment on another fresh source, with a same-capacity control.

