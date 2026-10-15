# Route A v22: visibility-balanced morphology expert

## Single hypothesis

The v19 morphology expert established that the fixed time-frequency features contain VFT-versus-ISRJ information, while v21 established that selecting a smaller downstream residual cannot safely recover weak-VFT evidence. **H-v22:** giving weak and strong VFT positives equal total training mass in the morphology expert will improve held-out weak-VFT-versus-ISRJ-negative ranking without materially degrading strong-VFT ranking. This tests representation learning, not another fusion-coefficient or threshold rule.

## New data and roles

- Generate a new train-only six-label source with base seed `620000000`, 5,040 unique receiver identities, 80 samples per nonempty label pattern, ISRJ width `[10,30] us`, and replicas `[3,4]`. No test split is generated.
- Retain ISRJ-present identities; target is VFT present versus absent. Define weak VFT with the existing 10 dB relative-power contract.
- Split by `receiver_noise_seed mod 10`: fit `{0..5}`, early-stop `{6,7}`, evaluation `{8,9}`. All roles must be receiver-identity disjoint and each must contain at least 20 weak VFT positives, strong VFT positives, and VFT-absent/ISRJ-present negatives.
- v19-v21 data and predictions are forbidden inputs. Q4, every historical locked test, test-v2, and sealed test-v3 are forbidden.

## Matched models and training

Both experts use the frozen v19 268-dimensional I/Q STFT morphology feature and the same `268 -> Linear(64) -> GELU -> Dropout(0.10) -> Linear(1)` head (17,281 parameters). Both use AdamW (`lr=3e-4`, `weight_decay=1e-4`), batch 64, 30 epochs, and identical seed-specific initializations.

- `M_BCE`: ordinary `BCEWithLogitsLoss`.
- `M_VBF`: per-sample weighted BCE. Strong VFT positives and negatives have weight 1. Weak VFT positives have the deterministic fit-role weight `n_strong_positive / n_weak_positive`, so weak and strong positive strata have equal total loss mass. The numeric weight is recorded but not tuned.

Both select their checkpoint by the same unweighted three-stratum early-stop loss: the mean of BCE losses on weak positives, strong positives, and negatives. No calibration, probability threshold, Q4 logit, width, residual, or learned gate is used.

## Outputs and gates

On held-out identities, store checkpoints, histories, normalization, predictions, hashes, and per-seed metrics for overall, weak-positive-versus-hard-negative, and strong-positive-versus-hard-negative AUROC/AUPR. Report paired bootstrap 95% CIs.

Promotion requires all of:

1. `M_VBF - M_BCE` weak AUROC mean >= `+0.030`, lower paired-bootstrap CI > 0, and >=4/5 positive seeds.
2. `M_VBF - M_BCE` weak AUPR mean >= `+0.020`.
3. Strong AUROC loss is no worse than `-0.010` in every seed; overall AUROC loss is no worse than `-0.010` in every seed.
4. Architecture, initialization, source audit, artifact hashes, and test-v3 seal all pass.

Failure stops visibility-balanced morphology training and rules out attaching it to Q4. Passing permits a separately preregistered fresh-source fusion experiment only; it does not change Q4 or authorize test-v3.
