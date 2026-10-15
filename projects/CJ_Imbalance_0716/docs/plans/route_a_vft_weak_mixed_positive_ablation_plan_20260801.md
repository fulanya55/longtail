# Route A VFT weak-only versus mixed-positive Ridge ablation

## Question and boundary

Does the VFT Ridge-STFT expert obtain its weak-component evidence specifically
from training on strictly weak VFT positives, rather than merely from adding
more (predominantly strong) VFT positives?  This is a construction-stage
ablation on the already-used development-v3 roles.  It does not modify the
frozen candidate, it does not select a new candidate, and `test-v3` remains
sealed.

## Fixed protocol

- Five receiver-noise-seed-isolated development-v3 role sets: `2039`--`2043`.
- Fixed 16x16 VFT STFT feature extractor and fit-only z standardisation.
- Same `RidgeClassifier(alpha=10)`, intercept, and balanced class weights.
- Same strict composition-matched weak-positive evaluation view on validation
  and early roles.
- Same weak definition: active VFT and component power at or below 10 dB.

## Only intervention

`weak_only` trains on composition-matched weak positives, exactly matching the
frozen Ridge expert's training construction.  `mixed_positive` trains on all
composition-matched VFT positives (weak plus strong) versus the corresponding
matched negatives.  Evaluation remains strictly weak-positive in both arms;
therefore a gain cannot be attributed to changing the target metric.

## Interpretation rule

The result is descriptive rather than a promotion gate.  A material reduction
in held-out weak AUROC/AUPR for `mixed_positive` supports visibility-selective
supervision; non-inferiority or improvement falsifies that explanation and
blocks its use as a mechanism claim.  Neither outcome permits threshold tuning
or any access to `test-v3`.

## Artifacts

Save per-seed feature/training receipts, score metrics, bootstrap aggregate
differences, source hashes, and an explicit `test_v3_used=false` receipt under
`outputs/tapc/route_a_vft_weak_mixed_positive_ablation/`.
