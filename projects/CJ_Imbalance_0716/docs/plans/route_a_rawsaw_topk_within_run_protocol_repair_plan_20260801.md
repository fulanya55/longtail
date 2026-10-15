# Route A RawSaw top-k 4/8/16: within-run protocol interpretation repair

Date: 2026-08-01

## Reason

The completed five-seed top-k experiment retrained 4/8/16 with equal budget,
parameter count, roles, calibration and threshold procedure. It failed only a
bitwise reconstruction check against an older RawSaw checkpoint after CUDA
epochs. The valid pooling comparator is the concurrently trained `top-k=16`
arm under the same run, not a consumed historical checkpoint.

The new audit is read-only. It preserves the historical state-drift values;
it never calls them reproducibility evidence, alters a model, or promotes a
top-k choice.

## Contract

- Inspect all five existing seed receipts and the source aggregate only.
- Verify common data SHA and five identity-isolated role hashes.
- Verify equal trainable parameter count across 4/8/16 within every seed,
  exact non-SawFM probabilities/predictions, and all test-use flags false.
- Report state drift from the historical checkpoint as a separate diagnostic,
  without making it an invalidation condition for the within-run contrast.
- Preserve original paired bootstrap summaries for top-k 4/8 versus 16.

## Interpretation

This completes core item 11 as a controlled pooling sensitivity ablation. It
cannot change the frozen RawSaw candidate, select a new top-k, or support
test-v3 use. Any later candidate change requires fresh confirmation.
