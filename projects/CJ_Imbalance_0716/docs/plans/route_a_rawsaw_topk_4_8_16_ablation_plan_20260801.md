# Route A RawSaw top-k 4/8/16 core ablation

## Mechanism question

Does the RawSaw specialist's local-evidence pooling scale materially explain
SawFM routing performance, or is the fixed `top-k=16` an arbitrary
implementation setting?  `top-k` changes the pooled evidence values but not
the model's trainable parameter count, so each arm must be retrained under the
same budget; substituting it into a frozen checkpoint would not be a valid
ablation.

## Fixed protocol

- development-v3 receiver-identity roles and seeds `2039`--`2043` only;
- raw I/Q frontend, six physical descriptor channels, hidden channels `16`,
  kernels `7/31/63`, dropout `0.1`, RawSaw-only routing;
- class-balanced BCE with AdamW (`lr=1e-3`, `weight_decay=1e-4`), batch `64`,
  gradient clip `5`, at most 30 epochs and validation patience `5`;
- ordinary fit-only RawSaw training; validation selects only the checkpoint;
  calibration uses the calibration role, and a per-label weak-recall/FPR
  threshold is selected only on threshold role with FPR cap `0.20`;
- the C6 base logits are unchanged.  Only SawFM logits are replaced; all five
  non-SawFM probabilities and predictions must be bit-identical across arms.

## Arms and non-selection decision

`top-k={4,8,16}` are all run.  The top-k=16 arm must reproduce the frozen
RawSaw source weights to numerical tolerance.  This is a core ablation, not a
candidate search: the frozen candidate remains top-k=16 regardless of the
ranking.  Any changed top-k could only be considered in a new candidate and a
fresh development-v13 protocol after all required evidence gates.

## Recorded metrics

For SawFM: AUPR, weak recall, weak-vs-negative AUROC, strong recall, FPR,
calibration/threshold, per-seed histories, checkpoints, and source hashes.
For the six-label routed output: macro-AUPR/F1 and exact non-SawFM safety
comparison.  `test-v3` is sealed and never opened.
