# Route A H0b: ISRJ width-normalized delay-map oracle result

## Decision

**Pass as an oracle physical-identifiability upper bound.** When every delay
row is expressed as an integer multiple of the record's true ISRJ width, the
fixed distributed complex correlation map recovers ISRJ replica count strongly
on held-out receiver identities. This establishes a physical representation
mechanism, but it is not yet deployable because it uses an oracle width to
define map coordinates.

## Fresh protocol

- New train-only source, seed `710000000`, SHA-256
  `bd0b15af56b404247c2210ce2d861c0a4787690be17999e127a597ef686c2f11`.
- 5,040 unique identities; 2,560 ISRJ-positive samples; fit/early/evaluation
  identity roles `1536/512/512`; zero overlap with H0 or any other audited
  historical train source.
- Width `[10,30] us`, replicas `[3,8]`; true width is used only to form H0b's
  diagnostic lag coordinate `w,2w,...,8w` and is not supplied to the regressor.
- A fixed `2x8x12` real/imaginary correlation map feeds a 2,497-parameter CNN
  replica regressor. `R-real` and same-initialized `R-perm` use identical
  maps, batches, AdamW/Smooth-L1 budget and early stopping; only fit/early
  replica targets are fixed-permuted for `R-perm`.

## Held-out five-seed result

| Quantity | Result |
| --- | ---: |
| R-real mean R2 | `0.71661` |
| R-real mean Spearman | `0.84959` |
| R-real minus R-perm R2 | `+0.72508`, 95% CI `[+0.70980,+0.74432]` |
| Positive R2 contrasts | `5/5` |
| R-real minus R-perm normalized MAE | `-0.17367`, 95% CI `[-0.17575,-0.17158]` |

All pre-registered gates pass: absolute R2/Spearman, paired magnitude and CI,
positive-seed count, parameter count and test-v3 seal. Neither historical
locked test, test-v2 nor test-v3 was opened.

## Mechanism conclusion and next gate

H0's absolute-lag grid failed because widths vary from 100--300 samples. H0b
shows that the delay-copy structure becomes coherent when its axis is normalized
by the physical capture width. This is direct evidence for a scale-conditioned
ISRJ self-similarity representation, not evidence for a six-label classifier.

The only permitted next step is H0c on a third fresh identity source: replace
true width with the frozen v16 I/Q-predicted width, compare predicted-map real
and permutation arms, and quantify loss relative to an oracle-map arm on that
same source. No classifier residual, fusion, calibration, threshold change or
test-v3 access is authorized by H0b.

## Artifacts

- Plan: `docs/plans/route_a_h0b_isrj_width_normalized_delay_map_20260802.md`
- Model/test: `cjlt/models/delay_block_self_similarity.py`,
  `tests/test_delay_block_self_similarity.py` (`4 passed`)
- Generation/audit/runner: `experiments/generate_route_a_h0b_isrj_width_normalized_delay_map.m`,
  `experiments/audit_route_a_h0b_isrj_width_normalized_delay_map.py`,
  `experiments/run_route_a_h0b_isrj_width_normalized_delay_map.py`
- Summary: `outputs/tapc/route_a_h0b_isrj_width_normalized_delay_map/route_a_h0b_isrj_width_normalized_delay_map_summary.json`
