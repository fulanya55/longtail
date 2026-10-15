# Route A H0: ISRJ delay-block self-similarity result

## Decision

**Stop the fixed absolute-delay correlation-map mechanism.** The map retains a
small, reproducible association with ISRJ replica count relative to its
same-architecture label-permutation control, but it does not meet the
pre-registered absolute identifiability thresholds. It cannot enter an H1
recognition experiment, be fused into any existing expert, or be evaluated on
test-v3.

## Fresh protocol

- Fresh train-only source: 5,040 identities, generator seed `700000000`, SHA-256
  `760861c243f267693be4bbe380b5c5da4794bad6789c3eb65a39748f34462fa3`.
- ISRJ-only roles: fit/early/evaluation = `1536/512/512` receiver identities,
  with zero identity overlap against every audited historical train source.
- ISRJ width is `[10,30] us`; replica count is integer `[3,8]`.
- The fixed representation scans 26 lags `80..480` samples and 12 overlap
  blocks, retaining real and imaginary normalized complex correlations
  (`2x26x12`); no ACF peak is selected per record.
- Both 2,497-parameter CNN regressors have identical initialization, map,
  optimizer, batches and early stopping. `R-real` receives true replica count;
  `R-perm` receives a fixed fit/early permutation. Five seeds are 2039--2043.

## Held-out result

| Quantity | Mean / result | Pre-registered requirement |
| --- | ---: | ---: |
| R-real R2 | `0.07074` | >= `0.20` |
| R-real Spearman | `0.32220` | >= `0.40` |
| R-real minus R-perm R2 | `+0.08590`, CI `[+0.06377,+0.10184]` | >= `+0.20`, lower CI > 0 |
| Positive R2 contrasts | `5/5` | >= `4/5` |
| Normalized-MAE contrast | `-0.02748`, CI `[-0.03017,-0.02566]` | descriptive |

All parameter-count, source hash, role, identity and sealed-test checks pass.
The formal receipt records `stop_delay_block_map`; historical locked test,
test-v2 and sealed test-v3 were not opened.

## Failure interpretation

The stable positive permutation contrast rules out a completely uninformative
map. The failure is instead absolute: fixed lags mix two generator axes. ISRJ
copies occur at integer multiples of a sample-specific width (100--300 samples),
whereas the representation uses one global absolute lag grid. A replica-count
change can therefore move correlation structure along the lag axis rather than
producing a common pattern. The current map also sees composite interference,
so its weak distributed signal is inadequate for a recognition branch.

This does **not** authorize a lag-grid, block-count, CNN-width, epoch or
regularization search on this source. A distinct future H0 may only test the
separate physical hypothesis that width-normalizing the delay axis exposes
replica structure. Such an oracle-only upper-bound screen must use a new source
and, if successful, must later prove that frozen predicted width can replace
the oracle before any classification experiment.

## Artifacts

- Plan: `docs/plans/route_a_h0_isrj_delay_block_self_similarity_20260802.md`
- Model/tests: `cjlt/models/delay_block_self_similarity.py`,
  `tests/test_delay_block_self_similarity.py` (`3 passed`)
- Generator/audit/runner: `experiments/generate_route_a_h0_isrj_delay_block_self_similarity.m`,
  `experiments/audit_route_a_h0_isrj_delay_block_self_similarity.py`,
  `experiments/run_route_a_h0_isrj_delay_block_self_similarity.py`
- Formal summary: `outputs/tapc/route_a_h0_isrj_delay_block_self_similarity/route_a_h0_isrj_delay_block_self_similarity_summary.json`
