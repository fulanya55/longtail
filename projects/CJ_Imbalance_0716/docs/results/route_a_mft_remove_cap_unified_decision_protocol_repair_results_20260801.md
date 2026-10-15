# Route A MFT residual-cap removal: role-correct protocol repair results

## Scope

This is an implementation-contract repair, not a new candidate search and not
a replacement for the historical failed receipt.  The original plan defined
FPR feasibility on the assigned `threshold` role, while its implementation
checked the independent `early` role.  The repaired runner replays the exact
frozen Q1/Q4 logits and MFT evidence, records both roles separately, and leaves
the historical failure artifact intact.

## Protocol integrity

All five seeds reconstruct cap 0.75 to <= `1e-6`; all non-MFT logits remain
exact.  Threshold-role FPR is feasible for every arm and seed:

| Seed | cap 0.75 threshold FPR | cap 1.0 threshold FPR | cap 0.75 early FPR | cap 1.0 early FPR |
|---:|---:|---:|---:|---:|
| 2039 | 0.19274 | 0.19516 | 0.18145 | 0.18387 |
| 2040 | 0.19194 | 0.19194 | 0.18226 | 0.17500 |
| 2041 | 0.19274 | 0.20000 | 0.17581 | 0.18871 |
| 2042 | 0.19435 | 0.19194 | 0.20161 | 0.19274 |
| 2043 | 0.19516 | 0.19919 | 0.18871 | 0.18629 |

The one early cap-0.75 observation above 0.20 remains visible: it is an
independent generalization observation, not evidence that the threshold-role
selection failed.  No threshold was adjusted based on that observation.

## MFT mechanism result

Natural cap 1.0 minus cap 0.75 produces MFT weak-negative AUROC `+0.005984`
(95% CI `[+0.005144,+0.006654]`, 5/5 seeds), weak recall `+0.008016`
(`[+0.003367,+0.012441]`, 4/5), and AUPR `+0.001391`
(`[+0.001140,+0.001570]`, 5/5).  Early MFT FPR changes by `-0.000645`
(`[-0.006935,+0.006774]`).  The cap is therefore a real but small ranking and
decision sensitivity, not an explanation for the whole MFT effect.

This closes core-14 item 8 as a role-correct **development mechanism
ablation** only.  It does not change the current candidate, override the
external-OOD failure, or permit test-v3 use.

## Artifacts and boundary

- Plan: `docs/plans/route_a_mft_remove_cap_unified_decision_protocol_repair_plan_20260801.md`
- Runner: `experiments/run_route_a_mft_remove_cap_unified_decision_protocol_repair.py`
- Tests: `tests/test_route_a_mft_remove_cap_unified_decision_protocol_repair.py` (2 passed)
- Summary SHA-256: `DF7B4AAB336BA6C7CB43588C7CDABFEF9ADE2A2D78253534518E08D61CD19146`
- `locked_test_used=false`, `test_v2_used=false`, `test_v3_used=false`.
