# Route A H0c: frozen predicted-width substitution result

## Decision

**Stop single-predicted-width substitution.** Although H0b's oracle
width-normalized map is strongly informative, replacing its coordinate with any
of the frozen v16 point predictions eliminates replica-count recoverability.
The v16 probes, normalizers and H0c models must not be refit or tuned to rescue
this mechanism.

## Fresh, frozen protocol

- Third fresh train-only source, seed `720000000`, SHA-256
  `aa28f80d9e427c3d3f89f0176e90e41bfc5147816ff7bbe2b2002706d0cb5386`;
  5,040 new identities and ISRJ-positive fit/early/evaluation roles
  `1536/512/512`, audited disjoint from H0, H0b and all historical sources.
- Each seed loads only its pre-existing frozen v16 `W_real` checkpoint and the
  same frozen fit normalizer, with recorded SHA-256 hashes. No source width
  model is fit, calibrated or selected.
- `O-real` is an oracle-map reference; `P-real` uses the frozen predicted,
  rounded/clamped width coordinate; `P-perm` has exactly the `P-real` map and
  initialization but fit/early replica labels are fixed-permuted. All three
  heads have 2,497 parameters and identical training budget.

## Five-seed held-out result

| Quantity | Result | Required |
| --- | ---: | ---: |
| P-real mean R2 | `-0.00842` | >= `0.20` |
| P-real mean Spearman | `0.10266` | >= `0.40` |
| P-real minus P-perm R2 | `+0.00142`, CI `[-0.01531,+0.01221]` | >= `+0.20`, lower CI > 0 |
| P-real minus O-real R2 | `-0.71536`, CI `[-0.74121,-0.68738]` | >= `-0.15`, lower CI >= `-0.20` |
| Positive predicted-vs-perm effects | `4/5` | >= `4/5` |

The small positive-seed count does not compensate for failed absolute,
permutation and oracle-fidelity gates. The formal receipt is
`stop_predicted_width_substitution`; no locked test, test-v2 or sealed test-v3
was used.

## Mechanism inference

H0b's lag rows are `m*w` for `m=1..8`. The frozen width predictors have a
useful continuous width signal but point error is amplified by multiplier:
even a few microseconds becomes a large discrete delay error at high `m`.
Complex local coherence is phase-sensitive, so rounding one predicted coordinate
per record causes the map to lose its alignment. This is an inference-time
representation mismatch, not grounds to alter the frozen width regression or
to attach an oracle map to classification.

The next distinct H0 may use a fixed *bank* of physically feasible width
hypotheses and aggregate repeated-copy lines without a predicted or oracle
width. It must use a fourth fresh source and a label-permutation control; it
cannot reuse H0c data or change H0c's point-prediction rule.

## Artifacts

- Plan: `docs/plans/route_a_h0c_isrj_predicted_width_substitution_20260802.md`
- Generator/audit/runner: `experiments/generate_route_a_h0c_isrj_predicted_width_substitution.m`,
  `experiments/audit_route_a_h0c_isrj_predicted_width_substitution.py`,
  `experiments/run_route_a_h0c_isrj_predicted_width_substitution.py`
- Formal summary: `outputs/tapc/route_a_h0c_isrj_predicted_width_substitution/route_a_h0c_isrj_predicted_width_substitution_summary.json`
