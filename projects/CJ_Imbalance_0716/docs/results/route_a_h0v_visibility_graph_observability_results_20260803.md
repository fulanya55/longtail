# Route A H0v: natural visibility graph observability result

Date: 2026-08-03  
Status: **failed the preregistered no-fit gate; H0v is closed.**

## Question and boundary

H0v tested one literature-derived, non-overlapping structural statistic from
[Cao et al., Sensors 2019](https://doi.org/10.3390/s19102344): natural
visibility-graph normalized degree entropy of 128 evenly spaced real-I samples.
It is not a model experiment. No classifier, calibration, threshold, loss,
network, or hyperparameter was fitted or selected.

The source was newly generated through the official 0716 closed-set generator:
1,800 exact absent/weak counterfactual pairs (600 each for MFT, VFT and ISRJ),
all with the target weak component fixed to -12 dB and identical nuisance
labels, sample seed and receiver-noise seed within each pair. A separate
30-pair smoke checked the MATLAB v7.3 complex-I/Q layout and is not used below.

The fixed temporal-order control permuted the same 128 values by the single
predeclared `numpy.random.default_rng(20260803)` permutation. It preserves the
marginal sample values while destroying temporal order.

## Results

| Target | true-order AUROC | permuted AUROC | true - permuted | paired weak > absent | Gate result |
| --- | ---: | ---: | ---: | ---: | --- |
| MFT | 0.5078 | 0.5114 | -0.0036 | 0.5500 | fail |
| VFT | 0.5013 | 0.5107 | -0.0094 | 0.5367 | fail |
| ISRJ | 0.5166 | 0.5254 | -0.0089 | 0.5600 | fail |

The 5,000-pair-resampling bootstrap 95% CI of the true-minus-permuted AUROC
was `[-0.0212, 0.0137]` for MFT, `[-0.0260, 0.0071]` for VFT and
`[-0.0272, 0.0104]` for ISRJ. All include zero and all point estimates are
negative. All true-order AUROCs are below the required 0.58; ISRJ alone meets
the paired-direction fraction, but this cannot compensate for the failed
AUROC and temporal-order-control gates.

## Interpretation and stop decision

The result is stronger than merely saying that a graph head did not train: the
predefined graph entropy itself contains no detectable weak-component ranking
signal under the matched-background, matched-receiver counterfactual contract.
Moreover, destroying time order did not lower discrimination; it weakly
increased AUROC for all three targets. Therefore this entropy is not evidence
of a new temporal physical observable in this protocol.

Per the preregistration, H0v is closed globally. Do not retry with a different
node count, I/Q projection, graph statistic, permutation, target power, ROI,
loss, threshold, backbone or fusion head. It does not reopen any previously
closed MFT/VFT/multi-PRI route and it supplies no authorization for a
LabelCoRank-adapted model or test-v3.

## Reproducibility artifacts

- Plan: `docs/plans/route_a_h0v_visibility_graph_observability_20260803.md`
- Official-generator wrapper: `experiments/generate_route_a_h0v_visibility_graph_pairs.m`
- No-fit analysis: `experiments/run_route_a_h0v_visibility_graph_observability.py`
- Formal source receipt: `outputs/tapc/route_a_h0v_visibility_graph_generation/formal_generation_receipt.json`
- Original analysis receipt (before explicit receiver/sample pair check):
  `outputs/tapc/route_a_h0v_visibility_graph_observability/formal_summary.json`
- Final pair-contract analysis receipt:
  `outputs/tapc/route_a_h0v_visibility_graph_observability/formal_summary_v2_pair_contract.json`

The final analysis has `trained_parameters=0`, `locked_test_used=false`,
`test_v2_used=false` and `test_v3_used=false`. The implementation tests passed
(`5 passed`) before the final contract-complete analysis.
