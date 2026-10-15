# Route A v17: frozen predicted-width / Q4 VFT-risk alignment result

## Evidence role and boundary

This is a **fresh external-physics frozen-risk diagnostic**, not a new classifier result and not evidence that a width residual improves recognition. It replayed the immutable five-seed `H1_B2_MFTSharedDecision_VFTRidgeFusion_v2` Q4 path on the v17 paired external intervention. Checkpoints, calibration, thresholds, and fusion features were reused exactly; no refit, tuning, calibration, threshold search, or hyperparameter search occurred.

The two v17 files are train-role external-physics data only and were audited as identity/label/nuisance matched. The analysis uses the paired `VFT=0 AND ISRJ=1` hard-negative view: 320 identities in narrow and the same 320 in wide per model seed. Historic locked test, test-v2, sealed test-v3, and v17 test roles were all unused (`false`).

## Prespecified gate outcome

All five frozen classifier seeds passed the alignment screen.

| Endpoint (wide minus narrow) | Mean over 5 frozen seeds | Seed-level bootstrap 95% CI | Positive seeds | Gate |
| --- | ---: | ---: | ---: | --- |
| Q4 VFT probability | +0.2361 | [+0.2015, +0.2611] | 5/5 | >= +0.020 and >=4/5: pass |
| Q4 VFT FPR at the original threshold | +0.3744 | [+0.2981, +0.4256] | 5/5 | >= +0.050 and >=4/5: pass |
| frozen width vs Q4 VFT probability Spearman | 0.6411 | [0.5826, 0.6933] | 5/5 | >= 0.20 and >=4/5: pass |

The fixed Q4 VFT false-positive rate moved from `0.0719--0.1688` in the narrow regime to `0.3938--0.5656` in the wide regime, depending on the frozen seed. In every seed, the receiver order, hard-negative labels, and original six-label threshold vector were exact across the two paired regimes.

## Interpretation

Together with v16 learnability and v17 width transport, this establishes the following limited chain:

1. I/Q contains recoverable information about the controlled ISRJ-width axis.
2. A frozen width probe transports to a new, paired external-physics range.
3. In the same paired hard-negative view, increasing frozen predicted width co-varies strongly with an *existing fixed Q4 VFT false-positive risk*.

Thus predicted ISRJ width is decision-relevant to Q4's VFT hard-negative failure mode. It does **not** establish causation for a proposed residual, nor prove that modifying Q4 will lower FPR without harming VFT recall.

## Decision and next evidence boundary

The pre-registered decision is to permit design of a **fresh-data-only bounded width-conditioned residual**. That candidate must be developed on new identity-isolated data (not v17), with coefficient/cap/activation and any threshold rule selected entirely there. It must compare fixed Q4 against a same-capacity non-width control; report VFT recall, VFT hard-negative FPR, ISRJ and SawFM non-regression; and use a separate fresh confirmation split. Neither v17 result may tune that candidate, and test-v3 remains sealed.

## Reproducibility artifacts

- Plan: `docs/plans/route_a_v17_predicted_width_frozen_q4_alignment_plan_20260802.md`
- Runner: `experiments/run_route_a_v17_predicted_width_frozen_q4_alignment.py`
- Formal summary and receipt: `outputs/tapc/route_a_v17_predicted_width_frozen_q4_alignment/`
- Per-seed Q4 predictions: `W10_15_R3_4/seed_*/Q4_VFTAlignedRidgeFusion/` and `W22_30_R3_4/seed_*/Q4_VFTAlignedRidgeFusion/`
- Implementation tests: `tests/test_route_a_v17_predicted_width_frozen_q4_alignment.py`

