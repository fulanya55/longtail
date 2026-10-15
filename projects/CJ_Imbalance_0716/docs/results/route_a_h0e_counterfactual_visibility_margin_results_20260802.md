# Route A H0e: Counterfactual Visibility Margin Result

## Decision

**Stop `counterfactual_visibility_margin`.** The training-time true-pair margin did not outperform its information- and compute-matched shuffled-pair control, and it degraded VFT/ISRJ all-positive ranking. It cannot enter six-label fusion, calibration, threshold selection, or test-v3.

## Data and boundary

- Fresh train-only data: `C:\LJQ_code\干扰数据\复合干扰数据0802_counterfactual_visibility_h0e\counterfactual_visibility_pairs_train.mat`; SHA-256 `39750e1345be3020b3fad4f54d400f7367b20c598a860135d5fe13b90a126f62`.
- 2,400 pair identities / 4,800 views. Pair members share sample seed, receiver-noise seed, retained components and parameters; only MFT, VFT, or ISRJ is toggled. Every target has 800 pairs at fixed target relative power -12 dB.
- Fit/early/evaluation are 1,440/480/480 pairs; every target has 480/160/160 pairs. All historical training identity intersections are zero.
- No historical locked test, test-v2, or test-v3 was generated, opened, analyzed, or used.

## Fixed comparison

All arms use the same 19,526-parameter `CounterfactualVisibilityProbe` and single-view I/Q inference.

| Arm | Training objective |
|---|---|
| B0 | six-label BCE |
| B1_shuffle | BCE plus `0.5*softplus(0.5-(z_plus-z_negative))`, with same-target negatives deterministically rematched each epoch |
| C_CVMC | identical BCE, margin, parameters, initialization and budget, with true same-background pairs |

AdamW: lr `3e-4`, weight decay `1e-4`, 20 epochs, 48 pairs/batch, seeds 2039--2043. Checkpoint selection uses only early-role three-target BCE. No calibration or threshold exists in this H0 screen.

## Formal result

| C_CVMC comparison | Mean three-target weak AUROC | 95% CI | Positive seeds | Mean pair-margin success |
|---|---:|---:|---:|---:|
| vs B0 | +0.01351 | [+0.00780, +0.01968] | 4/5 | -0.00917 |
| vs B1_shuffle | +0.00083 | [-0.00477, +0.00643] | 2/5 | +0.00000 |

The small B0 direction is mostly MFT (+0.02876); VFT is +0.00852 and ISRJ is +0.00326. Against B1_shuffle, MFT/VFT/ISRJ are +0.00553/+0.00129/-0.00434, respectively. Therefore the shared-nuisance pairing information itself has no stable incremental effect.

The safety gate also fails: compared with B0, all-positive AUROC deteriorates in every seed for VFT (about -0.076 to -0.131) and ISRJ (about -0.083 to -0.146). The B1_shuffle safety comparison also fails.

## Interpretation and stop rule

The pair contract proves that the data contain exact nuisance-matched interventions. The failure means that compressing this intervention into a direct target-logit margin does not transfer to safe single-view weak-evidence ranking; shuffled margin supervision produces essentially the same result. No margin, weight, width, nuisance-combination, weak-offset, threshold, or fusion search is permitted on H0e identities. Any future counterfactual study needs a materially different representation-level mechanism and a new identity-isolated source.

Machine-readable evidence: `outputs/tapc/route_a_h0e_counterfactual_visibility_margin_formal/route_a_h0e_counterfactual_visibility_margin_summary.json`; audit: `outputs/tapc/route_a_h0e_counterfactual_visibility_generation/identity_audit.json`.
