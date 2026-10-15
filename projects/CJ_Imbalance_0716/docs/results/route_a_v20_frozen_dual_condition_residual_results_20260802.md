# Route A v20: frozen dual-condition residual result

## Decision

**Stop the width-times-morphology product residual.** It lowers Q4 VFT hard-negative FPR, but fails its required comparison against morphology-only and violates weak-VFT recall safety. The gate is retained as preregistered.

## Cross-source protocol

v20 is a new train-role-only source (base seed `580000000`, 5,040 unique receiver identities; SHA-256 `3d996a0ddf8fce7481d10a46cc96237d9adb36b59f325b8bc526519f0c195529`). Fit/selection/evaluation roles are 3,024/1,008/1,008 identities; evaluation has 256 VFT-absent/ISRJ-present hard negatives.

All source models were frozen: Q4 from the frozen shared-policy manifest, v16 W-real predicted width, and v19 M-real nonlinear morphology. v20 did not read v17/v18, retrain any source model, recalibrate Q4, or select a threshold. Only each rule's scalar `c` was selected on fit identities under the stated overall VFT recall budget. The original Q4 calibration and thresholds applied unchanged. All non-VFT logits were exact; no locked-test, test-v2, test-v3 or v20 test role was used.

## Five-seed held-out result

| Comparison | Hard-negative VFT FPR delta | VFT positive recall delta | weak-VFT recall delta | Conclusion |
| --- | ---: | ---: | ---: | --- |
| Dual minus Q4 | -0.0344, CI [-0.0383, -0.0305] | -0.0035 | -0.0162, CI [-0.0216, -0.0108] | FPR passes; weak safety fails |
| Dual minus width-only | -0.0039, CI [-0.0102, +0.0023] | +0.0047 | +0.0126 | fails required <= -0.010 |
| Dual minus morphology-only | +0.0297, CI [+0.0164, +0.0438] | +0.0066 | +0.0234 | morphology-only is materially better FPR |

Dual VFT AUPR increased by `+0.0149` and non-VFT ISRJ/SawFM outputs stayed exact. These positives do not override the failed primary control and weak safety gates.

## Mechanism diagnosis

The morphology score is the operative second condition. Multiplying it by width discards morphology-negative hard cases at lower predicted widths, so the product has less corrective coverage than morphology-only. In addition, the fit selector protected overall VFT recall but not the within-label weak VFT subgroup: a small global loss concealed a consistent `-0.0162` weak loss.

This is direct evidence for the project’s visibility-within-label premise, but it rejects the width-product architecture. The next permissible hypothesis is not another product or threshold sweep: on another fresh source, choose the morphology residual coefficient under *both* global and weak-VFT recall constraints, and compare that visibility-balanced selector to the global-only selector at matched scalar capacity.

## Artifacts

- Plan: `docs/plans/route_a_v20_frozen_dual_condition_residual_plan_20260802.md`
- Generator/audit/runner: `experiments/generate_route_a_v20_frozen_dual_condition_residual.m`, `experiments/audit_route_a_v20_frozen_dual_condition_residual.py`, `experiments/run_route_a_v20_frozen_dual_condition_residual.py`
- Formal artifacts: `outputs/tapc/route_a_v20_frozen_dual_condition_residual/`

