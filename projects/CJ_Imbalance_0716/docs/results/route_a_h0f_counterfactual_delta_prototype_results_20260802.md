# Route A H0f: Counterfactual Delta Prototype Result

## Decision

**Stop `counterfactual_delta_prototype`.** Binding a nuisance-matched counterfactual feature difference to a true target prototype did not improve weak-component ranking over either ordinary BCE or the cyclic target-binding control. It also failed the all-positive safety and the prerequisite that the held-out difference itself be target identifiable. It must not enter the six-label model, calibration, threshold selection, fusion, or test-v3.

## Data, isolation and frozen boundary

- Fresh train-only source: `C:\LJQ_code\干扰数据\复合干扰数据0802_counterfactual_delta_h0f\counterfactual_delta_pairs_train.mat`; SHA-256 `ef1c47c8794c42085fbe59b5ed063c1caadb72601ae40e75f2997189a3b56cf8`.
- It contains 2,400 pair identities / 4,800 views, with 800 pairs for each of MFT, VFT and ISRJ. Within a pair, sample seed, receiver-noise seed, retained components and parameters are identical; only the target is toggled. The target is -12 dB relative power at JNR 20 dB.
- Fit/early/evaluation roles are 1,440/480/480 pairs, with 480/160/160 pairs per target. H0f receiver identities span `948677742--948680141` and have zero intersection with each live historical train source. The preserved H0e audit range is `938677742--938680141`, also with zero overlap; its source MAT is no longer live, so this latter comparison is explicitly recorded metadata rather than a reopened source-file audit.
- Historical locked test, test-v2 and test-v3 were not generated, opened, analyzed or used. All recorded flags are false.

## Fixed comparison

Every arm has the same 20,729-parameter `CounterfactualDeltaPrototypeProbe`: raw I/Q temporal encoder, six-logit global head, 16-D projection, three 16-D prototypes and bounded target-logit residual. The inference path uses one view only.

| Arm | Additional loss |
| --- | --- |
| B0 | six-label BCE only |
| B1 | BCE plus 0.20 delta CE with fixed cyclic wrong binding |
| C_CDP | BCE plus 0.20 delta CE with true target binding |

The normalized pair difference is `e(x_positive)-e(x_negative)`, and its prototype CE uses temperature 0.10. AdamW uses lr `3e-4`, weight decay `1e-4`, 20 epochs, 48 pairs/batch and seeds 2039--2043. Early selection uses only the early-role target BCE. There is no threshold, calibration, six-label fusion or test evaluation.

## Formal five-seed result

All values below are C_CDP minus the matched control on held-out evaluation pairs.

| Comparison | Mean three-target weak AUROC | 95% CI | Positive seeds | Per-target mean (MFT / VFT / ISRJ) |
| --- | ---: | --- | ---: | --- |
| vs B0 | -0.00185 | [-0.00890, +0.00590] | 2/5 | -0.00925 / +0.00421 / -0.00051 |
| vs B1 cyclic binding | +0.00274 | [-0.00245, +0.00668] | 4/5 | +0.00284 / +0.00029 / +0.00510 |

The required `+0.015` effect and positive lower confidence bound fail against both controls. The C_CDP target-binding accuracy is `0.54750` (95% CI `[0.52208, 0.57208]`), below the preregistered `0.60` gate. This means the feature-difference directions are not sufficiently target-specific even before asking them to improve single-view classification.

Safety also fails: relative to B0, all-positive AUROC changes are negative for VFT and ISRJ in every seed (roughly -0.014 to -0.058 and -0.074 to -0.089, respectively). Relative to the cyclic-binding control, at least one all-positive target is below the allowed -0.010 in multiple seeds. Parameter count, initialization, roles and the sealed-test contract all match.

## Interpretation and stop rule

H0f is stricter than H0e because it tests whether the exact paired intervention supports a target-discriminative **representation**, rather than only a direct target-logit margin. The result shows that the added target binding signal is weak: true binding is only modestly above chance and is not materially better than the deliberately wrong cyclic binding for downstream weak ranking. Thus pair differences in this encoder are not a deployable single-view local-evidence representation.

No source, projection width, prototype temperature, delta-loss weight, nuisance combination, target offset, residual magnitude, threshold or fusion may be tuned under H0f. A future study must use a physically or architecturally different observable and a new identity-isolated source.

Machine-readable result: `outputs/tapc/route_a_h0f_counterfactual_delta_prototype_formal/route_a_h0f_counterfactual_delta_prototype_summary.json`.
Identity audit: `outputs/tapc/route_a_h0f_counterfactual_delta_generation/identity_audit.json`.
