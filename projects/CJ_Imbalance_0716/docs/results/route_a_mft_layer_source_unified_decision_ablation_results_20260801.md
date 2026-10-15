# Route A MFT expert layer-source unified-decision ablation result

## Protocol

Global, final temporal, stage2 and stage3 frozen MFT probes were separately
inserted into the same stored Q4 decision as a zero-parameter bounded residual
`0.5 * sqrt(4p(1-p)) * tanh(z)`. Each arm used fit-only score standardization,
calibration-role Platt scaling, threshold-role FPR selection and early-role
evaluation. Five non-MFT logits were exactly preserved in every arm. No locked
test, test-v2 or test-v3 was used.

## Result

| Source | MFT weak-negative AUROC delta vs Q4 [95% CI] | AUPR delta [95% CI] | weak-AUROC positive seeds | Gate |
|---|---:|---:|---:|---|
| Global | +0.003745 [+0.002888,+0.004602] | +0.001258 [+0.000973,+0.001548] | 5/5 | fail (<0.005) |
| Final temporal | **+0.006213 [+0.004522,+0.007964]** | **+0.002294 [+0.001430,+0.003037]** | 5/5 | **pass** |
| Stage2 | +0.000855 [-0.001463,+0.003173] | +0.000382 [-0.000184,+0.000947] | 3/5 | fail |
| Stage3 | +0.004779 [+0.003581,+0.006461] | +0.001590 [+0.001310,+0.001864] | 5/5 | fail (<0.005) |

Only final temporal clears the pre-registered `+0.005` MFT weak-AUROC gate;
the other layers are not eligible expert sources under this fixed residual
rule. This is direct evidence against the idea that merely selecting an earlier
or more physical multiscale layer produces a better weak-component expert.

It remains a development mechanism result. Its effect is local, it does not
resolve the full unified-model/OOD gaps, and it cannot authorize test-v3.

Artifacts: `docs/plans/route_a_mft_layer_source_unified_decision_ablation_plan_20260801.md`,
`experiments/run_route_a_mft_layer_source_unified_decision_ablation.py`, and
`outputs/tapc/route_a_mft_layer_source_unified_decision_ablation/route_a_mft_layer_source_unified_decision_ablation_summary.json`
(SHA-256 `C76F44CAF3519876B7D49ADF2FD3B017535A023C34E079DE3D9ED4208E7E9615`).
