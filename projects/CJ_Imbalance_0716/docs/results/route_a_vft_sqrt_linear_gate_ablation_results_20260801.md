# Route A VFT 平方根门与线性门消融结果

## 结论

平方根不确定性门并非 VFT 收益的必要来源。G0 平方根门与原 Q4 在五个 seed 的 VFT logit 均逐样本精确复现；但相对同容量线性门，VFT 弱 AUROC 差异仅 `+0.00010`，95% bootstrap CI 为 `[-0.00027,+0.00042]`。该形状不应作为论文创新点。

## G0 平方根门减 G1 线性门

| 指标 | 均值差 | 95% bootstrap CI |
|---|---:|---:|
| VFT AUPR | +0.00022 | [+0.00015, +0.00029] |
| VFT 弱召回 | -0.00287 | [-0.00787, +0.00405] |
| VFT 弱 AUROC | +0.00010 | [-0.00027, +0.00042] |
| VFT FPR | -0.00081 | [-0.00468, +0.00355] |

两种门都使用同一 4 维融合输入、同一 Ridge-STFT 分数、相同 `C=0.1`、弱正权重 4、fit/calibration/threshold 角色协议和 FPR 上限。非 VFT logit 精确不变，test-v3 未使用。

## 解释

现有 VFT 证据的有效性来自对齐的 Ridge-STFT 物理分数及其与主分类器边界的互补，而非 `sqrt` 与线性不确定性形状之间的细微选择。后续方法描述应只把该门列为实现细节；若希望降低模型复杂度，可在独立新候选中使用线性门，但不得依据本消融反向宣称其更优。

## 产物

- 预注册：[route_a_vft_sqrt_linear_gate_ablation_plan_20260801.md](../plans/route_a_vft_sqrt_linear_gate_ablation_plan_20260801.md)
- 实现：`experiments/run_route_a_vft_sqrt_linear_gate_ablation.py`
- 汇总：`outputs/tapc/route_a_vft_sqrt_linear_gate_ablation/route_a_vft_sqrt_linear_gate_ablation_summary.json`
