# Route A 最终候选冻结组件消融结果

## 结论

在不训练、不重新校准、不重新选阈值的条件下，完整候选 `H1+B2+MFT shared-decision+VFT Ridge-STFT` 的三个目标组件均显示出标签特异的决策级贡献。

- VFT Ridge-STFT：相对同容量与 role-wise shuffled 控制，VFT 弱 AUROC 分别提高 `+0.01706` 与 `+0.01757`，弱召回分别提高 `+0.01871` 与 `+0.01768`；
- MFT guard：用 Q1 父模型替换后，完整候选减少 MFT hard-FPR `0.02658`、order-5 FPR `0.02800`，并提高 MFT 弱 AUROC `0.01640`，代价是 MFT 弱召回 `-0.01265`；
- ISRJ B2：用 Q0 父模型的 ISRJ 决策替换后，完整候选的 ISRJ AUPR、弱召回与弱 AUROC 分别为 `+0.02211`、`+0.10672`、`+0.08313`，是目前三组件中最强的独立增益。

## 完整候选减去单组件 lesion（五种子均值）

| lesion | macro-AUPR | macro-F1 | 目标弱召回 | 目标弱 AUROC | 目标 FPR | 关键标签结论 |
|---|---:|---:|---:|---:|---:|---|
| 用同容量 VFT 分类头替代 Ridge | +0.00223 | +0.00330 | +0.00624 | +0.00569 | -0.00087 | VFT AUPR `+0.01337`、弱 AUROC `+0.01706` |
| 用打乱 Ridge 的 VFT 分类头替代 | +0.00232 | +0.00328 | +0.00589 | +0.00586 | -0.00063 | VFT AUPR `+0.01395`、弱 AUROC `+0.01757` |
| 移除 MFT guard | +0.00069 | +0.00188 | -0.00422 | +0.00547 | -0.01133 | 安全性提升与有限召回损失的明确权衡 |
| 移除 ISRJ B2 | +0.00369 | +0.00580 | +0.03557 | +0.02771 | +0.00077 | ISRJ 弱召回 `+0.10672`、弱 AUROC `+0.08313` |

正值均表示完整候选优于对应 lesion；MFT 行的目标弱召回为负，说明 guard 的价值是降低高阶难负误报和提高排序，而不是抬高 MFT 召回。

## 解释边界

这是**冻结决策级消融**：每次仅替换一个已保存的目标 logit 或目标决策，并继续使用该路径预先冻结的校准器/阈值。它排除了“其他标签同步变化”造成的归因混淆，但不替代从头重训的模块删除实验。development-v12 已被本项目另一线路使用，故本结果只能作为当前线路的后验机制诊断，不能作为 project-globally fresh 确认，也不能用于选择新候选。

## 产物

- 预注册：[route_a_final_candidate_frozen_component_ablation_plan_20260801.md](../plans/route_a_final_candidate_frozen_component_ablation_plan_20260801.md)
- 实现：`experiments/audit_route_a_final_candidate_frozen_component_ablations.py`
- 汇总：`outputs/tapc/route_a_final_candidate_frozen_component_ablations/route_a_final_candidate_frozen_component_ablations_summary.json`
- `locked_test_used=false`；`test_v3_used=false`。
