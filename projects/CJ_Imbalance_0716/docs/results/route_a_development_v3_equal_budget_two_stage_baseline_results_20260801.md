# Route A development-v3 等总预算两阶段解耦基线结果

## 结论

等总预算的两阶段解耦不构成当前候选提升的充分解释。`Focal24 -> Frozen Linear BCE12` 的总训练预算与单阶段 `Focal-36` 完全相同，但整体排序和三目标平均弱分量表现均未改善。

## 公平协议

- 五个 seeds：2039--2043；同一 development-v3 身份角色、F2 初始化、partial shared scope、RawSaw SawFM 路由、Platt 校准和 FPR 约束阈值；
- 单阶段对照：Focal 36 epoch；
- 两阶段：Focal 表征 24 epoch，完全冻结 `forward_features`，随机初始化 `Linear(128,6)` 的 BCE 分类头 12 epoch；总计 36 epoch；
- 两阶段的两个阶段均运行完整预算，不通过 early-stop 少训练；全部 checkpoint、history、预测、校准、阈值、哈希与 receipt 已保存；
- `locked_test_used=false`，`test_v3_used=false`。

## 结果：two-stage 减 Focal-36

| 指标 | 均值差 | 95% bootstrap CI | 解释 |
|---|---:|---:|---|
| macro-AUPR | -0.00169 | [-0.00587, +0.00288] | 无整体排序收益 |
| macro-F1 | +0.00436 | [+0.00002, +0.00939] | 仅阈值点的轻微变化 |
| 三目标平均弱召回 | -0.00570 | [-0.01575, +0.00586] | 无弱分量收益 |
| 三目标平均弱 AUROC | -0.00601 | [-0.01334, +0.00126] | 无排序收益 |
| 三目标平均 FPR | +0.00134 | [-0.00129, +0.00409] | 没有安全性收益 |
| ISRJ AUPR | +0.00689 | [-0.00458, +0.01595] | 不稳定，CI 跨 0 |
| ISRJ 弱召回 | +0.01960 | [+0.00609, +0.03295] | 局部改善但未到 `+0.030` 门槛 |
| ISRJ 弱 AUROC | +0.00695 | [-0.00193, +0.01541] | CI 跨 0 |

## 对创新主张的影响

先前“Focal-36 后再额外训练冻结线性头”确实能显著改善部分指标，但其额外预算使解释不公平。现在的等总预算版本没有复制这种整体收益，因此当前统一候选的开发增益不能仅归因于“多一个训练阶段”或“额外 12 epoch 的线性头优化”。这仍不是对物理专家的单独证明；其直接证据仍需要来自专家移除、容量与破坏对照，以及外部/OOD 验证。

## 产物

- 预注册：[route_a_development_v3_equal_budget_two_stage_baseline_plan_20260801.md](../plans/route_a_development_v3_equal_budget_two_stage_baseline_plan_20260801.md)
- 实现：`experiments/run_route_a_development_v3_equal_budget_two_stage_baseline.py`
- 汇总：`outputs/tapc/route_a_development_v3_equal_budget_two_stage_baseline/route_a_development_v3_equal_budget_two_stage_baseline_summary.json`
- 单元测试：`tests/test_route_a_development_v3_equal_budget_two_stage_baseline.py`
