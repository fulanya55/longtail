# Route-A P2b：渐进式邻近功率混合（2026-08-27）

## 目的与协议

P2a 的全功率等权采样在等预算下未通过。本轮改为 curriculum mixing：75% 30 dB、12.5% 25 dB、12.5% 35 dB，总拟合预算 4,096 个样本；与 30 dB-only 对照共享 F2_DFMS_Full、BCEWithLogitsLoss、AdamW（1e-4/1e-4）、batch=16、2 epochs、split_seed=260827 和 3 个 seed。所有开发样本来自 train MAT 未见身份，未读取普通 test 或 locked test。

## 结果

| 策略 | 验证 Macro-AUPR | seed std | 验证组合准确率 | 10 dB AUPR | 25 dB AUPR | 35 dB AUPR | 50 dB AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| F2 30 dB-only | **0.999241** | 0.000169 | **0.934896** | 0.858850 | **0.978945** | **0.990172** | **0.814324** |
| F2 curriculum (30/25/35) | 0.997487 | 0.001056 | 0.797852 | 0.857569 | 0.970282 | 0.984547 | 0.811656 |
| F2 全功率等权（P2a 修正版） | 0.869698 | 0.021712 | 0.106283 | 0.796746 | 0.836762 | 0.830425 | 0.758751 |


## 结论

1. 渐进式邻近功率混合比全功率等权稳定得多，但在当前训练预算下仍损害 30 dB 组合识别，且未改善困难功率层。
2. 简单改变采样比例不能证明功率鲁棒创新有效；继续提高低功率采样权重可能进一步造成优化冲突。
3. P2b 不通过晋级门控。下一步应固定主训练分布，显式引入可见性辅助任务或有界条件残差，并对低/高功率分别报告可见性分层指标。

## 工件

- [P2b 汇总](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p2b_curriculum_20260827\p2a_power_balanced_summary.json)
- [P2b 角色清单](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p2b_curriculum_20260827\role_manifest.json)
- [P2b 终端回执](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p2b_curriculum_20260827\terminal_receipt.json)
- [执行脚本](E:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_p2a_power_balanced.py)
