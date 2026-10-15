# Route-A P2a：功率均衡采样实验（2026-08-27）

## 目的

检验在 C1 纯六类和 F2 主干固定的条件下，将 10–50 dB train-only 样本等功率混合训练，是否能够改善跨功率性能。

## 协议修正记录

第一次运行错误地将 30 dB-only 对照缩减为 456 个样本，而功率均衡使用 456×9 个样本，预算不等价；该运行保存于 `outputs/route_a_p2a_power_balanced_20260827`，标记为 protocol-invalid。修正版使用 30 dB-only 4,096 个样本，功率均衡 456×9=4,104 个样本，使用相同 F2、BCE、AdamW、batch=16、2 epochs、split_seed=260827 和 3 seeds。

## 修正版结果

| 训练策略 | 验证 Macro-AUPR | seed std | 验证组合准确率 | 10 dB 开发 AUPR | 40 dB 开发 AUPR | 50 dB 开发 AUPR |
|---|---:|---:|---:|---:|---:|---:|
| F2 30 dB-only | **0.999186** | 0.000213 | **0.933919** | 0.863508 | 0.922967 | 0.825216 |
| F2 power-balanced | 0.869968 | 0.021446 | 0.106283 | 0.797775 | 0.785554 | 0.768232 |

## 分析

功率均衡训练在低预算下明显损害了 30 dB 验证和组合识别，且没有带来足够的低功率收益。可能原因是：

1. 2 epoch 的更新预算不足以同时拟合九个功率域；
2. 功率变化与可见性变化混在一起，简单重采样没有显式建模可见性；
3. 30 dB-only 模型具有较强的同域先验，混合功率后出现优化冲突。

该结果不是“功率鲁棒性不可行”的证据，而是“不应直接用等功率混合替代主训练”的证据。P2a 不通过晋级门控，保留为负向结果。

## 下一步

进入 P2b：保持 30 dB 主训练稳定，采用小比例功率扰动或 curriculum power mixing，并加入有界可见性辅助任务；先做 3-seed 小预算筛选，只有同时保持 30 dB 组合准确率和改善至少两个困难功率层，才进入 5-seed 确认。

## 工件

- [修正版汇总](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p2a_power_balanced_v2_20260827\p2a_power_balanced_summary.json)
- [修正版角色清单](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p2a_power_balanced_v2_20260827\role_manifest.json)
- [修正版终端回执](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p2a_power_balanced_v2_20260827\terminal_receipt.json)
- [执行脚本](E:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_p2a_power_balanced.py)
