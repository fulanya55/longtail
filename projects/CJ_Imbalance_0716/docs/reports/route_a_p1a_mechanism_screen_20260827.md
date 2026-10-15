# Route-A P1a：F2 特征机制结构对照（2026-08-27）

## 实验边界

本轮只使用 C1 纯六类（FMB、AMA、SawFM、SinFM、MFT、ISRJ），排除 SN、VFT、RangeFakeTarget。拟合和开发均来自 train MAT：30 dB train 中固定 4,096 个拟合样本、2,048 个验证样本；各功率 train MAT 使用未参与拟合身份的 512 个样本作功率开发诊断。没有读取普通 test 或 locked test。

所有变体使用相同的 I/Q 输入、BCEWithLogitsLoss、AdamW（lr=1e-4，weight_decay=1e-4）、batch=16、2 epochs、split_seed=260827 和 seeds 2039/2040/2041。旧 F2 checkpoint 只加载名称/形状匹配的共享参数；变体专用不兼容参数保持随机初始化。

## 变体结果

| 变体 | 参数量 | 验证 Macro-AUPR | seed std | 验证组合准确率 | 10 dB 开发 AUPR | 50 dB 开发 AUPR |
|---|---:|---:|---:|---:|---:|---:|
| F2_anti_alias | 257,511 | **0.999788** | 0.000063 | 0.967611 | 0.858371 | 0.854283 |
| F2_full | 257,511 | 0.999736 | 0.000070 | 0.960124 | 0.857773 | 0.849990 |
| F2_no_phase | 240,806 | 0.999715 | 0.000064 | 0.962240 | 0.855186 | 0.836184 |
| F2_evidence_block | 313,191 | 0.999686 | 0.000096 | 0.961589 | 0.856823 | 0.854301 |
| F2_cross_stage | 310,376 | 0.999681 | 0.000165 | 0.962565 | 0.857161 | 0.846708 |

单次 batch-1 延迟测量约为：F2_full 4.145 ms、no_phase 4.061 ms、anti_alias 4.404 ms、cross_stage 5.281 ms、evidence_block 4.115 ms；该测量受 GPU 状态影响，仅作成本记录。

## 解释

1. 抗混叠下采样是当前探索集的数值第一名，且 10/50 dB 功率开发指标略高于 F2_full，但相对增益很小，3 个 seed、2 个 epoch 不能支持“抗混叠机制已经证明有效”。
2. 去相位后参数量从 257,511 降到 240,806，性能仅轻微下降；这说明当前 30 dB 小预算验证集并不能单独证明相位通道的增益，必须依赖困难功率和完整机制消融继续验证。
3. cross-stage 和 evidence-block 增加了参数与计算量，却未超过 F2_full，当前不应作为主路线晋级。
4. 本轮没有普通 test/locked test 证据，所有结论均为 train-only development observation。

## 门控结论

- F2_full：继续保留为连续性主基线。
- F2_anti_alias：进入下一轮 5-seed 结构确认候选，但只有在功率鲁棒指标和成本审计通过后才可冻结。
- F2_no_phase：保留为相位机制消融，不作为主模型。
- F2_cross_stage、F2_evidence_block：当前停止扩展，不删除结果。

下一步为 P1b：实现 I/Q 两路分组卷积与参数量匹配的实值卷积对照，仍保持数据、split、loss、optimizer、预算和阈值不变。

## 工件

- [P1a 汇总](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p1a_mechanism_screen_20260827\p1a_mechanism_summary.json)
- [角色清单](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p1a_mechanism_screen_20260827\role_manifest.json)
- [终端回执](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p1a_mechanism_screen_20260827\terminal_receipt.json)
- [P1a 脚本](E:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_p1_mechanism_screen.py)
