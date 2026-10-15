# Route-A P3c：同身份跨功率一致性正则（2026-08-29）

## 状态

**completed（3-seed 开发筛查）；部分支持，不冻结。** 本轮使用九个 train 功率视图中 receiver_noise_seed 完全配对的性质，在 30 dB 拟合集上同时读取同一身份/同一组合的 10 dB 和 50 dB 信号，对输出概率施加有界一致性约束。普通 test 与 locked test 均未打开。

## 协议

- 30 dB train 纯六类：4,096 拟合、2,048 验证；功率开发集为各功率 train-only 未见身份，每功率 256 个样本。
- 骨干固定 F2_DFMS_Full/dfms_iq，参数 257,511；输入 `[B,2,3000]`。
- 对照：`F2_BCE_30dB`；实验：`F2_BCE_power_consistency`。
- 实验损失为 30 dB BCE 加 `λ=0.1` 的 30/10/50 dB sigmoid 输出一致性 MSE；AdamW `lr=1e-4`、`weight_decay=1e-4`、batch=16、2 epochs、梯度裁剪 5.0；seeds 2039/2040/2041。
- 所有配对均按 receiver_noise_seed 和跨功率相同样本顺序核验，不使用 test 统计量或阈值搜索。

## 结果（3 seed 均值）

| 变体 | 验证 Macro-AUPR | 验证 Macro-F1 | 验证组合率 | 35 dB AUPR | 40 dB AUPR | 45 dB AUPR | 50 dB AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| F2+BCE（30 dB） | **0.9999990** | 0.998958 | 0.993652 | 0.999862 | 0.990615 | 0.947651 | 0.879065 |
| F2+BCE+power consistency | 0.9999989 | **0.999172** | **0.994954** | **0.999877** | **0.992941** | **0.961013** | **0.886896** |

低功率 10 dB AUPR 为 0.856894 → 0.855333，20 dB 为 0.920893 → 0.920248，说明一致性项主要改善中高功率，不能称为完整功率鲁棒解决方案。组合率在 35 dB 从 0.6823 提升到 0.7031、40 dB 从 0.3034 提升到 0.3216；10–20 dB 没有改善。

## 五种子统计确认

五种子（2039–2043）后，验证组合率配对差值为 `[−0.00049, 0, +0.00488, +0.00537, +0.00195]`，均值 **+0.00234**，seed-bootstrap 95% CI `[+0.00020,+0.00449]`，Wilcoxon 单侧 `p=0.0721`。Macro-F1 均值差为 +0.000386，95% CI `[+0.000033,+0.000739]`，同样 `p=0.0721`；Macro-AUPR 已饱和，差值约 `6.6e-8`。45 dB AUPR 均值提升 +0.00740，但 CI 为 `[-0.00158,+0.01416]`，`p=0.0938`；10 dB AUPR 均值下降 −0.00373，CI 全为负。样本数仅 5 个，统计结果只能作为开发证据，不能宣称显著创新。

## 错误图谱与门控

实验没有出现单标签独占预测，但困难功率层的 FPR 仍然较高。例如 50 dB 的 ISRJ FPR 三个 seed 分别为 0/0.227/0.121（实验组），基线为 0/0.235/0.098；改善不一致。P3c 因此判定为**部分支持**：它提供了“同身份跨功率一致性”可行性的证据，但还没有解决低功率边界和组合误报。

## 实现失败记录

初次运行曾误将 10 dB 路径作为 30 dB 验证路径，得到 Macro-AUPR 约 0.856、组合率约 0.292；该运行已中止并保留在 `outputs/route_a_p3c_power_consistency_20260829`，不纳入结论。修正后的有效运行位于 `..._v2_20260829`，验证路径和 role manifest 已核对。

## 工件

- [执行脚本](E:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_p3c_power_consistency.py)
- [有效汇总](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p3c_power_consistency_v2_20260829\p3c_power_consistency_summary.json)
- [五种子统计](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p3c_power_consistency_5seed_20260829\p3c_five_seed_statistics.json)
- [角色清单](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p3c_power_consistency_v2_20260829\role_manifest.json)
- [终端回执](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p3c_power_consistency_v2_20260829\terminal_receipt.json)

## 下一步

五种子完成后，power-consistency 可保留为“中高功率候选”，但不满足全功率晋级门槛。下一步优先尝试只在中高功率启用的一致性权重、低功率可见性/困难负样本约束，并预先固定门控规则；同时继续处理 P4 数据身份/反事实字段缺失问题。在 P4 严格困难负样本证据形成前，不构造深度专家模块。
