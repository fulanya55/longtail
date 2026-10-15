# Route-A P4：严格反事实困难负样本 preflight（2026-09-01）

## 状态

**completed（机制可执行性）；partial（正式泛化证据不足）。** 使用独立生成的派生数据，不修改九类源数据，不读取 test 或 locked test。

## 数据构造

由 MATLAB 生成器构造 6 组组合对：

`FMB/AMA`、`SawFM/SinFM`、`MFT/ISRJ`、`FMB+MFT/FMB+ISRJ`、`SawFM+MFT/SawFM+ISRJ`、`FMB+SawFM/FMB+SinFM`。

每组 64 个 pair。pair 两侧使用相同 `sample_seed`，因此共享参考信号和接收机噪声流，同时只改变活动组合；每组前 48 个 pair 用于拟合、后 16 个 pair 用于验证。波形输入为真实 I/Q `[2,3000]`。

## 模型与损失

- 固定 F2_DFMS_Full/dfms_iq，257,511 参数；从 C1 F2 共享特征初始化；
- `F2_BCE`：两侧样本的平均 BCE；
- `F2_BCE_counterfactualHN`：BCE + `0.1 × hinge`，对组合对称差集上的 logit 使用 margin=0.25；
- AdamW `lr=1e-4`、`weight_decay=1e-4`，batch=16，4 epochs，梯度裁剪 5.0；seeds 2039/2040/2041。

## 结果（3 seed 均值）

| 变体 | 验证 Macro-AUPR | 验证 Macro-F1 | 验证组合完全匹配 |
|---|---:|---:|---:|
| F2+BCE | 0.999833 | 0.972648 | 0.932292 |
| F2+BCE+counterfactual HN | **0.999905** | **0.976545** | **0.939236** |

单 seed 组合率：BCE 为 1.0000/0.8906/0.9063，counterfactual HN 为 1.0000/0.8958/0.9219。HN 在两个不稳定 seed 上均有提升。

## 逐类错误边界

AMA、SawFM、FMB 的 recall 基本为 1，FPR 接近 0。seed 2040 的 ISRJ recall 为 0.896 → 0.917，seed 2041 为 0.708 → 0.729；但 MFT 在 seed 2040 仍为 0.667，说明小型 pair 集对 MFT 的覆盖不足。seed 2041 的 SinFM recall 为 0.875 → 0.969，同时出现 FMB FPR 0.009，需在正式规模实验中复核。

## 判定

该 preflight 证明：

1. 生成器能够构造共享 sample_seed 的组合反事实对；
2. 有界 matched-HN 能在小型严格 pair 集上改善组合率和 Macro-F1；
3. 机制方向值得进入正式 P4，但当前样本规模、组合覆盖和 seed 数量不足以冻结创新点。

不能据此宣称困难负样本创新已成立，也不能进入 P5 深度专家。

## 正式 P4 进入条件

- 扩展到至少 20 个组合对、每对不少于 256 个 pair；
- 增加单标签/混合标签、只去除一个成分和组合不相交三类反事实；
- 训练/验证按 pair group 隔离，补齐 5 seeds；
- 重点报告 MFT、ISRJ、SawFM 的 recall/FPR、组合转移矩阵及跨功率开发性能；
- 若 matched-HN 不能在正式规模下同时降低困难负样本 FPR 且保持 recall，则停止 P4，不构造专家。

## 工件

- [生成脚本](E:\LJQ_code\CJ_Imbalance_0716\experiments\generate_route_a_p4_counterfactual_preflight.m)
- [训练脚本](E:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_p4_counterfactual_train.py)
- [派生数据](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p4_counterfactual_train_dev_20260901\route_a_p4_counterfactual_preflight.mat)
- [训练汇总](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p4_counterfactual_train_v2_20260901\p4_counterfactual_preflight_summary.json)
- [角色清单](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p4_counterfactual_train_v2_20260901\role_manifest.json)
- [终端回执](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p4_counterfactual_train_v2_20260901\terminal_receipt.json)

