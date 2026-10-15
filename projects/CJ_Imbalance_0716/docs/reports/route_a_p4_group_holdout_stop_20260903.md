# Route-A P4：严格反事实组合组留出停止报告（2026-09-03）

## 状态

**completed；failed gate；当前 matched-HN 路线停止。** 结果为开发集证据，未读取普通 test 或 locked test。

## 协议修正

首次 20 组、256 pair 的“正式”运行按同一 pair group 内重复序号切分，组合对同时出现在拟合与验证中，组合率接近 1。该运行保留在 `route_a_p4_counterfactual_formal_train_20260902`，标记为**协议无效/组合泄漏**，不用于结论。

修正后，20 个组合对按 group 整体划分：group 1–15 拟合（3,840 pair），group 16–20 验证（1,280 pair），拟合/验证组合对交集为 0。每个 pair 两侧共享 sample_seed 和接收机噪声，仅改变活动组合。比较 BCE 与 `BCE+0.1×counterfactual hinge`，margin=0.25，F2 参数量 257,511，5 seeds，3 epochs。

## 五种子结果

| 方法 | Macro-AUPR | Macro-F1 | 组合完全匹配 |
|---|---:|---:|---:|
| F2+BCE | 0.997101 | **0.654053** | 0.174531 |
| F2+BCE+counterfactual HN | **0.997118** | 0.653255 | **0.175547** |

组合率配对差值为 `[+0.00195,−0.00078,+0.00391,+0.00156,−0.00156]`，均值 +0.00102，bootstrap 95% CI `[-0.00070,+0.00266]`，Wilcoxon 单侧 `p=0.21875`。Macro-F1 均值下降 −0.00080，CI `[-0.00367,+0.00202]`。没有可靠统计证据支持 HN 改善。

## 失败图谱

组合组留出后 seed 间波动极大，BCE 组合率为 0.7066/0.0246/0.0820/0.0410/0.0184。SawFM、ISRJ、SinFM 的 recall 在多个 seed 明显下降；HN 没有稳定恢复这些标签。AUPR 仍很高而 0.5 阈值下组合率很低，说明未见组合的排序能力与决策边界存在严重脱节。

## 根因判断

1. 之前的近乎满分主要来自同组合对重复切分，不能代表组合泛化；
2. hinge 只要求 pair 内对称差标签 logit 排序，无法约束全局组合空间中的校准和阈值边界；
3. 20 组 pair 的训练覆盖仍不足以支撑 5 个完全未见组合对；
4. 当前失败属于机制/泛化失败，不是 GPU、I/O 或数值错误；
5. 继续调 margin、HN 权重或阈值会构成对开发验证的反复适配，不再盲目扩展。

## 停止决策

P4 matched-HN 在近似配对、严格小型 preflight、正式组合组留出三个层级上均未形成稳定、可泛化的门控证据。根据“同一机制连续失败不继续扩展”的规则，当前 HN 路线标记为**stopped**。P5 深度专家仍不允许启动。

后续如果重新开启 P4，必须改变研究假设而非调参：例如直接学习组合条件原型、显式校准未见组合边界，或生成覆盖更广的成分去除图；这些属于新路线，需重新预注册。

## 工件

- [严格组留出汇总](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p4_counterfactual_group_holdout_20260903\p4_counterfactual_preflight_summary.json)
- [角色清单](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p4_counterfactual_group_holdout_20260903\role_manifest.json)
- [五种子统计](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p4_counterfactual_group_holdout_20260903\p4_group_holdout_statistics.json)
- [终端回执](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p4_counterfactual_group_holdout_20260903\terminal_receipt.json)
- [生成脚本](E:\LJQ_code\CJ_Imbalance_0716\experiments\generate_route_a_p4_counterfactual_preflight.m)
- [训练脚本](E:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_p4_counterfactual_train.py)

