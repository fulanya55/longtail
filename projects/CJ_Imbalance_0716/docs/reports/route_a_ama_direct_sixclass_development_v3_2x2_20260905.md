# Route-A 新六类 development-v3：IF3 严格 2×2 长尾验证

## 1. 实验目的与证据边界

本阶段验证两个因素：标签长尾损失（普通 BCE vs Logit Adjustment）与组合采样（普通采样 vs 组合均衡采样）。四个条件均使用同一新六类 IF3 训练集、同一 F2_DFMS_Full backbone、同一优化器和训练预算，5 个独立种子。阈值只允许由 calibration 子集拟合；check 用于确认；development-test 只做一次冻结决策评估。

本阶段没有读取 locked-test，也没有修改冻结 E5。结果属于 fresh-sixclass development-v3 证据，不外推为最终泛化结论。

## 2. 数据审计

数据目录：`E:\论文相关\干扰数据\复合干扰数据0716\six_class_pure_label_longtail_v3_IF3_CF8_N8000_AMA`

- 类别：FMB、AMA、SawFM、SinFM、MFT、ISRJ；63 个组合完整覆盖。
- 训练集：8000，标签计数 `[6729, 2243, 3973, 3897, 4069, 3986]`，IF=3.0，组合 CF=7.8537，组合计数范围 41–322。
- 新鲜 development-test 池：6048，每个组合 96 个，IF=1、CF=1；按组合拆为 calibration/check/development-test，各 16/16/64 个组合样本（1008/1008/4032）。
- 训练 JNR 范围约 10–30 dB（均值约 19.95 dB），不是固定 30 dB 饱和协议。
- `receiver_noise_seed` 训练/测试交集为 0；训练和测试 SHA-256 已写入 manifest。
- 组合均衡条件使用每轮组合循环顺序，固定 8000 个更新样本；因为每组合样本有限，会产生全局重复率 0.38875，不能称为严格无放回采样，已作为成本/混杂项记录。

## 3. 固定网络与训练设置

- Backbone：冻结 F2_DFMS_Full（`dfms_iq`，width=16，dropout=0.2）初始化，只重新初始化/训练分类头及全模型参数；输入为归一化 `[B,2,3000]` I/Q。
- 损失：`BCEWithLogitsLoss`；Logit Adjustment 为在训练标签先验上加入 `log(p/(1-p))` 的逐标签 bias 后计算 BCE。
- AdamW，lr=1e-4，weight_decay=1e-4，batch=32，8 epochs，梯度裁剪 5.0。
- 种子：2050–2054；每个条件 5 次，最佳 checkpoint 按 calibration Macro-AUPR 选择。

## 4. 固定 0.5 阈值结果（5 种子均值±标准差）

| 条件 | check Macro-F1 | check 组合准确率 | dev-test Macro-F1 | dev-test 组合准确率 | dev-test Macro-AUPR | dev-test Hamming |
|---|---:|---:|---:|---:|---:|---:|
| BCE + ordinary | 0.9126±0.0071 | 0.5984±0.0292 | 0.9138±0.0075 | 0.6057±0.0265 | 0.9778 | 0.0847 |
| Logit + ordinary | 0.9184±0.0027 | 0.6312±0.0134 | 0.9185±0.0031 | 0.6339±0.0122 | 0.9775 | 0.0782 |
| BCE + balanced-cycle | 0.9214±0.0030 | 0.6298±0.0053 | 0.9205±0.0025 | 0.6343±0.0026 | 0.9762 | 0.0761 |
| Logit + balanced-cycle | 0.9168±0.0036 | 0.6159±0.0141 | 0.9154±0.0034 | 0.6181±0.0057 | 0.9761 | 0.0812 |

固定阈值下，三种改进相对 BCE+ordinary 的 dev-test 组合准确率增益分别为 +0.0282、+0.0286、+0.0124；但这不是纯粹的模型增益，因为 balanced-cycle 引入了 38.875% 的全局重复采样。

## 5. Calibration-only 阈值结果

在 calibration 上拟合逐类 Macro-F1 阈值或逐坐标组合最优阈值，在 check 和 dev-test 冻结使用：

| 条件 | dev-test F1-cal Macro-F1 | dev-test F1-cal 组合 | dev-test combo-cal Macro-F1 | dev-test combo-cal 组合 | rare 组合准确率（combo-cal） |
|---|---:|---:|---:|---:|---:|
| BCE + ordinary | 0.9256 | 0.6518 | 0.9243 | 0.6496 | 0.5606 |
| Logit + ordinary | 0.9244 | 0.6486 | 0.9239 | 0.6500 | 0.5668 |
| BCE + balanced-cycle | 0.9227 | 0.6313 | 0.9224 | 0.6370 | 0.5656 |
| Logit + balanced-cycle | 0.9235 | 0.6390 | 0.9226 | 0.6394 | 0.5702 |

相对固定 0.5，BCE+ordinary 的 dev-test 组合准确率提高约 +0.044（F1-cal）或 +0.044（combo-cal），Macro-F1 提高约 +0.012；Logit+ordinary 提高约 +0.015–0.016。相反，balanced-cycle 的校准后组合结果低于 BCE+ordinary，说明它在固定 0.5 下的优势主要来自决策偏置/重复采样，而非稳定的可分性增益。

逐类阈值的平均值显示明显的先验偏移：BCE+ordinary 的 F1 阈值约为 `[0.682, 0.159, 0.487, 0.343, 0.546, 0.441]`（顺序同类别），AMA 尾类需要显著降低阈值；这直接解释了固定 0.5 时尾召回不足和组合错误累积。

## 6. 尾类、头类与校准

- 固定 0.5 的 dev-test AMA（训练尾标签）Recall：BCE+ordinary 0.9208，Logit+ordinary 0.9537，BCE+balanced 0.9648，Logit+balanced 0.9855；但组合收益伴随不同程度的误报权衡。
- F1 校准后 AMA Recall 均约 0.977，说明大量差异来自阈值而不是排序能力。
- 固定 0.5 的 FMB（训练头标签）FPR：BCE+ordinary 0.1599，Logit+ordinary 0.0117，BCE+balanced 0.0137，Logit+balanced 0.0009；F1/组合校准后分别约 0.049/0.028/0.022/0.025。
- 温度缩放显著降低 ECE（约 0.0695→0.0435，BCE+ordinary），但在阈值仍为 0.5 时不改变 F1/组合准确率；因此温度解决的是概率校准，不是组合决策边界。

## 7. 阶段 A 判定与失败原因

1. 阶段 A 成功完成：四条件、5 种子、统一训练预算、独立 calibration/check/development-test、重复率和成本审计均有记录。
2. “长尾损失/组合采样形成稳定创新增益”未通过：校准后 BCE+ordinary 已达到 dev-test 组合约 0.650，Logit+ordinary 约 0.650，balanced-cycle 约 0.637–0.639；相对 BCE 的 5 种子配对差异不稳定或为负。
3. 当前主要瓶颈是决策校准与组合边界，而不是 AUPR 排序能力：四条件 Macro-AUPR 都约 0.976–0.978，而校准可将 BCE+ordinary 组合准确率提高约 4.4 个百分点。
4. balanced-cycle 的失败不能解释为“组合均衡无效”这一普遍结论：它不是严格无放回，38.875% 重复率造成有效样本量下降和训练分布改变；必须在真正独立、无放回 Anchor 上重做因果对照。

## 8. 下一道闸门

按原计划停止继续盲目调大损失权重，进入阶段 B：

1. 在新的身份划分上做 calibration/check 双分割的多目标阈值校准，并预先固定组合准确率、Macro-F1、尾 Recall、头 FPR 的约束。
2. 统计并挖掘一位差异困难负样本（优先 44→45、60→61、12→13 等实际高频转移），比较 BCE+校准与 BCE+困难负样本 margin。
3. 仅当困难负样本线在 ≥5 个新种子上改善 rare-combination accuracy 且不牺牲头类 FPR，才进入结构化组合解码器（classifier chain/标签图/energy joint decoder）。
4. 之后补充真正 IF≈1、CF≈1、无放回 Anchor，并完成 BCE、Logit Adjustment、采样、解码和校准的机制消融；在新身份、多功率分层和成本审计全部通过前，不申请 locked-test。

## 9. 可复核文件

- 数据审计：[audit_summary.json](E:/论文相关/干扰数据/复合干扰数据0716/six_class_pure_label_longtail_v3_IF3_CF8_N8000_AMA/audit_summary.json)
- 实验 manifest：[manifest.json](E:/LJQ_code/CJ_Imbalance_0716/outputs/route_a_ama_direct_sixclass_v3_2x2_20260905/manifest.json)
- 训练汇总：[summary.json](E:/LJQ_code/CJ_Imbalance_0716/outputs/route_a_ama_direct_sixclass_v3_2x2_20260905/summary.json)
- 校准与统计汇总：[analysis_summary.json](E:/LJQ_code/CJ_Imbalance_0716/outputs/route_a_ama_direct_sixclass_v3_2x2_20260905/analysis_summary.json)
- 可执行脚本：[run_route_a_ama_direct_sixclass_v3_2x2_20260905.py](E:/LJQ_code/CJ_Imbalance_0716/experiments/run_route_a_ama_direct_sixclass_v3_2x2_20260905.py)、[analyze_route_a_ama_direct_sixclass_v3_2x2_20260905.py](E:/LJQ_code/CJ_Imbalance_0716/experiments/analyze_route_a_ama_direct_sixclass_v3_2x2_20260905.py)
