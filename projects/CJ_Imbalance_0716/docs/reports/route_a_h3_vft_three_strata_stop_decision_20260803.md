# Route A H3：交错 PRI VFT 三层学习投影停止决策

日期：2026-08-03  
决策：**停止 H3，不做参数搜索或续跑。**

## 决策范围与数据边界

H3 是项目内新建的八脉冲、交错 PRI VFT 物理扩展；它不是原始 0716 单脉冲任务的改写。训练与 development 均使用新的、按 receiver identity 隔离的三臂反事实源：同一身份组中仅改变 VFT 的不存在、弱可见（8 dB）或强可见（20 dB）。本阶段未生成 external/test 角色。

`external_used=false`、`locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。H3 的失败不会消耗或改变任何既有冻结结论；test-v3 继续封存。

## 已通过的前置物理证据

在 development 三臂源上，固定且零训练参数的真实绝对时间戳相干统计具有明确的物理可辨识性：

| 比较 | 真实时间戳 AUROC | 置换时间戳 AUROC | 差值 / 95% paired bootstrap CI |
| --- | ---: | ---: | --- |
| weak VFT vs absent | 0.779026 | 0.570946 | +0.208079 / [0.189368, 0.226863] |
| strong VFT vs absent | 0.833417 | 0.602957 | +0.230460 / 未用于模型晋级 |

这说明交错 PRI 多脉冲观测下的原始物理量不是随机噪声，也说明 H3 的失败不能被归因为“时间戳根本没有信息”。该证据仅支持物理可观测性，不能替代学习模型的等容量对照。

## 冻结的学习模型比较

三个模型均采用固定的 24 epoch、BCE、AdamW（学习率 `3e-4`、weight decay `1e-4`）、batch 16、五个预注册训练种子。`T1-h3` 与 `T2-h3` 只有 VFT 专属的 25→16→1 投影输入不同：前者用真实时间戳，后者用固定置换时间戳；两者参数量均为 26,679。无校准、无阈值、无 F1 选择，结果仅为 raw-score development ranking 证据。

| 对比 | 平均差 | 95% bootstrap CI | 正向种子 | 门槛 | 判定 |
| --- | ---: | --- | ---: | --- | --- |
| T1 − T2，weak AUROC | -0.001365 | [-0.001879, -0.000864] | 0/5 | >= +0.030，CI 下界 > 0 | 失败 |
| T1 − T2，weak AUPR | -0.001061 | [-0.001464, -0.000641] | 0/5 | >= +0.025，CI 下界 > 0 | 失败 |
| T1 − T0，strong AUROC | +0.000864 | [-0.002439, +0.003640] | 3/5 | 均值 >= -0.010 | 仅安全门通过 |

T1 相对 T2 在每个种子都为负，且两项区间均完全低于零。因此不是“效果不够显著”或“随机种子不足”，而是当前可训练加性投影没有将已存在的物理顺序信息转化为超过信息量与容量匹配对照的 VFT 排序增益。

## 工程与可复核性

- 15 份训练运行的配置、checkpoint、预测和摘要由 `outputs/tapc/route_a_h3_vft_three_strata_artifact_audit.json` 审计。
- 汇总门槛结果位于 `outputs/tapc/route_a_h3_vft_three_strata_development_summary.json`；零拟合统计位于 `outputs/tapc/route_a_h3_vft_three_strata_observable/summary.json`。
- `T1/T2` 等参数量、非 VFT logit 与 global 路径相同、以及 VFT logit 为加性专家残差，均由训练后 CPU 合同收据 `outputs/tapc/route_a_h3_vft_three_strata_queue/model_contract_recovery_receipt.json` 验证。
- 原队列的收尾 pytest 在模型训练、工件审计和汇总都完成后遭遇一次 Windows PyTorch DLL 初始化错误。原始失败收据被保留；独立的 `postqueue_reconciliation_receipt.json` 明确记录该范围以及上述恢复验证，不将它伪装成模型结果。

## 禁止的后续动作

H3 的预注册停止规则现已生效。不得在 H3 identities 或新 H3 重复源上搜索：

1. 时间戳排列、ROI、网格或归一化；
2. 专家宽度、残差幅度或融合方式；
3. VFT JNR、epoch、学习率、损失或选择指标；
4. 校准器、阈值、外部/test 集评估。

这些改变都只能把同一个“真实时间戳加性专家是否胜过置换专家”的否定结果变成参数选择，而不能形成新的机制证据。

## 对总体目标的影响与下一关

H3 保留了一条有价值但狭窄的结论：**交错 PRI 多脉冲协议可物理地揭示 VFT 时间结构；当前目标专属神经投影不能证明有效利用它。** 它不改善原始单脉冲 0716 的 MFT/VFT 结论，也不推进完整六标签 VTSE-VBF 到可冻结测试的状态。

下一步不是直接提出 H4 或扩大主干。必须先完成一次独立的机制/文献审计，给出与“固定统计 + 加性 target logit 投影”信息路径不同的可证伪假设、最小零拟合诊断、信息/参数匹配控制、全新 identity 划分和五种子晋级门槛。若无法提出该非重叠假设，应将多脉冲 VFT 限定为物理可观测性附录，主论文只保留已经拥有多轮冻结复现的窄 ISRJ 证据，绝不以 H3 替代统一模型证据。

## 关联记录

- `docs/plans/route_a_h3_vft_three_strata_identifiability_20260803.md`
- `docs/results/route_a_h3_vft_three_strata_results_20260803.md`
- `docs/reports/route_a_goal_completion_gap_audit_20260802.md`
