# Route A 最终候选弱分量阈值敏感性结果

## 结论

完整候选对弱分量的排序增益不依赖单一 `10 dB` 定义。在 5、10、15 dB 三种固定阈值下，三目标平均弱正-vs-negative AUROC 增益均为正且超过 `+0.015`。但 15 dB 的极弱子集召回增益只有 `+0.02246`，不足原始 `+0.030` 门槛；论文应报告这一限制，不能把“所有弱度均达到同一召回增益”作为主张。

## Q6 完整候选减 Q0 H1（五种子均值）

| 弱阈值 | 每标签弱样本数（MFT/VFT/ISRJ） | 目标弱召回 | 目标弱 AUROC | MFT/VFT/ISRJ 弱 AUROC |
|---:|---|---:|---:|---|
| 5 dB | 5608 / 5592 / 5565 | +0.04310 | +0.03552 | +0.01270 / +0.02045 / +0.07340 |
| 10 dB | 2861 / 2907 / 2918 | +0.03759 | +0.03886 | +0.01640 / +0.01706 / +0.08313 |
| 15 dB | 853 / 875 / 859 | +0.02246 | +0.03049 | +0.01418 / +0.01209 / +0.06520 |

macro-AUPR、macro-F1 和目标 FPR 与弱 mask 无关，因此三个阈值下仍分别为 `+0.00660`、`+0.01099` 与 `-0.01144`。

## 解释

MFT 在三个阈值下都存在有限弱召回损失（5/10/15 dB 分别为 `-0.00835/-0.01265/-0.02110`），这是 MFT guard 用召回交换困难负样本 FPR 的已知设计结果；而 VFT 和 ISRJ 在三种阈值下保持正的弱召回与弱 AUROC，ISRJ 最稳定。

## 边界与产物

该工作只重新定义评估 mask，不改变模型、预测、校准器或阈值，不用于候选选择。development-v12 是已用的后验诊断集，不能作为新鲜确认；`locked_test_used=false`、`test_v3_used=false`。

- 预注册：[route_a_final_candidate_weak_threshold_sensitivity_plan_20260801.md](../plans/route_a_final_candidate_weak_threshold_sensitivity_plan_20260801.md)
- 实现：`experiments/analyze_route_a_final_candidate_weak_threshold_sensitivity.py`
- 汇总：`outputs/tapc/route_a_final_candidate_weak_threshold_sensitivity/route_a_final_candidate_weak_threshold_sensitivity_summary.json`
