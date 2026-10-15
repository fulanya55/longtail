# Route A H0s：高副本参数 + AR(1) 接收机噪声联合压力下的冻结 ISRJ 验证

## 决策

**H0s-R1 通过。** 在不重新训练、不重新校准、不重新选阈值且所有 B0/B2
工件冻结的前提下，aligned-periodicity 增量 B2 在一个同时改变 ISRJ 高副本
参数与接收机 AR(1) 噪声的新身份源上，稳定改善 ISRJ 弱分量证据。它是对
**窄 ISRJ 机制**的联合分布偏移复现；不构成完整六标签 VTSE-VBF 的晋级，
不修复 MFT/VFT，也不授权 test-v3。

## 协议与数据边界

- 正式 R1 源：`C:\LJQ_code\干扰数据\复合干扰数据0803_isrj_ood_h0s_r1_ar1\closedset_6class_train.mat`；SHA-256：`d0eafa3249949de90d3c056c3f9a08ae87d04a610c80f2dc564118b8f232b645`。
- 基础源 SHA-256：`93cce313fb9ad3302ce6b857badecaa4755a140497484bc8d2bfb797e1f51db1`；
  AR(1) I/Q 负载 SHA-256：`ca1078c625ba130a9b316ab9c1b28b5dcc0400ad0efc0b5477779f3713cff411`。
- 5,040 样本、63 种六标签组合、每组合 80 样本；5,040 个唯一
  `receiver_noise_seed` 位于 `1008677743--1008682782`。
- ISRJ 宽度在 `25--30 us`，副本数固定为 9。对基础 I/Q 逐 identity 去除
  精确参考白噪声并加入同功率 proper-complex AR(1) 噪声，`rho=0.80`。
  固定 200,000 点过程检查：均功率 `1.00354`、伪协方差模 `0.00140`、
  lag-1 相关 `0.80388`。
- 审计确认新的 receiver 身份与所有历史训练 MAT、以及 H0e/H0f/H0g/H0h
  记录区间零交集；标签、参数和所有非 I/Q 根数据集保持约定一致，固定列 I/Q
  发生改变。
- test-v3 只读取密封清单，未读取其数据；`access_authorized=false`、
  `analysis_allowed=false`、`used=false`。所有 H0s receipt 都记录
  `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。

### R0 身份冲突的处置

最初预留的 R0 seed `780000000` 在**模型推理前**的正式身份审计中，与既有
`复合干扰数据0802_raat_h0_v1` 完整重合 5,040 个 receiver 身份，因而被拒绝。
R0 没有产生任何预测、模型选择、校准、阈值选择或测试访问。随后使用只读
身份注册表选取未占用的 R1 seed `810000000`；物理参数、AR 系数、B0/B2 工件、
五种子、指标与判定门均未改变。R0 保留在磁盘上仅作为被拒绝的审计轨迹。

## 冻结比较

五个既定训练 seed（2039--2043）比较：

| Arm | 内容 |
| --- | --- |
| B0_H1_Frozen | Focal-36 编码器、H1 线性分类头、RawSaw 路由、已保存 B0 calibration/thresholds |
| B2_H1AlignedPeriodicity_Frozen | B0 加已保存的 310 维 aligned-periodicity expert、融合头、B2 calibration/thresholds |

没有在 H0s-R1 上拟合任何模型、归一化、分数、阈值或校准参数。每个 seed 的
backbone、head、RawSaw、expert、fusion、calibration 和 threshold 哈希均在
`seed_summary.json` 中保存。

## 五种子正式结果：B2 - B0

| 指标 | 均值差 | paired bootstrap 95% CI | 正向 seed |
| --- | ---: | --- | ---: |
| ISRJ AUPR | +0.11677 | [+0.09305, +0.13667] | 5/5 |
| ISRJ 弱分量召回 | +0.07285 | [+0.05695, +0.08278] | 5/5 |
| ISRJ 强分量召回 | +0.19070 | [+0.15675, +0.22464] | 5/5 |
| ISRJ weak-vs-negative AUROC | +0.09709 | [+0.08875, +0.10544] | 5/5 |
| ISRJ FPR | +0.00113 | [-0.00363, +0.00653] | — |
| 六标签 macro-AUPR | +0.01946 | [+0.01551, +0.02278] | 5/5 |
| 六标签 macro-F1 | +0.02644 | [+0.02209, +0.03077] | 5/5 |
| SawFM AUPR / 弱召回 | +0.00000 / +0.00000 | [0, 0] / [0, 0] | exact |

B2 的绝对 ISRJ AUPR / 弱召回 / 弱 AUROC / FPR 为
`0.70932 / 0.28245 / 0.58909 / 0.19315`；对应 B0 为
`0.59256 / 0.20960 / 0.49200 / 0.19202`。五个 ISRJ 弱 AUROC 增量分别为
`+0.08525, +0.08700, +0.10199, +0.11199, +0.09923`，弱召回增量为
`+0.08444, +0.07616, +0.08278, +0.07947, +0.04139`。

因此十项预注册门均通过：AUPR/弱 AUROC 正向、弱召回不低于 +0.030、两个弱
指标至少 4/5 正、B2 ISRJ FPR 不高于 0.25、macro-AUPR 非负、RawSaw 精确不变、
无外部 refit，以及 test-v3 保持密封。

## 非选择性安全描述

该表在完成冻结比较后生成，未参与选择。B2 仅替换 ISRJ logit，故其他五个标签
的预测与 B0 完全一致。B2 的 ISRJ 平均 FPR 为 `0.19315`（B0 `0.19202`）；五个
`receiver_noise_seed mod 5` cohort 中，B2 ISRJ FPR 范围为 `0.17661--0.21452`。
所有标签与 cohort 的最大平均 FPR 为 `0.38145`（SinFM cohort 4），该值与 B0
相同，不能被解释为 B2 获得的安全收益。

## 可主张与不可主张

可主张：在当前受控合成协议中，冻结的 ISRJ aligned-periodicity 增量已跨
development、参数 OOD、高副本 H0g、AR(1) H0h 及本次联合 H0s-R1 条件复现，且
本次弱证据增益并非阈值放宽造成的 FPR 大幅变化。

不可主张：这是完整六标签 VTSE-VBF 的外部泛化；MFT/VFT 的弱分量已经改善；
所有标签都具有低 FPR；或 test-v3 已验证。当前 unified C3 的 MFT/VFT/ISRJ
目标门和同容量 C4 对照仍然失败，test-v3 继续禁止。

## 工件

- 预注册：[route_a_h0s_isrj_combined_parameter_noise_frozen_20260803.md](../plans/route_a_h0s_isrj_combined_parameter_noise_frozen_20260803.md)
- 源审计：`outputs/tapc/route_a_h0s_isrj_combined_generation/identity_parameter_noise_audit.json`
- 冻结汇总：`outputs/tapc/route_a_h0s_isrj_combined_frozen/route_a_h0s_isrj_combined_frozen_summary.json`
- 逐 seed prediction/metrics/strata：`outputs/tapc/route_a_h0s_isrj_combined_frozen/seed_2039` 至 `seed_2043`
- 非选择安全表：`outputs/tapc/route_a_h0s_isrj_combined_frozen/nonselection_safety_report.json`
