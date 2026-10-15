# Route A 当前证据台账：H3 后的主张边界与下一准入门

日期：2026-08-03  
结论：**总目标仍未完成；test-v3 不得授权。**

## 本次台账的必要区分

项目中同时存在两类不能合并为一个结论的证据。

1. H3 对冻结 `H1_B2_MFTSharedDecision_VFTRidgeFusion_v2` 的新身份标准源
   确认已经通过。它表明该已冻结的标签特异物理专家组合，在相同官方生成器、
   新 receiver/noise identity 上仍能保留相对 H1 的统一收益和 SawFM 路由安全性。
2. VTSE-VBF 的论文级核心因果命题仍要求 C0--C4 的同协议比较。此前 C3
   相对等参数、无可见性机制的 C4 没有通过关键弱证据门，故不能将 H3 的
   冻结组件确认解释为“标签内可见性长尾机制已被最终证明”。

这两条记录并不冲突：前者提高了一个固定候选的可复核性，后者限制该结果可以
承担的因果和泛化主张。

## 已确认的正向证据

| 证据问题 | 当前结论 | 可报告范围 | 不可外推范围 |
|---|---|---|---|
| RawSaw SawFM 路由 | 已确认 | SawFM 标签局部替换可保持非目标决策安全 | 不证明 MFT/VFT/ISRJ 的可见性机制 |
| 频次组/标签频次损失 | 已确认（机制边界） | 频次组 SoftF1、BCE/Focal/ASL/CB/LA 对同标签 weak/strong 名称交换不变；DB 仅可受共现组合影响 | 不将频次或组合重加权写为标签内可见性建模 |
| ISRJ aligned-periodicity | 已确认（组件） | 新身份的高副本参数、AR(1) 噪声与二者联合条件下，冻结 B2 相对 B0 的弱证据提升 | 不证明完整六标签融合在所有 OOD 中安全 |
| MFT shared decision | 已确认（冻结候选） | H3 中 hard-negative FPR 降 `0.02950`，零 negative-to-positive 翻转，弱召回代价受限 | 不证明无代价改善或跨协议泛化 |
| VFT aligned Ridge | 已确认（冻结候选） | H3 中相对同容量和 shuffled 对照的弱 AUROC 分别 `+0.01670`、`+0.01724` | 不证明新的时序观测协议或真实数据有效 |
| H3 完整冻结候选 | 已确认（新身份标准源） | macro-AUPR `+0.00688`、macro-F1 `+0.01152`、三目标弱召回 `+0.03983`、弱 AUROC `+0.03999`；5/5 主弱召回为正 | 不是未知噪声、未知生成器或密封独立测试 |

H3 的原始结果、逐 seed 工件、源身份审计和封存检查见
`docs/results/route_a_h3_frozen_unified_external_confirmation_results_20260803.md`。

## 已排除或停止的路线

| 路线 | 裁决 | 证据原因 | 后续限制 |
|---|---|---|---|
| H1u MFT delay-aperture 学习 | 停止 | 残差自身有信息，但未转化为下游分类改善 | 不重调 bands/jitter/JNR/backbone/loss |
| H2v 条件互补性 | 停止 | 新 0/4/8 dB MFT 梯度源的 B0 准入未达 inversion 门 | 不训练后续条件校正器 |
| H1t 非对称高低功率转移 | 停止 | 目标弱 AUROC 无稳定收益且强正明显下降 | 不重调 teacher/student 配方 |
| H3/H4 时间戳或 delay-frequency 学习投影 | 停止 | 匹配置换对照未通过可辨识门 | 不以更宽网络、ROI、阈值或损失重试 |
| H5o 冻结联合 OOD 确认 | 停止（该条件） | 新 identity 的高副本 ISRJ + AR(1) rho=0.65 下，弱召回 `+0.01766 < +0.030`，目标 FPR `0.21336 > 0.20`，VFT FPR `0.32806`；只读分层后验确认 VFT 为阈值运输不足、MFT 为过度保守弱召回损失 | 不在 H5o 上调阈值、融合、损失或噪声参数；H3 标准身份结果仍单独成立 |
| C3 VTSE-VBF 晋级 | 未晋级 | 相对 C4 的 macro-AUPR `-0.00290`、弱 AUROC `-0.00401`，无法排除容量解释 | 不将其冻结为最终论文模型 |

## 目标完成条件的逐项状态

| 完成条件 | 状态 | 尚缺的直接证据 |
|---|---|---|
| 六标签统一模型 | 未满足 | 一个在 C0--C4 同协议中战胜 C4 的可晋级候选 |
| 14 项核心消融 | 覆盖完成 | 仅可服务于对应历史候选；不能替代新候选的因果消融 |
| 主流方法公平比较 | 部分完成 | BCE、pos-BCE、Focal、ASL、DB、CB、LA、两阶段、DR、RAL 已有；近期外部多标签方法尚无结构兼容的精确复现 |
| 多 seed 统计与工程成本 | 组件/冻结候选充分 | 新的最终统一 bundle 才能拥有终局统计和成本 |
| 弱/强/负三层机制 | 仅 ISRJ 组件充分 | MFT/VFT 在原单脉冲观测协议下的可辨识、非容量因果证据 |
| 完整候选鲁棒性 | 未满足 | H5o 给出了同一冻结候选的新 identity 联合参数/噪声 OOD 反例；此前仅 ISRJ 组件窄条件通过 |
| 一次性独立 test-v3 | 禁止 | 所有前置项完成、候选和推理 bundle 完全冻结 |

## 为什么 H3 不能授权 test-v3

H3 的有效性在于它不做任何选择；也正因如此，它不能回填一个不同的、需要
C3/C4 反容量对照才能回答的机制因果问题。若在 H3 成功后直接执行 test-v3，
将混淆“冻结候选在同生成器新身份上可复现”与“VTSE-VBF 的标签内可见性长尾
创新已被充分验证”。此外，完整候选过去在 ISRJ 参数 OOD 上存在 FPR 越界记录，
H3 的标准源不构成对此的反证。

test-v3 密封清单依旧满足：`access_authorized=false`、`analysis_allowed=false`、
`used=false`。本轮未读取其 I/Q 或标签。

## 唯一合理的下一研究门

原 0716 单脉冲协议已显示 MFT replica 重叠和 VFT 有效相位孔径过短的观测约束；
在该约束下继续扩大主干、改局部窗口、调整损失、阈值或同实体重跑，不能构成新的
可证伪机制。交错 PRI 的 VFT 物理筛查（H0q/H1a/H1b）曾证明新的慢时间观测量
可辨识，但其随后学习型投影/beamspace 路线（H2/H3/H4）未超过匹配置换或容量
对照，已经关闭；因此不能把“再做 multi-PRI”当作一个新实验方案。

若继续追求六标签统一论文，唯一可接受的下一门仍是**协议级可观测性准入**，但
必须是与已关闭读出不重叠的新物理机制，而不是新的融合头：

> 在与 0716、H0q/H1a/H1b/H2/H3/H4 identity 完全隔离的新来源中，提出一个
> 与 MFT delayed-copy correlation、LFM matched filter、既有 staggered-PRI
> coherence/beamspace 均不重叠、且可由文献和物理模型独立定义的新观测量；在不训练
> 分类器、无阈值和无融合搜索的匹配置换诊断中，验证它是否能稳定区分目标弱正和
> 对应 hard negatives。

只有当该诊断相对匹配置换控制满足预注册的可辨识门，才允许从新的 identity-
isolated H0/H1 训练一个新专家；失败则正式将论文收缩为“RawSaw + ISRJ
aligned-periodicity”的窄证据主张，而不是继续消耗 development 或 test-v3。

## 关联记录

- H3 结果：`docs/results/route_a_h3_frozen_unified_external_confirmation_results_20260803.md`
- H5o OOD 结果：`docs/results/route_a_h5o_frozen_unified_combined_ood_results_20260803.md`
- H5o 只读失败复盘：`docs/reports/route_a_h5o_frozen_ood_failure_postmortem_20260803.md`
- 频次组与可见性解析边界：`docs/reports/route_a_frequency_group_visibility_invariance_results_20260803.md`
- 目标状态审计：`docs/reports/route_a_goal_completion_state_audit_20260803.md`
- 核心缺口审计：`docs/reports/route_a_final_innovation_evidence_gap_audit_20260801.md`
- 近期文献/兼容性边界：`docs/reports/route_a_2026_literature_refresh_compatibility_audit_20260803.md`
- H5o 后机制准入刷新收据：`docs/reports/route_a_h5o_postmortem_mechanism_admission_refresh_20260803.md`
- test-v3 密封清单：`frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json`
