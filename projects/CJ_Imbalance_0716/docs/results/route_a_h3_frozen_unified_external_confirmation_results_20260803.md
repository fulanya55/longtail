# Route A H3：冻结统一候选的新身份外部确认结果

日期：2026-08-03  
状态：完成；通过预注册的 H3 确认门；**不授权 test-v3**。

## 问题与证据边界

H3 检验的是已经冻结的六标签候选
`H1_B2_MFTSharedDecision_VFTRidgeFusion_v2` 是否能在新的、身份隔离的
标准六类来源上保留对 H1 的改进。它不是重新训练、重新校准、重新选阈值或
调节物理专家的实验。

- 冻结清单：
  `frozen/20260801_route_a_h1_b2_mft_vft_shared_policy_v2/frozen_manifest.json`
- 冻结清单 SHA-256：
  `c29ce67bd0384cf6d54095343773b557258e287394c89f8deb3acbd6ba943747`
- 五个随机种子：2039--2043。
- 比较对象：冻结 Q0（H1 + 相同 RawSaw SawFM 替换）与冻结 Q6（完整
  Shared-MFT-decision + VFT-aligned-Ridge 候选）；Q3、Q5 为冻结 VFT
  机制对照。
- 新 H3 来源：25,200 条记录，63 个非空组合各 400 条；样本种子范围
  134,000,001--134,025,200。文件为
  `C:\LJQ_code\干扰数据\复合干扰数据0803_h3_frozen_unified_confirmation_r1\closedset_6class_train.mat`，
  SHA-256 为
  `09b99e65e2fb67f50418fc8a597bc6de993851dc6c6cd67457da7933ee76f8b3`。
- 数据身份审计确认其 receiver/noise identity 与每个可读历史训练来源零重合；
  未生成或读取测试分割。
- H3 中 `external_refit`、`hyperparameter_search`、`calibration_refit`、
  `threshold_refit` 全部为 `false`；locked test、test-v2、test-v3 均为
  `false`。

因此，H3 是“同一官方标准生成器下的新身份、冻结推理外部确认”。它支持可重复
性与身份隔离泛化，**不应表述为**未知噪声模型、未知信道或真实采集 OOD 的
鲁棒性证明。

## 统一提升门：13/13 通过

相对 Q0 的五 seed 均值如下：

| 终点 | Q6 - Q0 |
|---|---:|
| 六标签 macro-AUPR | `+0.006878` |
| 六标签 macro-F1 | `+0.011524` |
| MFT/VFT/ISRJ 平均弱分量召回 | `+0.039833` |
| MFT/VFT/ISRJ 平均弱正-vs-负 AUROC | `+0.039987` |
| 三目标平均负类 FPR | `0.185258` |
| 三目标平均强正召回 | `+0.024309` |
| MFT AUPR | `+0.004308` |
| VFT AUPR | `+0.013549` |
| ISRJ AUPR | `+0.023409` |
| SawFM AUPR / 弱召回 | 精确不变 |
| SawFM-present macro-F1 | `+0.010349` |
| SawFM-absent macro-F1 | `+0.012542` |

主弱召回端点在 5/5 个 seed 均为正：`+0.04630`、`+0.03212`、
`+0.04207`、`+0.03699`、`+0.04168`。所有目标标签 AUPR 均为正变化，且
目标 FPR 未超过预注册的 0.20 上限。

这确认了完整候选并非仅在此前已经被其他路线消费的 development-v12 上有效；
它在新的标准身份组上保留了 RawSaw SawFM 输出不变条件下的三目标弱分量收益。

## VFT 对齐证据门：9/9 通过

冻结的 Q4 对齐 Ridge 融合相对 Q2 和两个冻结对照显示：

| 指标 | 结果 |
|---|---:|
| 弱正-vs-负 AUROC，Q4 - Q2 | `+0.016699` |
| 弱分量召回，Q4 - Q2 | `+0.018344` |
| VFT AUPR，Q4 - Q2 | `+0.013549` |
| 弱 AUROC，Q4 - 同容量 Q3 | `+0.016699` |
| 弱 AUROC，Q4 - shuffled Q5 | `+0.017240` |
| VFT 负类 FPR | `0.194371` |

五个 seed 对 Q2 与 Q5 的对齐 AUROC 增益均为正，且非 VFT logits 完全一致。
因此，这一结果支持“与 VFT 局部证据对齐”而非仅增加相同容量或任意附加分数的
解释，但仍仅限于本标准生成协议。

## MFT 共享决策安全门：8/8 通过

共享 H1+B2 的 MFT 校准器和阈值，而非为残差另拟合决策映射后，结构性
no-new-positive 约束得到保持：

| 指标 | Q6 相对 Q1 的均值（bootstrap 95% CI） |
|---|---:|
| 高阶 hard-negative FPR | `-0.02950` (`[-0.03492, -0.02475]`) |
| order-5 FPR | `-0.02650` (`[-0.03500, -0.01850]`) |
| hard weak-positive-vs-negative AUROC | `+0.01395` (`[+0.01224, +0.01521]`) |
| MFT 弱召回 | `-0.00922` (`[-0.01097, -0.00713]`) |

五个 seed 的高阶 hard-negative FPR 全部下降；Q1 阴性变为 Q6 阳性的数量为
`0`；非 MFT 概率与决策失配均为 `0`。弱召回的小幅代价低于预注册的 0.020
上限，故该路线的结论是“以受控弱召回代价换取 hard-negative 安全性”，而非
无代价提升。

## 工件与可复核性

- 预注册方案：
  `docs/plans/route_a_h3_frozen_unified_external_confirmation_20260803.md`
- 生成前身份选择：
  `outputs/tapc/route_a_h3_frozen_confirmation_generation/pre_generation_identity_selection.json`
- 正式生成与身份审计：
  `outputs/tapc/route_a_h3_frozen_confirmation_generation/formal_generation_receipt.json`、
  `outputs/tapc/route_a_h3_frozen_confirmation_generation/identity_audit.json`
- 逐 seed 预测、指标和收据：
  `outputs/tapc/route_a_h3_frozen_unified_confirmation/seed_2039` 至
  `outputs/tapc/route_a_h3_frozen_unified_confirmation/seed_2043`
- 聚合结果与门判定：
  `outputs/tapc/route_a_h3_frozen_unified_confirmation/summary.json`

## 当前结论与禁止外推

H3 为完整、全冻结的 Route A 统一候选提供了独立的新身份确认：统一主门、VFT
机制门和 MFT 共享决策安全门都通过，且 SawFM 路由严格不回退。H1u（MFT 延迟
孔径学习失败）与 H2v（条件互补性准入失败）是已经关闭的替代机制，未参与 H3
选择或修复。

这一条证据尚不足以授权一次性 test-v3。test-v3 仍密封：`access_authorized`、
`analysis_allowed` 与 `used` 都为 `false`。在考虑最终测试前，仍须按总台账
补齐或正式停止以下证据：完整候选的核心消融覆盖、最新多标签长尾基线的公平
复现/审计、完整候选跨噪声/参数 OOD 鲁棒性、配对显著性与效应量、以及工程成本
核算。任何对候选、校准器或阈值的后续改变，都需要新的 development 身份来源，
不能回用 H3 作筛选。
