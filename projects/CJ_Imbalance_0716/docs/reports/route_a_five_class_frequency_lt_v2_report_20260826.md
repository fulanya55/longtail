# 路线 A：五类频次长尾 V2 自动验证报告

日期：2026-08-26  
结论：`[N] protocol_incomplete_test_contaminated / candidate_rejected`

> 完成性复审修正：本报告最初曾把 V2 标为已完成。逐条对照 Goal Prompt 后发现，当前 `closedset_6class_test.mat` 在探索期间被多次执行最终评估，违反“所有开发选择完成后只评估一次”的协议；Stage 1 未通过准入后，Stage 2/3 仍被执行，只能作为探索性诊断。此外，原冻结头实现不同方法的每 epoch 抽样预算并不完全一致。现有结果全部保留，但不得作为合规独立测试或正式创新证明。

## 1. 研究边界

本轮从训练和测试源中删除 SawFM 阳性样本，只保留 `FMB / SinFM / MFT / VFT / ISRJ`。训练、测试各保留 12,400 个样本；不使用九档功率数据、功率长尾、SawFM 专家、半监督、开集或封存 test-v2/test-v3。

固定骨干为 B0 `DelayedFusionMultiScaleIQClassifier (dfms_iq)`：width=16、feature_dim=128、dropout=0.2、五类线性输出。联合训练固定 AdamW、backbone lr `5e-5`、head lr `5e-4`、weight decay `1e-4`、gradient clip 5.0，每臂每 epoch 9,920 draws，5 个预注册 seed。

所有 sampler、loss、logit adjustment 和冻结头均在 train/early/calibration/threshold 角色中完成选择；test 在开发选择锁定后统一评估。

## 2. Stage 0：频次与组合覆盖审计

- 完整五类训练池五类正样本数均为 6,400，是标签边际频次平衡源。
- LabelIF4 目标及实际计数均为 `[4000,3000,2000,1500,1000]`。
- 完整池覆盖 31 个非空五类组合；LabelIF4 只覆盖 14 个组合，删除了 17 个组合。
- 因此 V1/V2 的 LabelIF4 不只是标签先验变化，还包含显著的共现组合覆盖损失。

## 3. Stage 1：完整池 label-aware sampler

| 方法 | test macro-AUPR，5 seed 均值±标准差 | 相对 Uniform |
|---|---:|---:|
| Uniform-Focal | 0.975610 ± 0.000765 | 基线 |
| MaxPositive-Focal | 0.975647 ± 0.000708 | +0.000037 |
| MeanPositive-Focal | 0.975609 ± 0.000965 | −0.000001 |

MaxPositive 和 MeanPositive 均未达到 macro-AUPR +0.010、tail +0.020、4/5 seed 同向的门槛。完整池五类计数相同，inverse-frequency 权重基本退化为常数，因此这些 sampler 实质上是等价控制。阶段 1 失败，后续使用 Uniform fallback 隔离损失因素。

## 4. Stage 2：LabelIF4 等预算损失比较

| 方法 | test macro-AUPR，5 seed 均值±标准差 |
|---|---:|
| Focal | 0.906710 ± 0.014011 |
| DB | **0.912393 ± 0.015871** |
| CB-BCE β=0.99 | 0.903237 ± 0.018240 |
| CB-BCE β=0.999 | 0.897865 ± 0.018603 |
| ASL-DB w=0.50 | 0.910552 ± 0.016757 |
| ASL-DB w=0.75 | 0.907793 ± 0.016626 |
| ASL-DB w=0.90 | 0.911144 ± 0.012854 |

early 预注册选择得到 `ASLDB_w0.50`；其相对 Focal 的最终 test 结果为：

- macro-AUPR 平均增益：`+0.003843`；
- tail mean 增益：`+0.002355`；
- 正向 seed：`3/5`；
- 最差头部类变化：`−0.042476`；
- 5,000 次 seed bootstrap 95% CI：`[−0.001824, +0.009509]`。

所有硬门槛均未满足。DB 虽有最高 test 均值，但它不是 early 锁定候选，而且其优势仍不足以形成稳定创新证据，不能在看到 test 后反向更换候选。

## 5. Stage 3：logit adjustment

5/5 seed 在 early 网格中选择 τ=0.25。每标签增加常数先验偏置不会改变标签内部样本排序，因此 test macro-AUPR 只出现约 `1e-6` 量级数值变化；它只能改变阈值、precision/recall 与校准，不能补充特征证据。

## 6. Stage 4：冻结 B0 特征分类头

完成性复审后，Stage 4 已按 Prompt 回退到 `B0-Focal-LabelIF4`，统一为每个分类头 `9,920 draws/epoch`，并只使用 early 角色重跑；没有再次访问 test。

| 冻结头 | early macro-AUPR 均值 |
|---|---:|
| OrdinaryLinearHead | 0.946396 |
| ClassBalancedBCEHead | 0.946396 |
| cRT | 0.945945 |
| RBC-cRT | 0.946590 |

RBC-cRT 相对普通头只增加约 `+0.000194`，cRT 下降约 `−0.000451`。class-balanced BCE 与普通头等价；冻结头没有稳定改善。分类边界只能解释很小部分误差，主要问题仍是 LabelIF4 删除组合后造成的训练证据覆盖不足。

## 7. 归因结论

1. `frequency_prior_effect`：标签计数由 `[6400×5]` 变为 `[4000,3000,2000,1500,1000]`。
2. `cooccurrence_coverage_effect`：组合覆盖由 31 降到 14，这是最强的结构性变化。
3. `optimization_budget_effect`：固定 9,920 draws/epoch 已消除方法间优化步数不一致；各类期望正曝光仍随 LabelIF4 计数变化。
4. B0 在完整池上达到约 0.976 macro-AUPR，证明骨干基础提取能力充分；当前不支持继续扩宽 backbone。
5. sampler、损失、先验偏置和冻结分类头均未达到预注册硬门槛。

## 8. 最终阶段判断

- `[F]`：SawFM 过滤、五类 B0、联合训练角色隔离、LabelIF4 与组合覆盖审计均已验证。
- `[A]`：DB/ASL-DB、RBC-cRT 和先验校正只保留为探索性机制诊断。
- `[N]`：没有候选满足 macro-AUPR +0.010、tail +0.020、4/5 seed 同向和头部安全门槛；receiver bootstrap CI 为 `[−0.006712,+0.012819]`；独立 manifest 非反转门槛未执行；当前 test 已被重复探索性访问。
- `[T]`：标签条件分类头和 stage4 adapter 不获准进入，因为当前门槛未通过。

本轮可靠结论不是“频次长尾已解决”，而是：当前删除样本式 LabelIF4 同时破坏组合覆盖；在固定有效 B0 下，仅靠 label-aware sampler、DB/CB-BCE/ASL-DB、logit adjustment 或冻结分类头都不足以稳定恢复性能。下一轮应优先生成保持全部组合覆盖、只改变每组合数量的新独立五类长尾数据源，而不是继续堆叠损失函数或扩宽主干。

由于没有新的、未污染的独立五类频次长尾 development/confirmation 源，本 Goal 不能形成协议合规的最终阳性验证。恢复条件是：生成新 identity、保持 31 个非空组合覆盖、预先划分 train/early/calibration/threshold/evaluation，并在模型完全冻结后只执行一次 evaluation。

## 9. 实验产物

- `outputs/route_a_five_class_frequency_lt_v2/run_receipt.json`
- `outputs/route_a_five_class_frequency_lt_v2/final_test_summary.json`
- `outputs/route_a_five_class_frequency_lt_v2/metrics_summary.csv`
- `outputs/route_a_five_class_frequency_lt_v2/bootstrap_summary.json`
- `outputs/route_a_five_class_frequency_lt_v2/cooccurrence_coverage_analysis.json`
- `outputs/route_a_five_class_frequency_lt_v2/sampler_and_cooccurrence_diagnostics.json`
- `outputs/route_a_five_class_frequency_lt_v2/gradient_proxy_analysis.json`
- `outputs/route_a_five_class_frequency_lt_v2/engineering_metrics.json`
- `outputs/route_a_five_class_frequency_lt_v2/completion_audit_receipt.json`
- `outputs/route_a_five_class_frequency_lt_v2/protocol_audit.json`
- `outputs/route_a_five_class_frequency_lt_v2/receiver_bootstrap_summary.json`
- `outputs/route_a_five_class_frequency_lt_v2/independent_manifest_gate_receipt.json`
- `E:\LJQ_code\干扰数据\复合干扰数据0716\route_a_five_class_frequency_lt_v2\stage0_audit.json`
- `E:\LJQ_code\干扰数据\复合干扰数据0716\route_a_five_class_frequency_lt_v2\label_if4_manifest.json`
