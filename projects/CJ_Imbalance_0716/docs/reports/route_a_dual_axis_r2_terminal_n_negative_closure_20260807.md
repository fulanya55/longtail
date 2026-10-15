# 路线 A：SawFM 范围双轴 R2 正式验证与终局 N 负结论报告

日期：2026-08-07  
项目：`E:\LJQ_code\CJ_Imbalance_0716`  
环境：`D:\anaconda3\envs\PY10\python.exe`  
证据等级：`[A] development` + `[N] negative/stopped`  
最终分支：`R_sawfm_scoped_dual_axis -> terminal_N`

## 1. 执行结论

本阶段完成了全新的 identity-paired development 数据、冻结预注册、R2 smoke、五 seed × 四 cell 的正式 2×2 主矩阵、5,000 次 receiver-identity 配对 bootstrap、成本审计和终态回执。

最终结论不是统一双轴方法成功，而是一个边界清楚的可靠负结论：

1. **SawFM 可见度轴通过。** RawSaw 相对 B0 的弱分量 AUROC 提升 `+0.316928`，弱召回提升 `+0.420806`，候选 FPR 为 `0.185887`，strong recall 差为 `+0.007500`。
2. **频次轴失败。** DB 相对 Focal 在 LT-VH/LT-VL 上 macro-AUPR 仅提升 `+0.000984`，低于预注册最小效应 `+0.005`；receiver bootstrap 95% CI 为 `[+0.000173,+0.001821]`，其上界仍低于门槛。
3. **组合独立增量失败。** MFV 相对 `best(MF,MV)` 的 LT-VL macro-AUPR 为 `+0.002650`，95% CI `[+0.001733,+0.003613]`，虽稳定非零但未达到 `+0.005`；弱召回增量严格为 `0`。
4. **正向机制 controls 未获准。** frequency-only 硬门失败，因此按事前规则停止，不运行以双轴正主张为前提的 MEC、MTS、phase-only、envelope-only 和 time-shuffled 对照。它们的状态是 `not_run_preregistered_stop`，不是遗漏或运行失败。
5. **禁止创建 confirmation。** R2 门槛未通过，未生成 `dual-axis-sawfm-confirmation-v1`，未访问 locked test、test-v2 或 test-v3。

因此，统一六类“双轴长尾复合识别”路线在本 Goal 内关闭；可保留的是 SawFM 标签特定 RawSaw 可见度模块及其严格限定的证据，不得将其表述为统一双轴网络。

## 2. 数据与身份协议

新 development 数据位于：

- 高可见度：`E:\LJQ_code\干扰数据\dual-axis-sawfm-development-v1\sawfm_level_05__jnr_30dB\closedset_6class_train.mat`；
- 低可见度：`E:\LJQ_code\干扰数据\dual-axis-sawfm-development-v1\sawfm_level_03__jnr_20dB\closedset_6class_train.mat`。

两份数据各有 25,200 个 identity，按同一 `receiver_noise_seed` 和组件随机种子一一配对；只有 SawFM 功率从 30 dB 改为 20 dB。SHA256 分别为：

- VH：`d597a448c1a31322b42b43422192d50292e9e05af5251dcc243d0bfa7a134a8d`；
- VL：`c2f67999d6ec93d64b2e8a05f615836dabe2dfdc88be3c2e9bfc7644a7324bb3`。

身份角色固定为 optimization 15,120、early 2,520、calibration 2,520、threshold 2,520、evaluation 2,520，互不重叠。频次 balanced 和 label-LT 训练 manifest 各 2,400 个 identity；LT 标签计数为 `[1600,400,1213,919,696,528]`，顺序为 FMB、SawFM、SinFM、MFT、VFT、ISRJ。

正式 20 个 cell 的 evaluation `sample_ids` 与 targets 完全对齐，每个 cell 均为 2,520 个唯一 identity。

## 3. 当前网络结构

### 3.1 B0 / MF core

核心模型为 `dfms_iq`，输入 `[B,2,3000]`，六标签输出。正式构造参数：

- `num_classes=6`；
- `width=16`；
- `dropout=0.2`；
- 参数量 257,511；
- 单样本 211,857,536 MACs。

`M0_B0_Focal` 与 `MF_B0_DB` 的网络结构完全相同，只改变训练损失。核心 loss 对五个非 SawFM 标签取均值，并以 0.2 权重加入 SawFM 项，防止核心训练被单一弱标签主导。

### 3.2 RawSaw 可见度专家

RawSaw 只接收原始 I/Q，并只产生 SawFM 一个 logit：

1. 从 I/Q 构造六通道描述：幅度均值、幅度最大、幅度标准差、幅度差分、相位余弦差、相位正弦差；
2. `Conv1d(6,16,k=1)` stem；
3. 三个 depthwise 多尺度分支，kernel `7/31/63`；
4. `Conv1d(48,16,k=1)` 融合；
5. 平均、标准差、最大值、top-k=16 四类统计；
6. `88 -> 32 -> 1` 分类器，dropout 0.1。

RawSaw 增量为 5,521 参数和 7,442,848 MACs。`RoutedSawFMModel` 用 RawSaw logit 完全替换 core 的 SawFM 列，其余五列保持 bitwise 不变。

### 3.3 四个正式模型

| 模型 | 频次机制 | 可见度机制 | 参数量 |
|---|---|---|---:|
| M0_B0_Focal | Focal | 无 | 257,511 |
| MF_B0_DB | DB | 无 | 257,511 |
| MV_B0_RawSaw | Focal | RawSaw 替换 SawFM | 263,032 |
| MFV_B0_DB_RawSaw | DB | RawSaw 替换 SawFM | 263,032 |

同一 seed/cell 的 M0 与 MF 从相同 core 初始 state 开始；MV 复用 M0 checkpoint，MFV 复用 MF checkpoint；MV 与 MFV 复用同一 RawSaw checkpoint，避免组合臂取得额外训练预算。

## 4. 损失、优化与阈值

### 4.1 Core

- M0：multi-label focal，`gamma_pos=0`、`gamma_neg=3`；
- MF：Distribution-Balanced Loss，由该 cell 的训练标签计数构造；
- optimizer：AdamW；
- backbone lr `5e-5`；classifier lr `5e-4`；weight decay `1e-4`；
- batch 64；max epoch 36；patience 8；gradient clip 5.0。

### 4.2 RawSaw

- BCE-with-logits；
- SawFM 正/负各 50% 的 replacement weighted sampler；
- 每 epoch 2,400 draws；
- AdamW，lr `5e-4`，weight decay `1e-4`；
- batch 64；max epoch 36；patience 8；gradient clip 5.0。

### 4.3 选择、校准与阈值

- early role 只选 checkpoint；
- calibration role 对六标签独立 Platt：`C=1e6`、`lbfgs`、`max_iter=1000`；
- threshold role 在 `0.05:0.01:0.95` 上搜索，在 negative FPR `<=0.20` 下最大化 weak recall；
- calibration 和 threshold 均使用配对 VH+VL，但与 evaluation identity 隔离；
- evaluation 只使用一次，之后不改模型、校准器或阈值。

## 5. 正式 2×2 结果

### 5.1 频次轴 MF-M0

五 seed 的 LT macro-AUPR 差为：

| Seed | LT 差 | FB 差 |
|---:|---:|---:|
| 260811 | -0.005167 | +0.009211 |
| 260812 | +0.001246 | +0.003327 |
| 260813 | +0.003988 | -0.000326 |
| 260814 | +0.000447 | -0.005362 |
| 260815 | +0.004405 | +0.006250 |
| Mean | **+0.000984** | **+0.002620** |

LT 为 4/5 seed 正向，但均值只达到最小科学效应的 19.7%。receiver bootstrap CI 的上界 `+0.001821` 仍明显小于 `+0.005`，所以不是简单的样本量不足，而是效应本身过小。

逐标签分解显示 DB 没有形成一致的尾类收益：

| Cell | 主要增益 | 主要损失 |
|---|---|---|
| LT-VH | VFT `+0.016004`、ISRJ `+0.005637` | SinFM `-0.009534`、SawFM `-0.006377` |
| LT-VL | VFT `+0.015418`、SinFM `+0.005634` | MFT `-0.007104`、SawFM `-0.004907` |

这说明当前 DB 主要在标签之间重新分配排序能力，没有稳定解决六标签的频次长尾。

### 5.2 可见度轴 MV-M0

| 指标 | 点估计 | receiver bootstrap 95% CI | 门槛 | 结论 |
|---|---:|---:|---:|---|
| SawFM weak AUROC 差 | +0.316928 | `[+0.304463,+0.329327]` | >=+0.020 | pass |
| SawFM weak recall 差 | +0.420806 | `[+0.409558,+0.432282]` | >=+0.050 | pass |
| candidate FPR | 0.185887 | `[0.172263,0.200078]` | <=0.20 | point pass |
| strong recall 差 | +0.007500 | `[0,+0.016667]` | >=-0.01 | pass |

五个 seed 的 AUROC 和 weak recall 均为正。FPR 点估计通过，但 CI 上界略高于 0.20，论文中应报告这一边界风险，不能称为“所有重采样均满足 FPR”。

### 5.3 组合轴 MFV-best(MF,MV)

| Seed | macro-AUPR 差 | weak recall 差 |
|---:|---:|---:|
| 260811 | -0.001850 | 0 |
| 260812 | +0.004692 | 0 |
| 260813 | +0.003563 | 0 |
| 260814 | -0.001994 | 0 |
| 260815 | +0.008839 | 0 |
| Mean | **+0.002650** | **0** |

五个 seed 的最佳单轴全部是 MV。receiver bootstrap 结果：

- macro-AUPR：`+0.002650`，95% CI `[+0.001733,+0.003613]`；
- weak recall：`0`，95% CI `[0,0]`。

macro 增量统计上非零，但 CI 整体低于预注册最小效应 `+0.005`；seed signs 只有 3/5。故 combined 独立增量失败。

## 6. 失败根因

### 6.1 组合结构没有显式交互通道

MFV 的 SawFM logit与 MV 完全相同；MFV 的其余五个 logits 与 MF 完全相同。该结构是模块拼接，不是频次轴与可见度轴的交互建模。因此：

- `MFV-MV` 的 SawFM weak recall 必然为 0；
- 组合 macro 增量只来自五个非 SawFM 标签的 `MF-M0` AUPR 变化，再被六标签平均稀释；
- 当前 `+0.002650` 不能解释为“双轴协同”，只能解释为在 RawSaw 模型上附加 DB 后的非 SawFM 小幅净变化。

### 6.2 DB 的收益不稳定且与可见度困难错位

DB 感知标签计数，但不感知同一标签内部的 weak/strong 可见度。SawFM 是 LT manifest 中最少的标签之一，但 DB 在 LT-VH 和 LT-VL 上的 SawFM AUPR均为负，说明“按标签频次重权”没有解决“标签内部弱证据”问题。VFT 的增益被 MFT、SawFM 或 SinFM 损失抵消，频次主效应无法达到门槛。

### 6.3 历史 Base-DB 证据不能直接迁移

历史 Base-DB 相对 Base-ASL 的 18 个 tail rotation 平均提升属于不同 source、不同 comparator 和不同训练协议。当前正式比较是新 identity 2×2 数据上的 DB 对 Focal，且要求同一初始化、训练预算、校准和阈值。因此历史结果只能作为提出 MF 的依据，不能覆盖本轮失败。

### 6.4 不是 backbone 容量不足的直接证据

RawSaw 仅增加 2.14% 参数、3.51% MACs，就获得很大的 SawFM 弱证据排序提升，说明弱分量问题主要是观测/归纳偏置不匹配，而不是单纯缺少主干容量。此前 generic backbone strengthening 已关闭，本轮也没有证据支持继续无定向扩宽主干。

### 6.5 训练预算不是首要解释

20 个 cell 中 RawSaw 有 10 个 best epoch 位于 36，core 的 best epoch 范围为 8–36；但正式门槛基于冻结预算和公平比较。更重要的是 receiver CI 已把频次效应上界限制在 `+0.001821`，继续在同一 source 上延长或调参属于结果后救援，不被允许。

## 7. 安全性与成本

LT-VL 上 MFV 相对 M0 的非 SawFM AUPR 差：

| 标签 | 点估计 | receiver bootstrap 95% CI |
|---|---:|---:|
| FMB | +0.000190 | `[+0.000021,+0.000408]` |
| SinFM | +0.005634 | `[+0.002784,+0.008825]` |
| MFT | -0.007104 | `[-0.010349,-0.003884]` |
| VFT | +0.015418 | `[+0.012611,+0.018297]` |
| ISRJ | +0.001762 | `[+0.000830,+0.002762]` |

预注册安全门按五 seed 均值判断，MFT 点估计仍高于 `-0.01`，故 point safety pass；但其 receiver CI 下界略低于 `-0.01`，必须作为风险披露。

RTX 5080 冻结成本审计：

| 模型 | 参数 | MACs/sample | batch-1 latency | batch-64 throughput |
|---|---:|---:|---:|---:|
| M0/MF core | 257,511 | 211,857,536 | 3.262 ms | 12,223.75 samples/s |
| MV/MFV routed | 263,032 | 219,300,384 | 4.226 ms | 10,479.95 samples/s |

增量参数 `+2.14%`，MACs `+3.51%`；batch-1 延迟增加约 29.6%。正式矩阵墙钟 4,617.74 秒，最大 CUDA memory allocated 1,051,497,472 bytes。

## 8. 终局 N 完成判断

| 终局 N 条件 | 证据 | 状态 |
|---|---|---|
| 新观测候选不合格或 S2 被控制否定 | Stage 0：无 admissible non-oracle second observation | completed |
| 路线 R 未形成独立作用 | frequency-only fail；combined fail | completed |
| 失败证据、控制、统计、receipt、边界完整 | 20/20 receipts；5,000 receiver bootstrap；条件 controls 明确 stopped；成本与复现审计 | completed |
| 关闭统一路线并收缩论文范围 | 本报告与 evidence ledger | completed |

本 Goal 因此满足终局 N，可以标记 complete。

## 9. 允许与禁止的论文表述

允许：

- `[A]` 在全新 identity-paired development 上，RawSaw 对 SawFM 低可见度证据具有稳定的大幅增益；
- `[N]` 在共同 2×2 协议中，当前 DB 频次轴没有达到预注册最小效应；
- `[N]` 当前模块化 MFV 没有证明频次与可见度的独立协同；
- `[F historical scoped]` 既有一次性 locked test 只支持 SawFM 标签特定 RawSaw 结论，不能扩展为统一方法。

禁止：

- “提出并验证了统一六类双轴长尾网络”；
- “frequency 与 visibility 存在显著协同”；
- “MFV 在 confirmation/test 上通过”；
- 用本轮 development bootstrap 替代独立 confirmation；
- 在同一 evaluation source 上继续调 DB 权重、loss、阈值或选择 seed 后重新宣称通过。

## 10. 范围收缩与后续唯一动作

本 Goal 内的唯一动作是**关闭统一双轴模型路线并冻结全部证据**。若后续另立研究 Goal，应使用全新 identity source，并在以下两个方向中只选择一个：

1. 将论文主张收缩为 SawFM 弱可见度标签特定专家，补独立 confirmation、equal-capacity 与信息破坏消融；或
2. 重新定义真正的频次长尾问题，使用能区分 tail label 与 label-internal visibility 的新机制和新 development-v2，不复用本轮 evaluation 进行选择。

不得把第二方向作为本 Goal 的无限续调理由。

## 11. 核心产物

- 正式汇总：`E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_dual_axis_next\r2_formal_primary\formal_primary_summary.json`；
- 正式 terminal：`E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_dual_axis_next\r2_formal_primary\terminal_receipt.json`；
- receiver bootstrap：`E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_dual_axis_next\r2_receiver_bootstrap_audit\receiver_bootstrap_summary.json`；
- 成本审计：`E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_dual_axis_next\r2_cost_audit\cost_summary.json`；
- evidence ledger：`E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_dual_axis_next\evidence_ledger.json`。

