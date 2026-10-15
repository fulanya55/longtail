# 路线 A 双轴下一阶段 R1：SawFM 频次 × 可见度协议预注册

## 1. 分支与证据角色

Stage 0 的机器判定为 `no_admissible_non_oracle_observation`，因此本阶段自动进入 `R_sawfm_scoped_dual_axis`。本预注册仅授权全新 development 数据合同、身份角色构建和后续公平模型矩阵；不授权访问历史 locked test、test-v2 或 sealed test-v3。

论文主张暂时限定为：验证 SawFM 标签范围内，标签频次长尾方法和时序可见度专家是否具有可区分主效应及组合增量。不得外推为 MFT/VFT/ISRJ 的统一可见度解决方案。

## 2. 预注册假设

- `H_F`：在标签长尾训练条件下，DB 频次方法相对 Focal 提高 macro-AUPR，且不损害频次平衡条件。
- `H_V`：在 SawFM 低可见度条件下，RawSaw 相对 B0 提高 SawFM weak-vs-negative AUROC 和受 FPR 约束的 weak recall，且保持 strong recall。
- `H_FV`：在 LT-VL 最困难单元中，DB + RawSaw 相对 `best(DB-only, RawSaw-only)` 仍有独立增量。
- `H_safe`：SawFM 专家不会使其余五标签任一 AUPR 下降超过 0.01，也不会造成系统性 FPR 或强正召回退化。

任何排序指标失败均不得通过放宽 FPR、重复选择 seed、重拟合阈值或复用评价身份进行救援。

## 3. 全新数据源与物理干预

数据源固定为：

```text
E:\LJQ_code\干扰数据\dual-axis-sawfm-development-v1
```

生成器固定为：

```text
E:\LJQ_code\CJ_Imbalance_0716\experiments\generate_route_a_dual_axis_sawfm_development_v1.m
```

数据参数：

- 基础种子：`26080701`，不得与 0716、0731_v3、历史 RawSaw、test-v2 或 test-v3 身份混用；
- 六标签顺序：`FMB, SawFM, SinFM, MFT, VFT, ISRJ`；
- 63 个非空组合，每组合 400 个 identity，共 25,200 个 identity；
- 只生成 development-train source，不生成 test 文件；
- VH：SawFM `30 dB`，其余激活分量 `30 dB`；
- VL：SawFM `20 dB`，其余激活分量 `30 dB`；
- 两个视图共享标签、组合、物理参数、`receiver_noise_seed`、背景和样本顺序；
- 同一 identity 的 VH/VL 视图必须进入同一角色；禁止把功率视图当作独立样本随机切分；
- SawFM 单标签样本在两视图中仍属于 strong positive；SawFM 与其他分量复合且处于 20 dB 时按相对 `-10 dB` 定义 weak positive；不含 SawFM 的样本为 negative。

旧 `power_sweep_9level_v1` 只扫描 MFT/VFT/ISRJ，SawFM 固定为 30 dB，因此不得作为本 R1 的正式 SawFM 可见度数据。

## 4. 身份角色与四单元

按 63 个组合分别进行固定配额的 identity 切分，每组合 400 个样本分为：

- optimization：240，共 15,120；
- early stopping：40，共 2,520；
- calibration：40，共 2,520；
- threshold：40，共 2,520；
- evaluation：40，共 2,520。

角色切分种子固定为 `26080702`。所有模型、损失和视图共享完全相同的角色哈希。

训练四单元定义为：

| 单元 | 频次条件 | SawFM 视图 |
|---|---|---|
| FB-VH | frequency-balanced | 30 dB |
| FB-VL | frequency-balanced | 20 dB |
| LT-VH | label-long-tail | 30 dB |
| LT-VL | label-long-tail | 20 dB |

为保证相同训练预算，FB 和 LT optimization manifest 均固定为 2,400 identity：

- FB：从 optimization 中确定性构造组合尽可能均衡的 2,400 identity；
- LT：沿用已验证 label-LT 目标正样本计数 `[1600, 400, 1213, 919, 696, 528]`，SawFM 为 tail label，样本数严格为 2,400；
- combination-LT：使用 IF=4 的组合长尾作为预注册稳健性分析，训练更新步数与主协议相同，不参与主候选选择。

evaluation 同时保留 VH 和 VL 配对视图，模型选择只使用 early；calibration、threshold 和 evaluation 身份相互隔离。

## 5. 公平模型矩阵

主矩阵固定为：

1. `M0_B0_Focal`：`B0_F2_DFMS_Full` + Focal；
2. `MF_B0_DB`：相同 B0 + 项目既有 DB Loss 默认实现；
3. `MV_B0_RawSaw`：B0 主体不变，仅 SawFM logit 使用 RawSaw 标签专家；
4. `MFV_B0_DB_RawSaw`：DB + 相同 RawSaw 专家；
5. `MEC_EqualCapacity`：增加与 RawSaw 参数量匹配的非时序、非相位对照；
6. `MTS_ThresholdOnly`：与 M0 排序完全相同，只执行同一校准和阈值流程。

RawSaw 结构冻结为：raw I/Q RMS normalization，hidden channels 16，卷积核 `7/31/63`，top-k 16，dropout 0.1，约 5,521 参数。只允许做输入 shape、字段和设备兼容性修复。

development 信息破坏消融固定为：phase-only、envelope-only、time-shuffled。消融不得访问或重跑历史 locked test。

## 6. 训练超参数

- 正式模型 seeds：`260811, 260812, 260813, 260814, 260815`；
- epochs：36；early patience：8；batch size：64；gradient clip：5；
- optimizer：AdamW；backbone LR `5e-5`，classifier/expert LR `5e-4`，weight decay `1e-4`；
- M0/MV 使用 Focal：positive gamma 0，negative gamma 3；
- MF/MFV 使用项目当前 `run_tapc --loss db` 的冻结实现和默认参数，必须记录源文件 SHA；
- 相同单元内所有臂共享初始化策略、identity batch 顺序、更新步数、AMP 策略和 checkpoint 选择规则；
- checkpoint 只按 early-role macro-AUPR 选择；不得使用 calibration、threshold 或 evaluation 选择 epoch。

若等容量臂因参数离散无法精确一致，参数差必须小于 RawSaw 增量参数的 5%，并报告实际参数、MACs 和延迟。

## 7. 校准、阈值与指标

- calibration：逐标签 Platt，`C=1e6`、`lbfgs`、`max_iter=1000`；
- threshold grid：`0.05:0.01:0.95`；
- threshold objective：负样本 FPR `<=0.20` 下最大化 weak recall；
- 排序门先于校准和阈值门；
- 同时报告 raw logits、Platt 概率、固定 0.5 和受约束阈值。

主指标：macro-AUPR、macro-F1、Hamming、tail-label AUPR、tail-combination exact match；SawFM weak-vs-negative AUROC、weak/strong recall、negative FPR、ECE、Brier；其余五标签逐标签 AUPR/FPR/strong recall；参数、MACs、GPU/CPU latency、显存和吞吐。

统计使用 receiver-identity paired bootstrap，报告 95% CI、效应量、5-seed signs、组合阶数、hard negatives 和 receiver cohort。

## 8. 硬门与自动停止

- `MF-M0`：LT 条件 macro-AUPR `>= +0.005`，至少 4/5 seeds 正向；FB 条件降幅不超过 0.005；
- `MV-M0`：VL 条件 SawFM weak AUROC `>= +0.020`、weak recall `>= +0.050`、FPR `<= 0.20`，strong recall 降幅不超过 0.01；
- `MFV-best(MF,MV)`：LT-VL macro-AUPR `>= +0.005` 或 SawFM weak recall `>= +0.020`，且配对 bootstrap 95% CI 下界 `> 0`；
- MFV 必须战胜 MEC 与 MTS；
- 其余五标签任一 AUPR 降幅不得超过 0.01；
- phase/time-order 信息破坏方向必须与历史 RawSaw 机制一致。

若 MF 和 MV 分别通过但 MFV 失败，结论限定为“两轴作用可区分，但不需要联合结构”。若 MV 无法在新 identity 复现，关闭 RawSaw 跨分布扩展。若单轴均失败，停止 R 路线模型扩展并形成负结论，不再调网络宽度或 loss weight。

## 9. 分阶段授权

R1 当前只授权：生成器 contract test、MATLAB dry-run、正式数据生成、回读验证、SHA/identity manifest 和四角色构建。只有以下数据门全部通过，才授权 R2 smoke：

1. 两视图各 25,200 identity 且顺序、标签、组合、物理参数、receiver seed 完全配对；
2. VH/VL 仅 SawFM 目标功率发生预注册变化；
3. 五角色互斥，同 identity 的两视图角色一致；
4. 与所有已知历史 development/locked/test-v2/test-v3 receiver identity 交集为零；
5. MAT/HDF5 schema、有限值、幅相和功率真值回读通过；
6. 数据与 manifest SHA、生成器 SHA、环境和 terminal receipt 完整。

sealed test-v3 在本阶段保持 `access_authorized=false, analysis_allowed=false, used=false`。
