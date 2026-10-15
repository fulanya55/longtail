# 路线 A：Raw+Platt 五种子确认与标签级路由计划（2026-07-25）

## 目标

确认 Raw+Platt 的独立 SawFM 决策能力能否跨五个完整模型种子稳定成立，并在不改动其他五类判决的前提下，验证标签级路由是否改善六类整体指标。

## 阶段 F1：补齐可比模型链

- seeds：2039、2040；
- 使用与 2041–2043 完全一致的 F2、H2 训练配置、数据角色和预算；
- 复用已经冻结的 `R0_FullPhysics` Raw checkpoint；
- 生成 checkpoint、history、metrics、predictions、split hash、receipt；
- 不使用 locked test。

停止条件：任一 seed 的数据角色、identity 隔离或 checkpoint 来源无法与既有三种子对齐，则不做五种子聚合。

## 阶段 F2：五种子冻结确认

- 固定 `RawStage1SawFMFrontend(mode="raw", hidden_channels=16, topk=16, dropout=0.1)`；
- ordinary calibration 仅拟合 Platt；
- 阈值网格固定为 0.01；
- ordinary 与各 power threshold FPR 固定 `<=0.20`；
- 不再比较新校准器，不依据 2039/2040 调参。

晋级门：至少 4/5 seeds 满足当前九项门禁，且五种子均值满足 ordinary AUPR `>=0.90`、recall `>=0.80`、power mean recall `>=0.90`、50 dB recall `>=0.85`、without-swept recall `>=0.70`。

## 阶段 F3：六类标签级路由

构造两个完全冻结的 development 方案：

- `L0_H2All`：六标签全部使用 H2；
- `L1_RawSaw_H2Others`：SawFM 概率和阈值来自 Raw+Platt，其他五类保持 H2 原概率与阈值。

禁止学习融合权重；禁止 Raw 改写其他标签；禁止使用 oracle power 或组合标签路由。

评估：macro/micro AUPR、macro-F1、每标签 recall/FPR/ECE、组合 exact match、Hamming loss，以及 power 40/45/50 dB 的 SawFM recall/FPR。

路由晋级门：五种子 macro-AUPR 不下降、macro-F1 提升、SawFM recall 提升至少 0.20、其他五类最大 FPR 增量不超过 0.01、组合 exact match 不下降超过 0.01，且至少 4/5 seeds 通过。

## 最终决策

- F2/F3 通过：冻结网络、校准和阈值，形成 locked-test 申请包；
- Raw 单标签通过但六类路由失败：保留为 SawFM 辅助专家或二阶段复核器，不替换主决策；
- 五种子确认失败：将三种子结果降级为探索性证据，停止接入主模型。

