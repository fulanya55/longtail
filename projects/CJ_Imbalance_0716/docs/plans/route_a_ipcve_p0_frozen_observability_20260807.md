# Route A IP-CVE P0：冻结 B0 三状态边际证据准入

日期：2026-08-07  
状态：预注册。  
数据：Stage 0 正式三元组 development source；confirmation 和 test-v3 禁止访问。

## 1. 唯一可证伪假设

如果正式 B0 的冻结结构图已经包含目标 MFT/VFT/ISRJ 的可部署边际信息，那么用 fit 三元组的 `low-absent` residual 训练固定低复杂度 target-binding readout 后，单视图得分应在新角色中同时：

1. 区分 low 与 matched absent；
2. 满足 high > low > absent；
3. 超过 unmatched background、cyclic component binding、time-permuted input 和等维 raw-IQ random projection；
4. 不引入任何新模型或非目标 logit 漂移。

P0 是冻结表征诊断，不是最终分类器，也不构成 H0f prototype 的复活。若 P0 失败，禁止进入 P1。

## 2. 五个冻结重复

分别使用正式 B0 seed 2039--2043 checkpoint。每个 checkpoint 均已在 Stage -1 核验哈希并严格加载。B0 不重训、不微调、不校准、不选阈值。

## 3. 预注册 feature source

每个 source 以固定 Gaussian orthonormal projection 压缩到 64 维，投影 seed 在代码中固定：

- `global_128`：B0 全局 128 维向量；
- `temporal_bin8`：`[128,375]` temporal map 的 8 段平均；
- `stage3_mag_bin8`：stage3 复特征幅度的 8 段平均；
- `stage4_mag_bin8`：stage4 复特征幅度的 8 段平均。

等维信息控制：原始归一化 I/Q 先固定平均到 64 个时间 bin，再随机投影到 64 维。禁止使用 natural-visibility-graph entropy、max/top-k、learned attention、pair logit margin 或 delta prototype。

## 4. 固定 readout

- 输入：fit 角色的 64 维 `low-absent` residual。
- 目标：MFT/VFT/ISRJ 三分类 target binding。
- 模型：`StandardScaler + LogisticRegression(C=1.0, lbfgs, max_iter=2000)`。
- readout 只用于诊断；对单视图应用其三个 decision scores，计算各目标 low-vs-absent AUROC/AUPR 和三状态顺序。
- 所有源、控制、角色和 B0 seed 使用相同 readout 容量。

## 5. 控制

- `matched_true`：真实同 identity residual 和真实 binding；
- `unmatched_background`：同 target、同 nuisance 内循环错配 absent；
- `component_permuted`：matched residual，但 binding 固定循环 MFT→VFT→ISRJ→MFT；
- `time_permuted`：同一固定时间索引置乱后重新提取所选 B0 source；
- `raw_iq_random_projection`：不经过 B0 的等维 raw I/Q 控制。

## 6. 角色与单次选择

- fit：训练 readout；
- early：比较四个 B0 source，固定选择 `matched_true` 相对 strongest preselection control 的平均 weak AUROC 增量最大的唯一 source；
- evaluation：只对已选 source 及其全部控制执行一次，不评价未选 source。

source selection receipt 必须在 evaluation 前写出。

## 7. 固定指标

- target-binding accuracy；
- 各目标及三目标平均 low-vs-absent AUROC/AUPR；
- 各目标 `score(high)>score(low)>score(absent)` 比例；
- evaluation 相对 strongest control 的逐 seed/逐标签差值；
- 五 B0 seed bootstrap 95% CI；
- evaluation-to-early gap；
- 冻结 B0 非目标概率漂移，仅作安全画像。

## 8. P0 硬门

相对 evaluation 上每 seed 的 strongest control，同时满足：

1. 三目标平均 weak-vs-absent AUROC 增量 `>= +0.030`；
2. 五 seed bootstrap 95% CI 下界 `>0`；
3. 至少 4/5 seed 为正；
4. 任一目标平均增量不低于 `-0.010`；
5. true binding accuracy `>=0.60`，且相对 strongest binding control 的五 seed CI 下界 `>0`；
6. 三目标的 true order fraction 均高于各自 strongest control；
7. evaluation-to-early mean weak AUROC gap `<0.15`；
8. B0 冻结，故新增非目标漂移必须为 0。

标签特定通过门定义为：该标签 weak AUROC 增量 `>=+0.020`、95% CI 下界 `>0`、至少 4/5 seed 为正，且 order fraction 高于控制。

分支：三标签通过才允许统一 P1；1--2 标签通过只允许窄标签候选并重新审计 H0f 重叠；0 标签或总体门失败则关闭 IP-CVE，并转向更长/多脉冲观测协议，禁止扩大分类器。
