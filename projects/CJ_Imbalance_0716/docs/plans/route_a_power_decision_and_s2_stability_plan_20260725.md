# 路线 A：功率域决策校准与 S2 稳定性验证方案（2026-07-25）

## 1. 目标

上一轮统一 `alpha=0.80` 的冻结 C2 在 35/40/45/50 dB 上显著改善 SawFM AUROC，但普通域阈值迁移到 45/50 dB 后弱正召回接近 0，且原种子 2043 的绝对 AUROC 明显落后。本轮将两个问题分开验证：

1. **决策层路线 D**：冻结 F2/H2/S2/C2，只校准 SawFM 分数并选择一个不读取功率元数据的统一阈值。
2. **稳定性路线 S**：固定种子 2043 的数据、结构、损失与训练预算，只改变三个网络初始化，判断 S2 失稳是否属于优化偶然性。

只有 D、S 两条门槛同时通过，才允许解冻最后一个多尺度 stage。

## 2. 路线 D：功率域决策校准

### 2.1 身份角色

- 从每种子合法 train role 中排除：原 S2 专家训练身份、上一轮普通路由训练身份。
- 剩余身份按 28 个 SawFM+扫频组合选择每组合 20 对正样本/hard negative。
- 每组合前 10 对构成 `power_calibration`，后 10 对构成 `power_threshold`；身份完全互斥。
- 每个身份取 40/45/50 dB 三个视图，但统计和拆分以 identity 为单位，不把视图作为独立身份。
- 普通 calibration role 与 power_calibration 一起拟合无功率元数据的一维 Platt 映射。
- 普通 threshold-selection role 与 power_threshold 一起约束统一阈值。
- early-development 的普通域及 40/45/50 dB 只做最终评价。

### 2.2 冻结模型

`C2 logit = 0.20 * H2 SawFM logit + 0.80 * calibrated S2 logit`

F2、H2、S2 和 alpha 全部冻结。本轮只允许学习：

- 一维 Platt：`p = sigmoid(a*z+b)`，2 个参数；
- 一个统一 SawFM 阈值。

### 2.3 对照

- `D0_OrdinaryThreshold`：上一轮普通 calibration/threshold。
- `D1_PowerPlatt0p5`：双域 Platt，固定阈值 0.5。
- `D2_DualDomainRobust`：双域 Platt，并在独立 power_threshold 上选择统一阈值。

D2 阈值最大化 40/45/50 dB 的宏平均正样本召回，同时要求每档 hard-negative FPR≤0.20、普通 threshold role 的 SawFM FPR≤0.20。并列时优先更高普通 F1、再优先更接近 0.5。

### 2.4 路线 D 门槛

- 40/45/50 dB 三档平均弱正召回≥0.50；
- 每档弱正召回≥0.35；
- 每档 hard-negative FPR≤0.20；
- 普通 early SawFM FPR 相对 D0 增量≤0.02；
- 普通 early SawFM recall 相对 D0 不下降超过 0.05；
- 3 个种子中至少 2 个满足上述单种子功率约束；
- 身份交叉为 0，locked test 未使用。

## 3. 路线 S：种子 2043 的 S2 多初始化重训

### 3.1 固定项

- 数据种子与 split：2043；
- 每组合 40 对，原 80/20 fit/validation；
- teacher 10 dB，student 40/50 dB，45 dB checkpoint validation；
- hard negatives、损失权重、rank=8 adapter、30 epoch、patience=8、AdamW `3e-4` 全部保持上一轮 S2 不变。

### 3.2 变化项

仅改变三个初始化种子：274301、274302、274303。

### 3.3 独立评价

- early-development 同身份匹配对的 35/40/45/50 dB；
- 指标为 S2 原始专家 AUROC/AUPR，不通过 Platt 或阈值掩盖表征排序；
- 按组合规模重点报告六分量组合。

### 3.4 路线 S 门槛

- 至少 2/3 初始化的四档平均 AUROC≥0.65；
- 至少 2/3 初始化的最差档 AUROC≥0.58；
- 三初始化四档均值的标准差≤0.10；
- 至少 2/3 初始化的六分量组合四档平均 AUROC≥0.55；
- 相比原 2043 S2，至少 2/3 初始化的四档平均 AUROC提高≥0.10；
- locked test 未使用，训练与开发身份交叉为 0。

## 4. 最终决策

- D、S 均通过：执行最后 stage 小学习率联合微调，主干 `1e-5`，专家/融合头 `1e-4`，冻结 C2 并行对照。
- D 通过、S 失败：保留冻结 C2，但先改进 S2 checkpoint 稳定性，不解冻。
- D 失败、S 通过：保持表征冻结，继续决策层的分布稳健校准。
- D、S 均失败：停止当前接入路线，S2 仅作为独立弱证据诊断专家。

## 5. 审计产物

路线 D 保存身份 manifest、Platt 参数、阈值搜索、普通/功率预测、逐档召回/FPR、receipts。路线 S 保存三个 checkpoint、history、四档预测、逐组合指标、初始化种子与所有源 checkpoint/split/identity 哈希。locked test 始终冻结。
