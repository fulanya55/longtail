# 路线 A：SawFM 专家受限接入验证方案（2026-07-25）

## 1. 已验证前提与本轮问题

上一阶段已经证明，冻结 F2 后的 stage2 低秩 SawFM 专家在同一 `receiver_noise_seed` 的 10/40/50 dB teacher-student 训练、45 dB 独立开发评价中有效：S2 三种子开发 AUROC 均值为 0.7972，且明显高于 H2 在同一弱可见度子域上的约 0.3999。与此同时，直接受限残差接回 H2 只带来约 0.0022 AUROC，说明当前瓶颈已从“专家能否学习弱证据”转为“如何识别何时相信专家、并避免破坏普通六类识别”。

本轮只验证专家路由与融合，不解冻 F2、H2 或专家，不重新扩宽主干，不使用 locked test。

## 2. 数据角色与防泄漏约束

- 主数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`。
- 运行环境：`D:\anaconda3\envs\PY10\python.exe`。
- 普通域路由 fit/validation：只从每个种子的合法 train role 选择 SawFM+扫频正样本及组合匹配 SawFM 缺失 hard negatives；排除上一轮专家训练使用的全部身份，再按 80%/20% 拆分。
- 弱域路由 fit/validation：按上一轮固定随机种子重建 45 dB 配对身份，并沿用原 80%/20% 拆分。
- 普通六类校准：原 calibration role，仅用于温度缩放。
- 普通六类阈值：原 threshold-selection role，仅用于阈值选择。
- 普通六类最终评价：原 early-stop development role。
- 弱域最终评价：early-stop role 内组合匹配的 45 dB SawFM 正样本与 SawFM 缺失 hard negatives。
- 全过程以 `receiver_noise_seed`/identity 绑定功率视图，禁止把多功率视图当独立样本；locked test 始终冻结。

## 3. 冻结网络与路由结构

### 3.1 冻结模块

1. F2 DFMS-IQ 主干：输入 `[B,2,3000]`，输出 H2 使用的 128 维全局特征及 stage2 复数特征图。
2. H2_PosBCE 六标签分类头：输出六类 logits，SawFM 为索引 1。
3. S2_TeacherHardNeg SawFM 专家：输入池化后的 stage2 `[B,64,375]`，含 rank=8 低秩 adapter、局部证据统计与单标签证据头，共 3651 个参数。

### 3.2 专家标量校准

只用路由 fit 数据拟合一维 Platt 映射：

`e_cal = a * e_raw + b`

该映射只校正专家和 H2 logit 的数值尺度，不访问校准、阈值或最终评价角色。

### 3.3 低容量路由器

路由输入为 8 个无功率元数据特征：

1. H2 SawFM logit；
2. 校准后专家 logit；
3. 专家与 H2 的差值；
4. 两者差值绝对值；
5. H2 其余五类最大 logit；
6. H2 其余五类平均 logit；
7. 三类扫频干扰最大 logit；
8. FMB/VFT 最大 logit。

标准化后进入 `Linear(8,1) + sigmoid`，仅 9 个可训练参数：

`z_saw = (1-w) * z_H2 + w * e_cal`

除 SawFM 外的五个 logits 完全保持 H2 原值。路由损失为 SawFM BCE，普通域与弱域总权重相同，并加入普通域 gate 均值惩罚 0.05，限制专家在普通样本上的无差别接管。

## 4. 对照组

- `R0_H2`：原 H2 六类 logits。
- `R1_ExpertReplace`：SawFM logit 完全替换为校准专家 logit，用于验证“专家单独很好但是否破坏普通域”。
- `R2_FixedBlend`：`alpha=0.00...1.00`、步长 0.05 的固定凸融合。
- `R3_LowCapacityRouter`：9 参数样本条件路由。

固定融合 alpha 与路由 checkpoint 均在 train 内双域 validation 上选择。选择目标为弱域 AUROC 与普通域 SawFM AUPR 的均值，同时要求普通域 SawFM AUROC、AUPR 相对 H2 均不下降超过 0.005；若无可行候选，回退 H2。

## 5. 训练超参数

- 种子：2041、2042、2043。
- 普通域：每个匹配组合 20 对，排除专家训练身份，80% fit、20% validation。
- 弱域：上一轮每组合 40 对，80% fit、20% validation，45 dB 仅供路由分数学习/选择。
- 专家 Platt 映射：L2 Logistic Regression，`C=1.0`，`max_iter=1000`。
- 路由优化器：AdamW，学习率 `1e-2`，weight decay `1e-3`。
- 路由训练：最多 300 epoch，全批训练，普通域 gate 惩罚 0.05，梯度裁剪 5.0，patience 40。
- 六类温度缩放：calibration role。
- 阈值：threshold-selection role，在负样本 FPR 上限 0.20 下最大化弱分量召回。

## 6. 晋级门槛

R3 只有同时满足以下条件，才允许进入“最后一个多尺度 stage 小学习率联合微调”：

1. 45 dB 匹配弱域 AUROC 均值不低于 0.70；
2. 相对 H2 的弱域 AUROC 均值增益不低于 0.20，且至少 2/3 种子为正；
3. 普通六类 early development 的 macro AUPR 相对 H2 不低于 -0.003；
4. 普通域 SawFM AUPR 相对 H2 不低于 -0.005；
5. SawFM negative FPR 增量不高于 0.02；
6. 弱域平均 gate 至少比普通域平均 gate 高 0.15，证明路由确实学习了证据条件，而非固定接管；
7. 身份交叉为 0，locked test 未使用，产物哈希完整。

若弱域有效但普通域不满足门槛，停止解冻，下一步改为“域外拒绝/不确定性门控”；若弱域也未满足，保留 S2 为独立诊断专家，不再连接主任务。

## 7. 审计产物

每个种子和每个变体保存：路由/校准参数、训练历史、普通域六类预测、45 dB 配对预测、温度、阈值、指标、身份与 split 哈希、checkpoint 哈希和 run receipt。总结果保存聚合均值、逐种子差值、晋级门槛与下一步决策。
