# Route A 冻结 Ridge 可见度条件融合 development-v3 预注册

## 1. 单一可证伪假设

E1 的对齐预测可见度相对打乱对照只获得弱 AUROC `+0.00310`，可能不是“可见度对分类无用”，而是 D2 的共享 rank-32 多任务瓶颈把可见度预测质量从冻结全维探针的弱/强 AUROC `0.88210` 降低到约 `0.71469`。若使用只在 fit 正样本上训练、随后完全冻结的 512 维 Ridge 可见度预测器，样本对齐的可见度条件融合应显著优于同边际分布的角色内打乱对照。

## 2. 固定模型与容量控制

- 六标签主干、C0/C4 logits、RawSaw SawFM 专家及其 checkpoint 全部冻结；
- presence 分数仍来自冻结 D2 presence head，避免同时改变分类证据来源；
- 每个目标标签使用一个 Ridge (`alpha=10`) 从 final-temporal 512 维标准化统计预测相对功率差，只使用 fit 中该标签为正且参数有效的样本；
- Ridge 训练完成后完全冻结，推理不使用真实功率、弱/强 mask 或可见度真值；
- 融合器保持 E1 的 12 个标量参数和 `0.50` 有界残差；
- E2 RidgeShuffled 与 E3 RidgeAligned 使用完全相同的 Ridge、presence、融合器容量、优化预算和阈值协议，唯一差异是预测可见度是否在各角色内部打乱。

## 3. 数据角色与禁止访问

- 仅使用 development-v3，SHA-256 固定为 `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`；
- 每个种子的 fit 用于拟合 Ridge 和融合器，validation 只用于融合早停，threshold/calibration/early 角色保持原协议；
- 角色内打乱使用固定随机置换并保存置换 SHA-256；
- 禁止访问历史 locked test、test-v2 和 sealed test-v3；每个 receipt 明确记录三个 `used=false`。

## 4. 训练与阈值

- Ridge：`alpha=10`，不搜索；输入均值/方差只由 fit 计算；
- 可见度映射：以 `-10 dB` 为中心、温度 `2.5 dB` 的固定 sigmoid，不搜索；
- 融合器：AdamW，学习率 `1e-2`，weight decay `1e-4`，最多 200 epoch，patience 20，gradient clip 5；
- 损失：目标三标签 BCE + residual L2 `0.02` + bias L2 `0.05`；
- 校准与逐标签阈值完全复用统一模型的 development-v3 协议，三目标平均负样本 FPR 上限 `0.20`。

## 5. 对照与指标

- C0 F2+C6；
- C4 SameCapacity-NoVisibility；
- D2 LowRank-VisibilityAux；
- E1 Aligned-D2-Visibility（上一轮冻结结果）；
- E2 RidgeShuffled（同容量机制对照）；
- E3 RidgeAligned（候选）。

主要指标：三目标弱召回、弱正-vs-负 AUROC；次要指标：六标签 macro-AUPR、macro-F1、强正召回、平均 FPR、ISRJ FPR、SawFM AUPR/弱召回。另报告 Ridge 的正样本 Spearman 和弱-vs-强 AUROC，验证可见度预测质量。

## 6. 晋级门槛

E3 必须同时满足：

1. 相对 E2：弱召回 `>=+0.010`；
2. 相对 E2：弱正-vs-负 AUROC `>=+0.010`；
3. 相对 E2：macro-AUPR 不低于 `-0.003`；
4. 相对 E2：强正召回不低于 `-0.010`；
5. 相对 C0：弱召回 `>=+0.030`、弱 AUROC `>=+0.015`；
6. 三目标平均 FPR `<=0.20`；
7. 至少 4/5 种子相对 C0 的弱召回增益为正；
8. SawFM 指标保持不变；
9. 推理不使用真实可见度，身份和测试边界安全。

## 7. 失败后的动作

若 Ridge 自身保持高可见度预测质量但 E3 仍未通过，则排除“仅用全局预测可见度调制最终 logits 可形成主要分类收益”的假设，停止 final-logit 可见度后处理路线。后续转向标签条件局部证据结构本身，并将可见度只作为训练期分层和机制分析变量，不再搜索融合温度、残差上限或 Ridge 超参数。
