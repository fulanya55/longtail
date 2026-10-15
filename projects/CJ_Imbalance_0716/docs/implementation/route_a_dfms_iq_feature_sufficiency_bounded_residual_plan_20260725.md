# 路线A：F2特征充分性与硬约束弱证据残差试验方案（2026-07-25）

## 1. 试验目的

上一轮 Q1/Q2/Q3 失败不能直接等价为“F2 主干没有学到弱分量特征”。已有冻结全局特征线性 probe 的三种子结果为：Macro-AUPR 约 0.8657、弱正—负样本 AUROC 约 0.6448；H2 的对应弱 AUROC 约为 0.6524。说明全局 128 维表示已经包含中等强度的弱分量信息，但标签条件池化和 SawFM 专家没有把信息稳定转化为分类收益。

本轮回答两个问题：

1. F2 的 global、融合后 temporal map、stage2 complex map 分别保留了多少弱分量可辨识信息？
2. 将弱证据读出限制为零均值、硬幅度上界和低置信度门控后，能否安全改善 H2？

锁定测试集继续冻结。本轮只使用训练清单、head-validation、early-stop development、calibration 和 threshold-selection 角色。

## 2. 固定基础模型与数据协议

- 主干：每个 seed 对应的 `F2_DFMS_Full` 检查点，完全冻结。
- 分类器：每个 seed 对应的 `H2_PosBCE`，完全冻结。
- seeds：2041、2042、2043。
- label-LT：复用 F2 的 `longtail_manifest.json`，不重新抽样。
- 弱分量：标签存在，且相对同样本最强阳性分量不高于 -10 dB。
- 训练清单内部拆分：沿用 `57000 + seed` 的 80%/20% fit/validation。
- early、calibration、threshold 每个角色最多 300 个样本。
- 所有 split、manifest、F2、H2 哈希写入回执。

## 3. 冻结特征接口与描述量

### 3.1 G：global

直接使用 F2 的 `[B,128]` 全局特征。

### 3.2 T：temporal-safe

对融合后 `[B,128,375]` 时序图逐通道计算：

```text
mean + std + max + top8-mean -> [B,512]
```

该描述量不依赖可学习注意力，不使用上一轮有偏的 query heatmap。

### 3.3 S2：stage2-physics

从 stage2 complex map `[B,32,1500]` 构造六类物理描述量：

1. magnitude；
2. absolute envelope difference；
3. adjacent phase cosine；
4. adjacent phase sine；
5. absolute phase-cosine difference；
6. absolute phase-sine difference。

每类描述量逐通道计算 `mean/std/max/top8-mean`，得到 `[B,768]`。

### 3.4 F：G+T+S2

拼接后为 `[B,1408]`，只用于线性可辨识性和受限残差验证，不增加深层 MLP。

所有特征仅使用 fit 子集均值/标准差标准化。

## 4. 阶段一：oracle weak-positive probe

对 G、T、S2、F 四种表示分别训练六个独立线性二分类器。标签 `c` 的训练样本只包含：

- 正类：`label[c]=1 AND weak[c]=1`；
- 负类：`label[c]=0`；
- 强阳性：从该 probe 中排除，防止强模式主导弱证据边界。

参数：

| 参数 | 数值 |
|---|---:|
| classifier | Linear(D,1) |
| loss | class-balanced BCEWithLogits |
| optimizer | AdamW |
| learning rate | 1e-3 |
| weight decay | 1e-3 |
| epochs | 60 |
| patience | 10 |
| batch | full fit subset per label |
| checkpoint selection | validation weak-vs-negative AUROC，AP 同分决胜 |

oracle probe 是特征可辨识性诊断，不是生产分类器。

### 特征充分性判据

- 总体充分：某表示相对 H2 的弱正—负 AUROC 平均提高至少 0.020，且至少 2/3 seeds 提高。
- stage2 SawFM 充分：S2 的 SawFM 弱 AUROC 至少 0.55，且相对 H2 提高至少 0.020、至少 2/3 seeds 提高。
- 标签级不足：某标签在 G/T/S2/F 全部表示上的三种子弱 AUROC 均低于 0.55，则该标签当前冻结特征证据不足。

## 5. 阶段二：硬约束弱证据残差

每种表示复用对应 oracle probe 的线性分数 `raw[c]`，在 fit 子集上记录逐标签中心和标准差：

```text
z[c] = (raw[c] - fit_mean[c]) / fit_std[c]
uncertainty_gate[c] = exp(-abs(H2_logit[c]))
residual[c] = 0.20 * tanh(z[c]) * uncertainty_gate[c]
final_logit[c] = H2_logit[c] + residual[c]
```

性质：

- 残差硬上界为 0.20；
- fit 分布中心近似为零，避免再次学习标签先验平移；
- H2 已高置信的强正/明确负样本调整接近零；
- 不进行端到端联合训练，便于把变化归因于特征层级。

变体：

| 变体 | 弱证据特征 |
|---|---|
| C0_H2 | 无，冻结基线 |
| R1_GlobalBound | G |
| R2_TemporalBound | T |
| R3_Stage2Bound | S2 |
| R4_FusedBound | G+T+S2 |

每个变体独立进行标量温度校准和 FPR 约束阈值选择。主指标仍在 early-stop development 角色报告。

### 残差晋级门

候选相对 C0_H2 必须同时满足：

- Macro-AUPR 下降不超过 0.003；
- 弱正—负 AUROC 提高至少 0.010；
- 弱分量召回提高至少 0.020；
- Hamming loss 增量不超过 0.005；
- negative FPR 增量不超过 0.010，且绝对值不超过 0.25；
- SawFM AUPR 不下降；
- 弱 AUROC 和弱召回至少 2/3 seeds 为正。

只有残差晋级门通过，下一轮才允许解冻 stage4/temporal 最后阶段；否则继续保留 F2+H2。

## 6. 输出目录

```text
outputs/tapc/route_a_dfms_iq_feature_sufficiency/
docs/results/route_a_dfms_iq_feature_sufficiency_results_20260725.md
```

保存内容包括：probe 权重、标准化统计、训练历史、逐角色 logits、温度、阈值、完整指标、逐标签弱证据指标、哈希、回执和晋级决定。

## 7. 执行状态（2026-07-25）

- GPU smoke：1 seed × 4 表示完成；
- 正式试验：3 seeds × 4 oracle probes × 6 labels 完成；
- 硬约束残差：3 seeds × C0/R1/R2/R3/R4 完成；
- 所有残差晋级门均失败；
- `unfreeze_decision=stop_before_unfreeze`；
- locked test 未使用；
- 完整结果见 `docs/results/route_a_dfms_iq_feature_sufficiency_results_20260725.md`。
