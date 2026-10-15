# Route A Backbone M1：固定复数抗混叠下采样结果与关闭结论

日期：2026-08-06  
候选：`B1_F2_DFMS_AntiAlias = B0 + M1`  
证据等级：`five_seed_development_only_m1_candidate`  
决策：`fail_stage2_close_candidate`

## 1. 唯一结构改动

在现有三次 stride-2 AvgPool 前，对实部和虚部分别施加同一固定五点二项式低通核 `[1,4,6,4,1]/16`。该模块没有可训练参数；其余主干、分类器、损失、五 seed、batch order、训练预算、校准和阈值规则均与 B0 相同。

真实 MAT/CUDA smoke 已先完成，且仅标记为 `smoke_only_not_formal_model_evidence`。正式五 seed 均从对应 seed 的确定性随机初始化开始，未加载 B0 权重。

## 2. 相对 B0 的核心结果

| 指标 | B0 | M1 | 配对均值差 | 配对 bootstrap 95% CI | 正增益 seed |
|---|---:|---:|---:|---:|---:|
| macro-AUPR | 0.843746 | 0.848401 | +0.004655 | [+0.002605, +0.006706] | 5/5 |
| 三目标弱正 AUROC | 0.555864 | 0.555509 | -0.000354 | [-0.008331, +0.008087] | 2/5 |
| 三目标弱正 recall | 0.237524 | 0.240318 | +0.002794 | [-0.005525, +0.015090] | 2/5 |
| 三目标强正 recall | 0.771434 | 0.771315 | -0.000119 | [-0.006069, +0.006147] | 2/5 |
| 三目标负样本 FPR | 0.195215 | 0.194677 | -0.000538 | [-0.008011, +0.006935] | — |

macro-AUPR 的小幅改善稳定存在，但未达到预注册的 `+0.005` 效应量门槛。更关键的是，弱正 AUROC 没有改善，弱正 recall 增益只有 `+0.0028`，且二者都只有 2/5 seed 为正，远低于预注册要求。因此 M1 不能支持“抗混叠增强弱证据保留”的机制主张。

## 3. 逐标签归因

- FMB AUPR `+0.03982`，5/5 seed 为正；SawFM AUPR `+0.01395`，5/5 为正。
- SinFM AUPR `-0.02005`，95% CI `[-0.02821,-0.01143]`，5/5 seed 均下降；SinFM 弱正 AUROC `-0.01603`，5/5 下降。这是明确的系统性安全标签退化。
- MFT 弱正 AUROC `+0.00484`，但只有 2/5 为正；弱正 recall `+0.01432`，仍低于目标平均门槛。
- VFT 弱正 AUROC `-0.00834`，5/5 下降；弱正 recall `-0.00650`，4/5 下降。
- ISRJ 弱正 AUROC `+0.00244`，只有 2/5 为正，不能形成稳定增益。

结果表明固定低通主要重分配了不同调制类型的频谱偏好：它显著帮助 FMB，却稳定损害 SinFM 与 VFT 的弱证据排序。它不是统一的弱证据保留机制。

## 4. 晋级门槛

通过：绝对目标 FPR、目标强正 recall 安全界、任一目标 AUPR 下降不超过 0.01、成本收据、数据与 test-v3 边界。  
失败：macro-AUPR 效应量、弱正 AUROC、弱正 recall、4/5 seed 一致性、非目标标签系统性退化。  
总决策：不进入 equal-capacity 或 information-destroyed 控制；永久关闭 M1 及其换核、加宽、调阈值版本。

## 5. 代价与工件

- 参数：257,511，与 B0 完全相同。
- hook 画像 MACs：211,857,536；补计固定 `F.conv1d` 后估算 213,177,536，较 B0 增加约 0.623%。
- batch-1 CUDA latency：M1 五 seed 均值 5.742 ms，B0 为 4.056 ms；本机测量增加约 41.6%。
- 最大峰值 CUDA memory：1,187,308,032 bytes。
- M1 summary SHA256：`3a120bb4ec580e98c1b3bf2864fe65a9db706c5b937f99b1a11832148424afbd`。
- comparison SHA256：`2516391c9b5971fad070145de18a9dcb8d1bb22ea722c37549be7f4d0e3ce528`。
- terminal receipt 状态：`completed`，正式运行耗时 295.895 s。

结论只适用于 fresh development-v1。test-v2、locked test、test-v3 均未使用，test-v3 继续密封。下一步按 Goal 独立实施 M2（CSFP），M1 的失败不影响 M2 的预注册与检验。
