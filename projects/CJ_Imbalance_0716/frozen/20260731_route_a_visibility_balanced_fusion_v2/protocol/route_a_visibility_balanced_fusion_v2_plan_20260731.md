# Route A：可见性分层均衡融合器确认方案

## 问题定位

VTSE 平方根门已使弱正—负 AUROC 提升 0.0172，但 macro-F1 和弱召回仍只提升
0.0069/0.0269。说明专家排序有效，失败发生在手工 score-to-residual 映射，而不是
特征或专家本身。

## 模型

冻结 F2、C6 和三标签 VTSE。对 MFT/VFT/ISRJ 分别学习：

`raw_k = softplus(alpha_k) * normalized_expert_k + bias_k`

`gate_k = sqrt(4*p_base_k*(1-p_base_k))`

`residual_k = 0.50*tanh(raw_k)*gate_k`

`final_logit_k = base_logit_k + residual_k`

只有 6 个可训练标量（3 个 alpha、3 个 bias）。非目标标签完全不变，残差仍有
±0.50 硬上限。

## 标签内可见性均衡损失

对每个目标标签把 optimization-fit 样本分为：弱正、强正、负样本。分别计算
BCE 均值后等权平均，使三层各占 1/3，而不是由数量最多的强正或负样本主导。

总损失：

`L = mean_k((L_weak + L_strong + L_negative)/3)
     + 0.02*mean(residual^2) + 0.05*mean(bias^2)`

optimization 内固定 80/20 fit/validation；score orientation/center/scale 只由 fit
确定；validation 用于早停。最多 200 epoch，AdamW lr=0.01，patience=20。

## 评估与门槛

calibration、threshold-selection、early 角色和首轮 VTSE 完全一致。晋级门保持：
macro-AUPR 非劣、macro-F1 +0.010、目标弱召回 +0.030、目标弱 AUROC +0.015、
FPR<=0.20、强召回非劣、4/5 种子弱召回正增益、SawFM-present 非劣、残差不越
±0.50。test-v2 与历史 locked test 均保持封存。

