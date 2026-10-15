# Route A VTSE 固定低权重强正锚点 development-v3 结果

日期：2026-07-31  
证据等级：五种子 development-v3；test-v3 未使用。  
结论：soft anchor 略优于 hard mask，但仍不如同参数量 all-positive，对该损失线停止调权。

## 1. 结果

| 模型 | Macro-AUPR | Macro-F1 | 三目标弱召回 | 三目标弱 AUROC | 强正召回 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.85050 | 0.73409 | 0.29937 | 0.58902 | 0.79720 | 0.18812 |
| hard-mask C3 | 0.87838 | 0.76623 | 0.30896 | 0.59167 | 0.79870 | 0.19177 |
| all-positive C4 | 0.88128 | 0.77043 | 0.31192 | 0.59568 | 0.81290 | 0.19032 |
| soft-anchor S1 | 0.87864 | 0.76675 | 0.30912 | 0.59227 | 0.80055 | 0.19156 |

S1 相对 hard mask：Macro-AUPR `+0.00026`，Macro-F1 `+0.00052`，弱召回
`+0.00016`，弱 AUROC `+0.00061`。方向略有改善，但弱召回 bootstrap 区间跨 0。

S1 相对 all-positive：Macro-AUPR `-0.00264`，Macro-F1 `-0.00368`，弱召回
`-0.00280`，弱 AUROC `-0.00341`，强正召回 `-0.01236`。因此固定 0.20 强正锚点
仍不能建立优于普通正类学习的可见性机制证据。

## 2. 机制判断

hard mask 过度删除强正是问题之一，但不是全部根因。soft anchor 改善幅度很小，而且
VBF 仍在 2039/2040/2043 三个种子第 1 epoch 早停。核心问题更接近：二值 weak mask
把相对功率的连续变化压缩为 10 dB 单一边界，训练目标在阈值附近不连续，且弱正样本
数量不足以稳定估计独立“弱类方向”。

## 3. 决策

- 不搜索 strong weight，不把 0.20 改为 0.1/0.3/0.5；
- hard-mask 与固定 soft-anchor 均记为未通过；
- 保留 RawSaw 统一路由和 all-positive temporal expert 作为当前最强 development 基线；
- 下一轮只验证连续物理可见性加权：仍学习“正类存在”，但依据相对功率差连续提高弱正
  梯度，避免把弱正另立为数据稀少的新类。

## 4. 证据边界

本轮未访问历史 locked test、test-v2 或 test-v3。候选未晋级，不冻结、不授权 test-v3。

关键汇总：
`outputs/tapc/route_a_vtse_soft_strong_anchor_v3/route_a_vtse_soft_strong_anchor_v3_summary.json`。
