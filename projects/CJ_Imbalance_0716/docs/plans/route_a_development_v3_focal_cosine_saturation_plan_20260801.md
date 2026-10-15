# Route A development-v3 Focal cosine收敛终审预注册

## 目的

36-epoch恒定学习率 Focal 已成为最强传统基线，但 4/5种子最佳点命中预算上限。本轮只解决一个问题：使用固定、常规且不搜索的余弦学习率衰减后，强 Focal 是否达到实际平台，并能否解释 F3 的 ISRJ 弱周期收益。

本轮结束后不再通过无限增加 epoch反复扩张传统损失预算；后续转入分类头和同容量结构基线。

## 固定协议

- 数据、角色、种子、F2初始化、RawSaw、校准、阈值和可训练范围与前两轮完全一致；
- 只训练 Focal：`gamma_pos=0, gamma_neg=3`；
- 从原始 F2重新训练，不从36-epoch checkpoint续训；
- AdamW，classifier初始 LR `5e-4`，shared初始 LR `5e-5`，weight decay `1e-4`；
- cosine multiplier 从1.0衰减到0.01，因此终点 LR分别为 `5e-6` 与 `5e-7`；
- 完整运行72 epochs，不使用早停截断；最佳 checkpoint仍按 validation common-five macro-AUPR、target weak AUROC破同分选择；
- batch 64，gradient clip 5，train seed `seed+73000`；
- test-v3保持封存。

## 实际平台判定

逐种子同时满足：

1. 最后12 epochs validation common-five macro-AUPR的首尾增量绝对值 `<=0.0005`；
2. 最后12 epochs 的最大值减最小值 `<=0.0010`；
3. 最终 classifier LR `<=5.5e-6`。

五个种子至少4个满足则认为最强传统损失达到“实际平台”。该判定不要求最佳 epoch早于72，因为余弦尾部可能产生微小单调改善。

## 方法判定

- 在冻结 Focal-36 与 Focal-cosine-72 中按 mean macro-AUPR选择最终强传统基线；
- F3相对最终强基线仍使用原门槛：ISRJ AUPR为正、弱召回 `>=+0.030`、弱 AUROC `>=+0.010`、弱召回和弱 AUROC均至少4/5正向、F3 ISRJ FPR `<=0.20`；
- 若平台门和 F3门同时通过，则支持“F3 的 ISRJ弱证据增益不能由主流损失、三倍预算或标准余弦收敛解释”；
- 若平台门通过但 F3门失败，则以 Focal为主基线并收缩创新主张；
- 若平台门失败，保留收敛限制，但不再继续无界延长训练。
