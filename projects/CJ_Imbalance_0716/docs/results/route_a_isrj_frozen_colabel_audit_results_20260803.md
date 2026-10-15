# Route A：冻结 ISRJ 周期专家的 MFT/VFT 共标签分层审计

## 结论

这是对 H0g 与 H0h 已完成、冻结 B0/B2 预测的非选择性审计，不包含任何重训、校准、阈值或融合修改。两种外部条件下，ISRJ `B2-B0` 弱正—负样本 AUROC 在所有九个受支持的 MFT/VFT 共标签条件中均为正，且五种子 bootstrap CI 下界均大于零。

这强化了一个有限结论：**已冻结的 aligned-periodicity ISRJ 排序增益并不只出现在单一的 MFT/VFT 共现条件。** 但固定阈值下的弱召回与 FPR 会随共标签显著移动，因此不能以此主张统一阈值已在全部复合上下文中安全，也不能将后验分层用作重新选择阈值的依据。

## B2−B0：ISRJ weak-vs-negative AUROC

| 外部来源 | 全体 | MFT 在 | MFT 不在 | VFT 在 | VFT 不在 | MFT0/VFT0 | MFT0/VFT1 | MFT1/VFT0 | MFT1/VFT1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H0g：高副本 ISRJ | +0.09965 | +0.09342 | +0.10772 | +0.06871 | +0.14121 | +0.17627 | +0.08414 | +0.13833 | +0.05451 |
| H0h：AR(1) 噪声 | +0.08037 | +0.10813 | +0.06252 | +0.06386 | +0.10415 | +0.11498 | +0.04150 | +0.12246 | +0.09494 |

全部格子均由五个训练种子支持，AUROC 差的 seed-bootstrap 95% CI 下界为正。较弱但仍正的条件是 H0h 的 `MFT0/VFT1`（`+0.04150`, CI `[+0.03016,+0.05425]`）与 H0g 的 `MFT1/VFT1`（`+0.05451`, CI `[+0.04832,+0.06045]`）。

## 必须保留的阈值风险边界

- H0g `MFT1/VFT1`：排序 `+0.05451`，但固定阈值弱召回 `−0.02745`，FPR `−0.06500`。这是保守阈值导致排序收益未转化为召回，而不是排序机制失效。
- H0g `MFT0/VFT0`：排序 `+0.17627`、弱召回 `+0.32190`，但 FPR `+0.11929`。不能据此为该上下文降低阈值，因为本轮禁止选择。
- H0h `MFT1/VFT0` 与 `MFT1/VFT1`：排序分别 `+0.12246/+0.09494`，弱召回与 FPR 均向有利方向变化，说明 AR(1) 条件下该 MFT 共现环境未暴露同样的阈值代价。

这些现象说明 B2 的**排序**机制跨上下文存在，但“排序→固定阈值召回/FPR”的映射受背景和校准分布影响；它们不能与现有全局安全门混为一谈。

## 审计与不可越界条件

- 读取输入：H0g/H0h 五个种子的现有 `B0_H1_Frozen` 与 `B2_H1AlignedPeriodicity_Frozen` 预测，共 20 个预测文件；所有文件 SHA-256 已记录在机器账本中。
- 同一 seed 的两臂 `targets` 与 `weak_mask` 均逐元素一致；H0g/H0h summary 均确认 completed、external_refit=false、locked/test-v2/test-v3 未使用。
- 不产生新数据、不拟合任何参数、不修改预测、不使用结果选择阈值。

机器账本：[route_a_isrj_frozen_colabel_audit.json](C:/LJQ_code/CJ_Imbalance_0716/outputs/tapc/route_a_isrj_frozen_colabel_audit/route_a_isrj_frozen_colabel_audit.json)。脚本：[analyze_route_a_isrj_frozen_colabel_audit.py](C:/LJQ_code/CJ_Imbalance_0716/experiments/analyze_route_a_isrj_frozen_colabel_audit.py)。
