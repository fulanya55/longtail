# Route A 分层组合匹配负样本可分性 development-v3 结果

## 1. 结论

Global、stage2、stage3 和 final-temporal 均未通过弱正—组合匹配负样本可分性门槛，不允许直接进入冻结局部专家训练。Final-temporal 是当前最佳层，但三标签平均 AUROC 仅 `0.58002`，低于 `0.620` 绝对门槛；其 validation→early 落差达到 `0.20874`，说明线性探针严重过拟合 development 内部组合。

结果证明当前弱分量问题已深入到冻结 F2 表征，而不仅是分类器、阈值或损失设计问题。test-v3 未访问。

## 2. 五种子结果

| 表征 | 特征维数 | early AUROC | early AUPR | validation→early AUROC gap |
|---|---:|---:|---:|---:|
| Global | 128 | 0.57451 | **0.25140** | 0.23595 |
| Final-temporal | 512 | **0.58002** | 0.24318 | 0.20874 |
| Stage2 physics | 768 | 0.54676 | 0.20628 | 0.17252 |
| Stage3 physics | 1152 | 0.56815 | 0.22005 | **0.16953** |

相对 Final-temporal：Global `-0.00551`，stage2 `-0.03326`，stage3 `-0.01186`；没有候选达到 `+0.030`，stage2 的置信区间完全低于零。

Final-temporal 的逐标签 AUROC为 MFT `0.58729`、VFT `0.56015`、ISRJ `0.59262`，三类均没有达到稳定局部专家所需的 `0.62`。

## 3. 机制含义

1. stage2 并未保留比 final-temporal 更直接的弱分量物理证据，早分叉本身不能解决问题；
2. stage3 泛化落差略小，但绝对可分性仍不足，不能以较小过拟合换取低质量边界；
3. Global AUPR较高但 AUROC 与泛化落差不合格，说明其排序依赖组合先验，而非稳定弱证据；
4. 组合匹配负样本消除了“其他标签组合不同”这一捷径，因此本轮比普通负样本 AUROC 更能反映真实标签证据；
5. 继续在冻结 F2 上增加 attention、MLP 或 loss 权重，只会在不足的表征上拟合 development 组合。

## 4. 下一步

先审计独立原始 I/Q 物理证据是否仍可分：MFT 使用延迟自相关，VFT 使用 STFT 时频脊，ISRJ 使用相位复位/包络差分周期性。该轮先用固定物理摘要加同协议线性探针，避免直接训练大网络。只有固定原始 I/Q 证据相对 final-temporal 提升至少 `0.03` 且绝对 AUROC达到 `0.62`，才允许将对应分支实现为可训练专家。

主要产物：`outputs/tapc/route_a_layerwise_matched_negative_separability_v3/route_a_layerwise_matched_negative_separability_v3_summary.json`。
