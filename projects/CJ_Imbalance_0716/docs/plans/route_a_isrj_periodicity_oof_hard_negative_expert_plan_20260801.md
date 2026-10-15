# Route A ISRJ 周期专家 OOF 难负重训预注册方案

## 假设

既有 310 维 aligned HGB 能提取有效周期证据，但其 composition-matched 负样本内部仍包含高分难负，导致接入 H1 后在 FMB/VFT/高阶组合上产生误报。若用训练外分数识别并加权这些难负，专家自身的弱正—匹配负样本排序应改善。

## 固定协议

- 数据与角色：development-v3；仅 fit/validation/early；种子 2039—2043。
- 视图：弱 ISRJ 正样本与非 ISRJ 共现组合完全匹配的 ISRJ 负样本。
- 特征：冻结的 310 维 joint periodicity features；aligned 与 block-shuffled 两个家族。
- 难负挖掘：fit matched view 内 5 折 StratifiedKFold OOF HGB 分数；负样本最高四分位定义为难负。
- 模型容量固定：`HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1, early_stopping=False)`。
- H0/S0：普通类别平衡权重；H1/S1：类别平衡权重乘以难负权重 4。
- aligned 与 shuffled 使用完全相同的折数、分位点、权重和模型容量；不搜索任何超参数。

## 组件晋级门禁

H1 aligned 必须同时满足：

1. early matched AUROC 相对 H0 平均提升至少 0.010；
2. early matched AUPR 相对 H0 非负；
3. early matched AUROC 相对 S1 至少高 0.020；
4. 至少四个种子的 AUROC 相对 H0 为正；
5. 至少四个种子的 AUROC 高于 S1；
6. H1 validation-early AUROC 平均落差小于 0.15；
7. 五个种子均产生两类 OOF 目标和非空难负集合；
8. aligned/shuffled HGB 配置完全一致；
9. test-v3 边界完整。

只有组件门禁通过，才允许以完全冻结的 H1 专家重新训练一次低容量 H1+B2 融合头。失败则说明现有 310 维专家特征不能靠难负重加权获得稳定改进，需要新增能够区分 FMB/VFT 周期伪影的物理特征，而不是继续调权重。

