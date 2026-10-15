# Route A H1+B2 周期难负分数解耦预注册方案

## 假设

两轮上下文阈值均能降 FPR，但无法保留至少 80% 的弱召回增益，说明主要问题是校准前弱正与周期性难负的排序重叠。本轮只检验：在冻结 H1/B2 分数上，对周期增量最高的 ISRJ 负样本增加训练权重，能否改善弱正—负样本排序。

## 模型与固定设置

- 主干、H1、RawSaw、周期专家和既有 B2 全部冻结。
- 输入特征固定为：B1 ISRJ 分数、`delta=B2-B1`、`delta*sigmoid(FMB logit)`、`delta*max(sigmoid(non-ISRJ logits))`。
- 低容量头：标准化后 `LogisticRegression(C=0.1, solver='lbfgs', max_iter=1000)`。
- D0 控制组：弱 ISRJ 正样本权重 4，其余为 1。
- D1 处理组：与 D0 相同，另将 fit role 中 ISRJ 负样本里 `delta` 最高四分位样本权重设为 4。
- 四分位切点只在 fit role 计算；不搜索分位点、权重或特征形式。
- calibration role 重新拟合普通六类 Platt；threshold role 在 FPR 0.20 约束下选阈值；early role 只评估。

## 晋级门禁

D1 必须同时满足：

1. 相对原 B2 的 ISRJ 弱正—负样本 AUROC 平均增加至少 0.005；
2. 相对 D0 的弱 AUROC 平均增加至少 0.003；
3. 相对原 B2 的 ISRJ AUPR 非负；
4. 相对原 B2 的弱召回非负；
5. 平均 ISRJ FPR `<=0.20`；
6. 最差受支持 composition/co-label 子群 FPR `<=0.25`；
7. macro-F1 相对 B2 不下降超过 0.005；
8. 至少四个种子的弱 AUROC 相对 B2 为正；
9. 非 ISRJ logits 完全不变；
10. test-v3 未访问、未分析、未使用。

门禁失败则否定“仅靠冻结分数头难负加权”足以修复风险，不得搜索更多权重或分位点；应回到周期专家训练数据，构造机制一致但无 ISRJ 的 matched hard negatives。

