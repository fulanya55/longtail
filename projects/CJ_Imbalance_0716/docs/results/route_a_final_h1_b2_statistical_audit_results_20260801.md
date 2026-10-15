# Route A 最终 H1+B2 五种子统计审计

## 结论

H1+B2 在 development-v3 和身份隔离 parameter-OOD-v1 上，三项 ISRJ 主指标均为 5/5 种子正增益，单指标 exact one-sided Wilcoxon `p=0.03125`，且 paired Cohen's dz 为大效应；但每个数据集内对三项共同主指标做 Holm 校正后均为 `p_adj=0.09375`。因此可以支持“方向一致、效应量大、跨数据身份复现”，但不能声称在三重比较的 0.05 family-wise 水平全部显著。

## 主指标

| 数据 | 指标 | 平均 B2-H1 | paired dz | 正增益种子 | exact p | Holm p |
|---|---|---:|---:|---:|---:|---:|
| development-v3 | ISRJ AUPR | +0.020881 | 2.107 | 5/5 | 0.03125 | 0.09375 |
| development-v3 | 弱召回 | +0.092397 | 1.792 | 5/5 | 0.03125 | 0.09375 |
| development-v3 | 弱 AUROC | +0.070336 | 3.404 | 5/5 | 0.03125 | 0.09375 |
| OOD-v1 | ISRJ AUPR | +0.067064 | 4.407 | 5/5 | 0.03125 | 0.09375 |
| OOD-v1 | 弱召回 | +0.086190 | 2.727 | 5/5 | 0.03125 | 0.09375 |
| OOD-v1 | 弱 AUROC | +0.102612 | 6.505 | 5/5 | 0.03125 | 0.09375 |

OOD-v1 的效应量大于 development-v3，说明增益没有依赖开发集的 ISRJ 参数范围；其 width、replica 数均超出训练范围，且模型、校准器、阈值完全冻结。

## 辅助指标与安全性

- development-v3：macro-AUPR +0.003480、macro-F1 +0.005596、强召回 +0.045477，均为 5/5 正增益；FPR 平均 +0.004355，只有 2/5 种子更低，`p=0.78125`，不能主张开发集 FPR 改善。
- OOD-v1：macro-AUPR +0.011177、macro-F1 +0.016105、强召回 +0.131802，均为 5/5 正增益；FPR 平均 -0.007097，4/5 更低，单侧 `p=0.09375`。
- 分层审计仍发现 FMB/VFT/高阶组合的局部 FPR 可超过 0.25；这必须作为部署局限报告，不能被总体 FPR 抵消。

## 对论文证据链的含义

当前证据足以支持一个谨慎的创新表述：低容量 aligned periodicity expert 在冻结强基线 H1 上，对弱 ISRJ 识别产生跨训练种子、跨 receiver identity、跨 ISRJ 参数范围的一致大效应增益，而且 block-shuffle 控制证明收益依赖长周期对齐结构。

若要做更强的“多重校正后统计显著”声明，需要预注册并增加独立训练种子，或在最终 test-v3 上以 receiver identity 为聚类单位给出一次性置信区间；不得把现有五种子重复运行当作新增独立证据。

## 产物

- 方案：`docs/plans/route_a_final_h1_b2_statistical_audit_plan_20260801.md`。
- 实现：`experiments/audit_route_a_final_h1_b2_statistics.py`。
- 测试：`tests/test_audit_route_a_final_h1_b2_statistics.py`，1 passed。
- 输出：`outputs/tapc/route_a_final_h1_b2_statistical_audit/route_a_final_h1_b2_statistical_audit.json`。
- 汇总 SHA256：`77FC19C699D77CA8EAAFDFB6DDAFF3B67C08B72860B2CE48719C50D72CF6F8B0`。
- test-v3 未使用。
