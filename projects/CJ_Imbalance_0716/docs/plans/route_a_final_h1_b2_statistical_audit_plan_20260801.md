# Route A 最终 H1+B2 统计审计方案

对 development-v3 与 identity-isolated ISRJ parameter OOD-v1 的五种子 B2-H1 配对差值进行统一审计。

- 主指标：ISRJ AUPR、弱召回、弱正—负样本 AUROC。
- 辅指标：macro-AUPR、macro-F1、强召回、FPR。
- 报告均值、样本标准差、paired Cohen's dz、正增益种子数、exact one-sided Wilcoxon；FPR 使用“B2 更低”的方向。
- 三项主指标在每个数据集内做 Holm 校正。
- n=5 时显著性功效有限，必须同时解释效应量、bootstrap CI 和方向一致性，不以单一 p 值宣称结论。
- 不访问 test-v3。

