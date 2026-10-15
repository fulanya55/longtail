# Route A 物理可见度差冻结探针 development-v3 预注册

## 1. 研究问题

当前 final-temporal 512 维统计是否已经包含足以恢复目标分量相对功率差的连续信息？如果
冻结特征上的简单回归探针都无法恢复该变量，则继续在分类 BCE 上重加权缺少机制基础；
应转向更早阶段的定向 adapter 或同 receiver identity 的多功率配对表征学习。

## 2. 冻结协议

- 使用与 C0-C4 完全相同的 development-v3 五组角色和 F2+C6 checkpoint。
- 冻结整个 F2；仅提取 final-temporal 512 维统计并按 fit 均值/方差标准化。
- 对 MFT、VFT、ISRJ 分别拟合固定 `Ridge(alpha=10)`，只使用该标签为正且功率有限的样本。
- 回归目标为 `gap = P_label - max(P_other positive)`，截断到 `[-40,0] dB`；单标签正样本
  定义为 0 dB。alpha 不搜索。
- validation 只用于诊断；early 作为 development-v3 独立报告角色，不用于调参。
- test-v3 禁止访问。

## 3. 指标与可证伪门槛

每标签报告 Pearson、Spearman、MAE、R2，以及用预测 gap 区分 `gap<=-10 dB` 弱正与强正
的 AUROC。机制可行门槛预注册为：

- early 三标签平均 Spearman `>=0.30`；
- early 三标签平均 weak-vs-strong AUROC `>=0.65`；
- 至少 2/3 标签同时满足 Spearman `>=0.25` 和 AUROC `>=0.62`；
- 五个随机种子中至少 4 个通过前两项平均门槛。

若通过，下一轮实现共享低秩 adapter + presence/visibility 双头，并设置同容量无辅助损失对照。
若失败，停止 frozen-final-temporal VTSE 路线，生成 development-v3 identity-safe 多功率视图，
从 stage2 训练小型 adapter 的配对一致性与连续功率回归。

