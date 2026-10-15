# Route A Raw SawFM 跨种子阈值稳健性计划（2026-07-26）

## 1. 失败结论

上一阶段的逐种子 `FPR<=0.15` 阈值主方案没有通过预注册门槛：

- seed 2039 在独立 `ordinary_early` 上的 FPR 仍为 0.2374；
- 五种子平均 ordinary recall 相对当前阈值下降 0.0226，超过允许的 0.02；
- 但排序能力、功率开发集召回和六类路由指标仍然稳定，因此当前失败不是新的特征学习失败，而是阈值从 selection role 向 early role 迁移时缺少跨种子约束。

探索结果显示共享阈值 0.48 能把 seed 2039 ordinary FPR 降至 0.1871，同时基本保持路由 macro-F1。由于该结果已经看过 early role，不能把它直接称为独立确认。

## 2. 改进设计

采用留一种子外推（leave-one-seed-out, LOSO）阈值：

1. 对目标 seed 的阈值选择完全排除该 seed；
2. 仅使用其余四个 seed 的 `ordinary_threshold + power_threshold` 搜索表；
3. 要求候选阈值在四个来源 seed 上均满足 ordinary/power FPR 不高于 0.15；
4. 在目标 seed 的 `ordinary_early + power_development` 上评估；
5. 不重训网络，不重拟合 Platt，不访问 locked test。

对比：

- `C0_Current`：当前逐 seed、FPR 0.20 阈值；
- `C1_LOSO_FPR15_Shared`：主要方案，四 seed 选择、目标 seed 外推；
- `C2_LOSO_FPR10_Shared`：保守诊断方案，不允许在本轮直接晋升。

## 3. 主要方案门槛

`C1_LOSO_FPR15_Shared` 必须同时满足：

1. 五个 held-out seed 的 ordinary FPR 均不高于 0.20；
2. 五种子平均 ordinary FPR 不高于 0.16；
3. 五种子平均 ordinary recall 不低于 0.84；
4. 50 dB 平均召回不低于 0.95；
5. 每个 seed、每个功率级 FPR 均不高于 0.20；
6. 六类路由 macro-F1 相对 `C0` 下降不超过 0.005；
7. exact match 相对 `C0` 下降不超过 0.01；
8. 路由门槛至少 4/5 seed 通过；
9. 五个 LOSO 阈值的极差不超过 0.05；
10. identity overlap=0，locked test 未使用。

## 4. 决策规则

- 全部门槛通过：把 LOSO 得到的稳定共享区间固化为新的候选阈值规则，并重新生成 locked-test preflight；这里只能称为开发集跨种子稳健性通过。
- FPR 通过但 recall/路由失败：保留当前冻结候选，不再提高阈值，下一步统一 Raw frontend 训练协议。
- FPR 仍失败：判定阈值修补不足，下一步必须增加新 seed/新 split 的独立开发确认或改进校准方法。

