# Route A FullAlpha1p5 + ISRJ Periodicity 统一组合预注册方案

## 单一机制假设

`FullAlpha1p5` 的配对 SawFM label-deletion task vector 已稳定改善 MFT/VFT 与六标签宏观指标；aligned periodicity expert 已独立改善 ISRJ。两者作用于不同层级：前者改变共享块和分类器，后者只替换 ISRJ 输出分数。若收益近似标签选择性正交，则组合后应同时满足统一模型的宏观、三目标弱召回与 weak-AUROC 门槛。

## development-v3 组合与对照

- U0：`C6_Joint + RawSaw`。
- U1：`FullAlpha1p5 + RawSaw`。
- U2：`FullAlpha1p5 + RawSaw + aligned periodicity ISRJ fusion`。
- U3：与 U2 完全相同，但周期信号按固定 25 点 block permutation 破坏长周期对齐。
- FullAlpha1p5、RawSaw、310D HGB 全部使用既有冻结 checkpoint。
- 只在 development-v3 fit role 训练一个 ISRJ LogisticRegression 融合头：`C=0.1`、弱 ISRJ 正样本权重 4；不搜索特征、C 或权重。
- calibration role 拟合逐标签 Platt；threshold role 在逐标签负样本 FPR 0.20 约束下最大化弱召回；early role 只做组合门禁。
- 非 ISRJ logits 必须与 U1 逐元素完全一致，SawFM 必须来自同一个 RawSaw 路由。

## development-v3 组合门禁

U2 相对 U0 必须同时满足：

1. macro-AUPR `>= +0.005`；
2. macro-F1 `>= +0.010`；
3. MFT/VFT/ISRJ mean weak recall `>= +0.030`；
4. 三目标 mean weak-vs-negative AUROC `>= +0.015`；
5. 三目标 mean FPR `<=0.20`；
6. 三目标每标签 AUPR 均不下降超过 0.010；
7. 三目标 mean strong recall 不下降超过 0.010；
8. 至少 4/5 种子 target mean weak recall 为正；
9. U2 相对 U1 的 ISRJ weak AUROC `>= +0.005`、弱召回 `>= +0.010`；
10. U2 相对 U3 的 ISRJ weak AUROC `>= +0.005`；
11. SawFM AUPR/弱召回与 U0、U1 完全一致；
12. 非 ISRJ logits 不变，test-v3 边界完整。

## 后续动作

- 通过：冻结 U0/U1/U2/U3 的模型、融合头、校准器和阈值；生成全新 identity-isolated development-v7，只作一次冻结确认。v7 不允许重新拟合或选择。
- 失败：否定两机制可直接正交组合；不得扫描融合权重。保留 FullAlpha1p5 与 B2 为两个独立局部证据，统一创新范围降级。

