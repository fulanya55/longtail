# Route A H1+B2 多标签连续风险阈值预注册方案

## 研究问题

FMB 二值上下文把 B2 相对 H1 的弱召回增益保留率从 39.36% 提高到 63.29%，但仍未达到 80%，且最差风险会转移到 VFT 共现和四分量组合。本轮检验一个单一、可部署、可证伪的假设：ISRJ 假阳性风险可由冻结模型对五个非 ISRJ 标签的连续校准概率共同估计，而不是由 FMB 单一硬标签表示。

## 冻结项与角色隔离

- 冻结 Focal-36 表征、H1 六类线性头、RawSaw、310 维周期专家、B2 融合权重、六类 Platt 校准器和原始阈值。
- 风险特征固定为 `[P(FMB), P(MFT), P(SawFM), P(SinFM), P(VFT)]`，禁止加入 ISRJ 分数、真实共现标签或样本参数。
- calibration role：仅在真实 ISRJ 负样本上拟合风险校准器；目标为“原 B2 决策是否产生 ISRJ 假阳性”。
- threshold role：只选择风险条件 ISRJ 阈值。
- early role：只报告开发结果和晋级门禁。
- test-v3 保持 `access_authorized=false, analysis_allowed=false, used=false`。

## 固定风险模型与决策规则

- 风险模型：`LogisticRegression(C=0.1, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=0)`。
- 不搜索特征、C、类别权重、solver 或风险切点。
- 高风险定义固定为 `risk_probability >= 0.5`。
- 非高风险样本保持原 B2 ISRJ 阈值；高风险样本使用一个不低于原阈值的新阈值。
- threshold role 选择准则：满足整体 ISRJ FPR `<=0.20` 且高风险负样本 FPR `<=0.22` 的候选中，最大化弱正召回；并列时选择更低的新阈值。

## 五种子开发门禁

必须同时满足：

1. early 最差受支持 composition/co-label 子群 FPR `<=0.25`；
2. 平均 ISRJ FPR `<=0.20`；
3. 保留至少 80% 的 B2-vs-H1 弱召回增益；
4. 相对 B2 的强召回下降不超过 0.03；
5. macro-F1 下降不超过 0.005；
6. ISRJ AUPR 不变；
7. ISRJ 弱正—负样本 AUROC 不变；
8. threshold role 风险 AUROC 五种子均值 `>=0.60`；
9. 至少四个种子的风险条件阈值不低于原 B2 ISRJ 阈值；
10. test-v3 边界完整。

失败后不得在 development-v3 继续搜索风险模型形式。成功后也不得复用 OOD-v1 作为晋级证据，必须生成全新 receiver identity 的 OOD-v2 后冻结验证。

