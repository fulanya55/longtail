# Route A：冻结 ISRJ 周期专家的 MFT/VFT 共标签分层审计

## 目的与证据边界

现有 H0g/H0h 已在两类新外部条件下冻结验证 `B2_H1AlignedPeriodicity_Frozen` 对 ISRJ 的整体增益，但尚未明确说明其是否仅由某个 MFT/VFT 共现上下文驱动。本轮只对已写入的冻结预测做**非选择性描述性审计**：不产生新模型、不重训、不重校准、不重设阈值，且不读取任何 sealed test。

它可加强或限定 ISRJ 的条件适用范围；它不能成为新的选择证据，也不能用于调整 B2、融合、阈值或决定 test-v3。

## 锁定输入与分层

- 输入：H0g 与 H0h 的五个训练种子（2039--2043）下 `B0_H1_Frozen`、`B2_H1AlignedPeriodicity_Frozen` 的既有 `predictions.npz`。
- 固定目标：ISRJ（索引 5）。每个分层内，正类是 ISRJ 弱正，负类是 ISRJ 缺失；强正不参加弱排序 AUROC。
- 固定共标签分层：全体、MFT present/absent、VFT present/absent、以及 `(MFT,VFT)` 的四种二值组合。每个分层要求至少 10 个弱正和 10 个负样本，其他分层报告为 unsupported。
- 端点：ISRJ weak-vs-negative AUROC、固定预测的 weak recall、ISRJ negative FPR，以及 B2−B0 的五种子均值和种子 bootstrap 95% CI。
- 审计：每个读取预测文件记录 SHA-256；同一种子两个臂的 targets/weak mask 必须完全一致；所有输入 summary 都必须表明 completed、frozen external、test-v3 unused。

## 解释规则

这是后验、非选择性分层报告，不设 promotion gate。只有当所有受支持分层的方向一致时，才可谨慎说“已观察到的 ISRJ 增益不局限于所报告的 MFT/VFT 共现上下文”；若有反向或不支持分层，只如实界定机制的已观测范围。
