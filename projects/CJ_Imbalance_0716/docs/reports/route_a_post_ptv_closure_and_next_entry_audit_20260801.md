# Route A：PTV 终止后的证据闭环与下一研究入口审计

日期：2026-08-01  
审计性质：只读汇总；未训练模型、未生成数据、未读取任何 test 文件。

## 结论先行

**当前不应再继续 PTV（paired task-vector）稳定化分支，也不应把 H1+B2 统一候选送入 test-v3。**

可以保留的最强结论是：`H1 + B2 + MFT shared-decision + VFT Ridge-STFT` 在既有身份隔离的开发协议中显示出标签特异机制增益；但该候选尚未形成项目全局新鲜确认，并且在完整系统的 ISRJ 参数 OOD 上未通过预注册门。因此，它只能作为受限开发证据，而不是“已证明的可靠创新点”。

PTV 方向则已经给出两个互补的停止事实：先前 FC-PTV 已被永久停止；随后 TRB-PTV 作为第二个同类机制循环，在冻结求解器的 smoke 中即无法满足数值可行性契约。二者共同意味着：**不得再通过调整 task-vector 系数、半径、约束、激活门、初值、fallback 或重用 v14 身份来继续该方向。**

## 1. 已闭环的 PTV 停止事实 [F]

|分支|已观察到的事实|裁决|不可做的事|
|---|---|---|---|
|FC-PTV|历史异质性审计只允许其作为 TRB 预注册的前置资格；审计明确保留 FC 的永久停止决定|已停止|不得恢复 FC 的半径、系数、阈值或重新解释其为条件化机制成功|
|TRB-PTV|冻结的 fit-only SLSQP 从 `q=(1.5,1.5)` 出发返回 status 8；MFT 风险相对 C6 增加 `+0.01018602`，大于冻结容差 `1e-7`|`stop_trb_ptv_before_formal_evaluation`|不得改初值、加 fallback、放宽风险、改边界/激活或在 v14 上重跑|

TRB 仅运行到单 seed、单 epoch、截断角色的 smoke；没有启动正式五种子评估，也没有访问 locked test、test-v2、test-v3 或 test-v14。故它是**实现/可行性失败记录**，不是性能反例，更不是可用于比较的开发结果。

## 2. 非 PTV 主候选：哪些已成立，哪些仍未成立 [F]

### 2.1 可保留的开发集机制证据

在冻结候选 `H1_B2_MFTSharedDecision_VFTRidgeFusion_v2` 的 line-held-out development-v12 上，相对 H1：

|端点|五种子均值变化|证据解释|
|---|---:|---|
|六标签 macro-AUPR|`+0.00660`|整体排序改善|
|六标签 macro-F1|`+0.01099`|固定决策表现改善|
|MFT/VFT/ISRJ 平均 weak recall|`+0.03759`|主弱分量端点改善|
|MFT/VFT/ISRJ 平均 weak-vs-negative AUROC|`+0.03886`|弱正负分离改善|
|VFT aligned expert 相对 capacity / shuffled control 的 weak AUROC|`+0.01706` / `+0.01757`|VFT 路由不是单纯增容或随机特征效应|
|MFT 高阶难负 FPR|`-0.02658`|共享决策和非正残差具有受控的负迁移安全性|

RawSaw 的 SawFM 输出在该候选中逐样本保持不变。这些结果支持“标签特异的物理/时序证据加共享安全决策”这一**开发协议内**主张。

### 2.2 不能外推的边界

development-v12 的数据身份虽然对该候选为冻结后的 line-held-out 确认，但此前已被 FC-PTV 研究线消耗；所以它**不是项目全局新鲜确认**。任何改变后的候选都不能再拿 v12 选型或确认。

完整候选的 ISRJ 参数 OOD 没有通过：统一目标 FPR 为 `0.20914`（上限 `0.20`），VFT FPR 为 `0.32653`，且 VFT 弱召回相对 Q2 的增益仅 `+0.000649`，低于 `+0.010` 门。即便其中 ISRJ 专家本身存在局部正向迹象，也不能用局部端点覆盖完整系统失败。

因此当前禁止的表述包括：已实现完整跨参数泛化、外部鲁棒性、最终独立测试验证，或已证明可泛化的 PTV 创新。

## 3. 已被否定的修正路线 [F]

|假设|结果|停止决定|
|---|---|---|
|用 ISRJ logit 的低维交叉头抑制 VFT 假阳|construction-v3、v12 诊断及 ISRJ-OOD 上 VFT AUPR 均为负；OOD 的 ISRJ-present/VFT-negative FPR 达 `0.45875`|不再调权重、阈值或正则，也不生成 development-v13|
|扩大/加深普通浅层 STFT-CNN|VFT/ISRJ 来源可辨识门未通过|不再以增容补救|
|ISRJ copy-consistency|相对原 310D 专家 early AUROC `-0.00367`|不接回统一候选|
|以低维相位复制、脊线、复位摘要替代 Ridge-STFT|source-v1 相对 Ridge-STFT early AUROC `-0.06972`，5/5 为负|停止低维摘要替代路线；只可保留“相位组织有信息但压缩不足”的诊断结论|

## 4. 唯一合理的下一入口 [D]

在当前 Goal 的证据边界内，下一步不是再提出一个相近模型，而是完成**论文级证据账本**。它不会改变模型、数据或测试状态，且能把当前成果收束为可信、可审计的研究结论。

1. 建立基线与消融账本：把 H1、B2、VFT、MFT、ISRJ、RawSaw 路由及所有已停止分支映射为“问题—唯一变化—五种子效果—失败/保留原因—数据角色”。
2. 固化主张分层：`[F]` 可复核事实、`[I]` 有限机制解释、`[D]` 研究决策；明确 development-v12、ISRJ-OOD 与密封 test-v3 的不同地位。
3. 补做文献与可复现实装审计，而非临时比较：确认最新多标签长尾基线是否存在可审计、可等预算复现的官方实现；没有时只做定性文献对照，不能虚构实验比较。
4. 准备一份“下一研究问题”的全新预注册草案，但不执行：该问题必须与 PTV 稳定化实质不同，先给出可证伪机制、独立身份数据计划、早期诊断门和停止规则；获得单独授权后才可生成新的 development 数据。

## 5. 后续新 Goal 的准入门（不是本次执行授权） [D]

只有同时满足以下条件，才可启动任何新的模型研究：

- 新假设不属于 task-vector 配对/稳定化、交叉 logit、低维摘要替代或浅层 STFT 增容的改写；
- 先完成针对该假设的文献冲突审计，明确与已有方法的差异和可证伪预测；
- 模型、损失、选择指标、早停、校准、阈值、五种子门、OOD 门和 stop rule 均在生成新数据前冻结；
- 使用全新且按 `receiver_noise_seed` 隔离的 development 身份；不得重用 v12、v14 或任何 test 身份；
- 只有新鲜 development 的所有升级门通过，才可以请求一次密封 test-v3；test-v3 仍不得用于调参或反复确认。

## 证据制品

- 历史异质性与 FC 停止边界：`docs/reports/route_a_fullalpha_effect_heterogeneity_pretrb_audit_20260801.md`
- TRB 冻结 smoke 失败：`docs/results/route_a_development_v14_trb_ptv_smoke_failure_20260801.md`
- TRB failure receipt：`outputs/tapc/route_a_development_v14_trb_ptv_smoke_v2/run_receipt.json`
- H1+B2 v11/v12 证据：`docs/results/route_a_h1_b2_mft_vft_v11_v12_shared_policy_results_20260801.md`
- 创新证据缺口与 OOD：`docs/reports/route_a_final_innovation_evidence_gap_audit_20260801.md`
- VFT-ISRJ 交叉头否定结果：`docs/results/route_a_vft_isrj_cross_expert_head_results_20260801.md`
- 原创新 Goal 终止决定：`docs/results/route_a_innovation_goal_final_decision_20260801.md`

本审计不构成对任何冻结测试的授权；所有结论均为既有制品的只读汇总。
