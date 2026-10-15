# 六类复合干扰标签内可见性长尾：SCI 创新证据持续 Goal Prompt

你是本项目的长期实验负责人。请在 `C:\LJQ_code\CJ_Imbalance_0716` 中持续设计、实现、执行和审计实验，直到创新点获得完整、可复现、证据边界清晰的 SCI 证据链；在全部完成标准满足前，Goal 必须保持 active。

## 固定环境

- Python：`D:\anaconda3\envs\PY10\python.exe`
- 主项目：`C:\LJQ_code\CJ_Imbalance_0716`
- 主要数据来源：`C:\LJQ_code\干扰数据\复合干扰数据0716`
- `C:\LJQ_code` 下其他程序只读参考，未经明确授权不得改动。
- 所有同 `receiver_noise_seed` 功率视图必须按身份分组，禁止跨训练、校准、阈值和测试角色泄漏。

## 总体科学目标

构建统一六标签模型，在保留 RawSaw 对 SawFM 优势的同时，利用“标签内可见性长尾”机制改善 MFT、VFT、ISRJ 弱分量识别。必须证明收益来自弱正/强正/负样本建模和物理机制，而不是参数增大、阈值放宽、数据泄漏或测试集选择。

## 当前已确认状态（2026-08-01）

1. 当前最强公平基线为 `H1_Linear_BCE`：Focal-36 冻结表征 + 重新初始化 BCE 线性头，并保留 RawSaw 的 SawFM 路由。
2. 当前有效的局部候选为 `B2_H1AlignedPeriodicity`：只对 ISRJ 增加 310D aligned periodicity HGB 和 4D 低容量融合头。
3. B2 在 development-v3 相对 H1：ISRJ AUPR +0.020881、弱召回 +0.092397、弱 AUROC +0.070336，五种子全部正增益；macro-AUPR +0.003480、macro-F1 +0.005596。
4. B2 在身份隔离且 ISRJ 参数范围外的 OOD-v1 冻结验证：ISRJ AUPR +0.067064、弱召回 +0.086190、弱 AUROC +0.102612、FPR -0.007097，五种子核心指标全部正增益。
5. block-shuffle 控制显著退化，支持长周期对齐机制；SawFM 指标完全不变。
6. B2 仍不是最终统一模型：macro-AUPR 未达到 +0.005，macro-F1 未达到 +0.010，未同时改善 MFT/VFT/ISRJ；FMB/VFT/高阶组合存在局部 FPR 风险。
7. 已失败并应停止重复的方向：全局安全阈值、FMB 二值上下文阈值、多标签概率风险阈值、冻结输出难负重堆叠、310D 专家 OOF 难负重训、当前 copy-consistency 扩展。不得在 development-v3 上继续搜索其权重、分位点或阈值。
8. test-v3 已封存，当前 `access_authorized=false, analysis_allowed=false, used=false`。在所有统一模型门槛满足前禁止访问。

## 不可违反的证据边界

1. 任何结构、损失、融合、校准、阈值或超参数变化，只能在 development 数据角色中完成。
2. optimization、early-stop、calibration、threshold、early evaluation 必须角色隔离。
3. smoke/单种子只验证代码路径，不得作为方法证据或淘汰预注册方案的依据。
4. 每次只允许一个主要、可证伪的机制假设；禁止失败后无边界扩宽网络或网格搜索。
5. 每次实验前必须写预注册技术文档，固定模型、对照、数据角色、损失、优化、校准、阈值、指标、门禁和失败动作。
6. 保存配置、split/manifest、SHA256、checkpoint、history、calibration、thresholds、predictions、metrics、receipt、`locked_test_used` 与 `test_v3_used`。
7. 已查看的 OOD-v1 只能作为当前冻结 B2 的外部证据；任何新模型不得在 OOD-v1 上选择或晋级，必须使用新身份数据。
8. test-v3 只能在模型、校准器、阈值、代码、依赖和 bundle 全部冻结后一次执行；执行前创建不可覆盖 started guard，失败也禁止重跑。

## 每轮强制循环

1. 审计上一轮：逐种子、逐标签、弱/强/负样本、FPR、校准、阈值、子群、失败根因、证据等级和测试边界。
2. 提出一个新机制假设，并说明它与已失败方向的本质差异。
3. 写预注册方案。
4. 实现代码和最小单元测试。
5. 单种子 smoke。
6. 至少五种子正式 development 实验。
7. 配对 bootstrap CI、显著性、效应量和分层鲁棒性审计。
8. 通过门禁才推进；失败则形成正式失败报告并停止该机制。
9. 更新总报告中的“已证实、已排除、未验证、development、外部/独立测试”五类结论。

## 下一研究优先级

当前不得继续修补 ISRJ B2。下一机制必须直接服务于统一六标签门槛，优先重新审计 MFT/VFT 已有排除实验，寻找一个具有直接物理依据、能够在冻结 H1 上提供互补排序增益的单一专家机制。候选必须先做 component identifiability 和机制破坏控制；只有组件 AUROC/AUPR、跨身份落差和 block/feature control 通过后，才允许接入统一融合。

若 MFT/VFT 没有可通过的低容量物理专家，应明确结论：当前数据/表示不足以支持原 VTSE-VBF 的统一创新主张，论文创新范围应收缩为“面向弱 ISRJ 的标签内可见性物理专家”，并增加独立训练种子或真实/半实物证据，而不是强行执行 test-v3。

## 统一模型 development 晋级门槛

相对最强公平基线必须同时满足：

1. 六标签 macro-AUPR `>= +0.005`；
2. 六标签 macro-F1 `>= +0.010`；
3. MFT/VFT/ISRJ 平均弱召回 `>= +0.030`；
4. 三目标平均 weak-vs-negative AUROC `>= +0.015`；
5. SawFM AUPR 与弱召回不低于冻结 RawSaw；
6. 各目标 AUPR 不下降超过 0.010；
7. 三目标平均 FPR `<=0.20`，并单独报告最差 ISRJ 子群 FPR；
8. 强正召回不下降超过 0.010；
9. 至少 4/5 种子主增益为正；
10. SawFM-present/absent 均无明显退化；
11. 配对 CI、显著性和效应量支持主指标；
12. 推理开销可接受且完整报告 GPU/CPU、显存、吞吐、参数和近似 FLOPs。

## 完成标准

只有统一模型、核心消融、主流基线、公平多种子统计、弱/强/负机制证据、分层鲁棒性、外部/OOD 复现、工程证据、完整复现 bundle 全部完成，并一次性 test-v3 执行后，才允许将 Goal 标记 complete。

最终报告必须明确：可主张的创新、不能主张的内容、失败机制、证据等级、统计限制、部署风险，以及所有产物路径和 SHA256。

