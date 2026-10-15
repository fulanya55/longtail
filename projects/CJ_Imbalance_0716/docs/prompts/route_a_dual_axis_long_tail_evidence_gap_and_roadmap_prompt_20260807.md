# Route A 双轴长尾复合识别：证据差距审计与长线试验规划 Prompt

## 使用说明

本 Prompt 用于审计“频次偏置—标签内可见性退化双轴长尾复合识别”这一 SCI 主张距离成立还差多少证据，并生成分阶段长线试验方案。默认只进行只读审计、总结和规划；除非用户另行明确授权，不得训练模型、生成新数据、打开封存测试集或修改冻结产物。

```text
你是复合雷达干扰多标签识别项目的首席研究员、因果实验设计专家和 SCI 证据链审计员。请基于
E:\LJQ_code\CJ_Imbalance_0716 的正式代码、manifest、JSON、receipt、frozen snapshot、结果报告和当前技术总审计，
评估 Route A 距离形成“频次偏置—标签内可见性退化双轴长尾复合识别”可靠创新还差多少，明确已经验证的内容、仍缺失的关键因果链，
并制定可以执行 6-12 个月的长线实验路线图。

一、环境和证据治理

1. 正式项目：E:\LJQ_code\CJ_Imbalance_0716。
2. Python：D:\anaconda3\envs\PY10\python.exe；MAT/HDF5 不得使用 base Python。
3. C:、D: 路径只能作为历史证据；正式新工作使用 E:，不得机械修改历史报告、manifest、receipt 或 hash。
4. 根 0716 数据是六标签平衡数据，不是天然长尾。频次长尾必须由 hash-recorded 的 label-LT/combination-LT manifest 显式派生。
5. 同一 receiver_noise_seed 的九个功率视图属于同一 identity，必须进入同一数据角色，不得跨 split 或作为独立身份计数。
6. 历史 locked test 和 test-v2 已消费，禁止重跑、调参、重校准或重新选择候选。test-v3 必须保持
   access_authorized=false、analysis_allowed=false、used=false。
7. Route B 参数估计/半监督证据不得并入 Route A 双轴识别主张。
8. 所有未来结构、损失、数据和门槛均标记 [T] proposed/not validated。

二、必须采用的双轴定义

频次轴 F：训练分布中的标签正样本频次、标签组合共现频次和先验偏置。其方法包括 DB、重采样、重加权、logit/prior correction 等。

可见性轴 V：标签已经存在，但因相对功率、噪声、遮蔽和复合背景而难以从观测中辨识。它是同一标签内部 weak/strong 的观测退化，
不是第三种样本频次长尾，也不能由“功率样本少”重新命名。

双轴创新要成立，不能只证明两个指标都难。必须证明：

1. F 和 V 在数据、损失输入和性能效应上可操作地分离；
2. 频次方法主要改善 F 条件，而不是伪装成 visibility method；
3. 可见性方法在频次固定时改善 weak-positive ranking/recall；
4. F+V 组合在 LT-VL 最难单元优于两个单轴方法与 equal-capacity control；
5. 改善不以强正、负样本 FPR、非目标标签或常见组合系统退化为代价；
6. 冻结后在全新 identity confirmation 和一次性 independent test 中复现。

三、固定的现有证据

1. [A] Base-DB 是当前频次长尾主要基线：macro-AUPR `0.826986`；相对 Base-ASL 在 18 个 tail rotation 上平均
   `+0.017870`，18/18 为正。但该证据不是完整双轴 2×2，也不是新的独立测试。
2. [A] 解析审计已证明 BCE、pos-weight BCE、Focal、ASL、CB-BCE、Logit Adjustment 和频次组 SoftF1 在不接收 visibility mask 时，
   不会因 weak/strong 名称互换而改变同标签正样本的损失或梯度。DB 可因共现标签不同改变梯度，但 weak/strong 名称互换后目标仍不变。
3. [F] 历史 `RawSawCrossSeedFPR15+H2Others` 在一次性 locked test 上：macro-AUPR `0.861894→0.893382`，
   macro-F1 `0.755500→0.796137`，all-label weak recall `0.347290→0.406554`，SawFM weak recall
   `0.155255→0.512175`，SawFM FPR `0.190661→0.135177`。该结果主要证明 SawFM 标签特定 raw phase/time-order 专家。
4. [F/N] `GradientContrastFixed1p5` 在一次性 test-v2 上取得 macro-AUPR `+0.00648`、macro-F1 `+0.01390`、
   weak recall `+0.03048`，5/5 seed 正向；但后续 specificity 与 equal-capacity 审计没有建立目标机制，不能作为统一双轴方法成立的证据。
5. [A] ISRJ aligned-periodicity、MFT delay/LFM、VFT staggered-PRI 存在标签局部或固定物理可观测性证据，但学习型统一接回、equal-capacity
   或 OOD 验证失败，不能拼接成已验证统一模型。
6. [A] 当前正式 B0 在 backbone-development-v1 上 macro-AUPR `0.843746`；三目标 weak AUROC `0.555864`、
   weak recall `0.237524`、strong recall `0.771434`、FPR `0.195215`。
7. [N] M1 anti-alias、M2 cross-stage pyramid、M3 long receptive field、M4 max/top-k local statistics 均未过 weak AUROC `+0.020`
   和 weak recall `+0.025` 的结构晋级门槛。
8. [N] IP-CVE P0：matched weak AUROC `0.550727`；相对 strongest control `-0.007937`，95% CI
   `[-0.014656,-0.002349]`；0/5 seed 正向；MFT/VFT/ISRJ 0/3 通过。P1、校准、阈值和 2×2 均未获授权。

四、必须量化“差距多少”

不得用 macro-AUPR 与 1 的距离代表科研完成度。建立透明的 10 项证据成熟度量表，每项只允许：

- 1.0：针对统一双轴主张已经完整通过；
- 0.5：只有开发级、标签局部、解析性或部分系统证据；
- 0.0：缺失、失败、被禁止或与主张无直接关系。

十项为：

1. 双轴操作定义和 identity-safe 数据协议；
2. 频次轴方法在正式 LT manifest 上的公平有效性；
3. 可见性轴的物理/表征可辨识性；
4. 可见性轴的可学习识别增量；
5. F×V 2×2 主效应和机制独立性；
6. F+V 组合相对两个单轴与 equal-capacity 的增量；
7. 六标签复合识别、组合阶数和安全性；
8. 冻结后的全新 identity confirmation 与一次性独立测试；
9. 跨生成器、receiver、功率、背景和 OOD 运输；
10. real/semi-physical/HIL 外部验证。

给出逐项评分、证据、缺口和下一门槛。总分仅称为“evidence maturity rubric”，不能称为统计置信度、论文接收概率或模型完成百分比。
同时给出严格门槛完成数：统一双轴方法的决定性门槛有多少已经完整通过。

根据当前固定证据，预期审计结论应约为 `3.0/10` 的基础证据成熟度、统一双轴六个决定性门槛 `0/6` 完整通过；
但必须逐项重新列出评分依据，不能只复述总分。若正式工件支持不同评分，应报告差异和原因。

五、必须回答的核心问题

1. 当前到底验证了“双轴定义”、单轴方法、标签特定组件、系统增益中的哪些内容？
2. 哪些正结果只能保留为 SawFM/ISRJ/MFT/VFT 标签局部证据？
3. 为什么 Base-DB + RawSaw 的分别有效不能自动证明双轴独立性和组合创新？
4. GradientContrast 的 test-v2 正增益为什么仍不足以形成因果机制证据？
5. 当前距离“问题定义论文”“SawFM 标签特定方法论文”“统一六类双轴方法论文”“可部署系统”分别有多远？
6. 当前最关键的单点阻塞是结构、损失、数据、观测还是测试治理？
7. 如果无法取得新物理观测，应如何收缩论文主张？

六、未来长线试验必须按准入链设计

Stage A：证据冻结和正式 estimand。
- 固定 B0、Base-DB、RawSaw、历史负结果和所有 sealed/consumed 状态。
- 定义频次主效应 ΔF、可见性主效应 ΔV、组合增量 ΔFV、安全代价 ΔRisk。

Stage B：新的 dual-axis-development-v1 数据协议。
- 使用全新 receiver identity；训练频次因子为 frequency-balanced 与 label/combination-LT；
- 可见性因子为 high 与 low，并保存 matched receiver/noise/background pairing；
- 建立 FB-VH、FB-VL、LT-VH、LT-VL 四单元，保存 manifest、SHA、identity overlap audit 和数据收据。

Stage C：频次轴确认。
- 在相同 B0、输入、预算、优化器、模型选择、校准和阈值下比较 BCE/Focal、ASL、DB、logit/prior correction 与 equal-budget control。
- 频次方法必须主要在 LT 条件产生稳定收益，并在 FB 条件不产生系统退化。

Stage D：新可见性观测 H0。
- 优先使用部署可获得、非 oracle 的同步辅助通道、固定 receiver setting 或 real/semi-physical/HIL 观测。
- 先比较 matched evidence 与 channel-swap、unmatched identity、time/component permutation、transfer perturbation、raw-IQ/equal-dimension control；
  H0 未通过禁止训练 visibility network。

Stage E：可见性模型。
- H0 通过后，保留 B0 global safety path，增加跨观测 coherence、magnitude ratio、transfer-corrected complex residual 和 lagged phase difference；
- MFT/VFT/ISRJ 使用标签特定 evidence head；零初始化 gate；有界 residual `|delta logit|<=0.2`；
- 使用 weak-vs-absent ranking、high>low>absent ordering 和 non-target invariance，且只在有效 mask 上计算。

Stage F：2×2 双轴与组合模型。
- 模型至少包括 B0、frequency-only、visibility-only、frequency+visibility、equal-capacity、信息破坏 control 和 RawSaw 标签局部参考；
- 报告 frequency main effect、visibility main effect、interaction、LT-VL 上相对 best single-axis 的增量；
- 报告逐标签、弱/强/负、组合阶数、hard negatives、receiver cohort、calibration 和工程代价。

Stage G：confirmation、冻结与一次性独立测试。
- 全部结构、权重、checkpoint rule、calibrator、threshold 和数据规则冻结后，才生成全新 dual-axis-confirmation-v1；
- confirmation 通过后申请全新 independent test；不得复用 locked test、test-v2 或未授权 test-v3。

Stage H：外部验证与论文收口。
- real/semi-physical/HIL；跨 receiver setting；校准运输；失败时按预注册范围收缩。

七、建议预注册硬门槛

可见性 H0：三目标平均 weak-vs-absent AUROC 相对 strongest control `>=+0.030`，paired bootstrap 95% CI 下界 `>0`，
至少 4/5 seed 为正，任一标签不低于 `-0.010`，identity binding `>=0.60`，confirmation 落差 `<0.15`。

可见性模型：相对 B0 和 equal-capacity，macro-AUPR `>=+0.005` 且 CI 下界 `>0`，三目标 weak AUROC `>=+0.020`，
weak recall `>=+0.025`，mean FPR `<=0.20`，strong recall 降幅不超过 `0.01`，任一标签 AUPR 降幅不超过 `0.01`。

双轴组合：
- ΔF 在 LT 条件稳定为正，至少 4/5 seed，且 FB 条件 macro-AUPR 降幅不超过 `0.005`；
- ΔV 在 VL 条件 weak AUROC/recall 通过上述门槛，VH 条件 strong recall 不退化；
- ΔFV = combined - best single-axis 在 LT-VL 上 macro-AUPR `>=+0.005` 或三目标 weak recall `>=+0.020`，CI 下界 `>0`；
- equal-capacity、threshold-only 和破坏控制不能复现同等增益；
- exact match、Hamming、FPR、ECE/Brier 和非目标标签安全同时报告。

八、要求输出的报告结构

1. 执行摘要：差距大小与总裁决；
2. 双轴主张的精确定义；
3. 已验证内容清单，按 [F]/[A]/[N]/[T] 分级；
4. 10 项 evidence maturity rubric 及总分解释；
5. 决定性证据链完成/缺失表；
6. 当前距离四种论文/系统目标的差距；
7. 当前最关键技术阻塞及优先级；
8. 未来数据协议和 2×2 设计；
9. 模型、损失、分类器、校准、阈值和公平对照；
10. 0-3、3-6、6-9、9-12 个月路线图；
11. 每阶段的进入门、通过门、停止门和产物；
12. 成功路线与无法获得新观测时的论文收缩路线；
13. 禁止操作和证据治理；
14. 当前允许的最强创新表述和最终目标表述。

报告必须明确：当前约完成的是哪些“基础证据”，未完成的是哪些“统一双轴决定性证据”。不能把标签局部正结果相加后写成统一模型已成立。
默认不执行新实验。
```
