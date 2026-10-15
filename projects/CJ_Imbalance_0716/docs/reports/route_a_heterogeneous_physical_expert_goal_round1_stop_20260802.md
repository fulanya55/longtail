# 异质物理局部证据专家 Goal：第 1 轮停止与转向报告

日期：2026-08-02  
状态：**当前候选池耗尽；没有获准进入 H0/H1/H2 的新统一模型。**

## 总结性决定

本 Goal 的 Stage 0 已完成审计、文献重叠比对、预注册、实现冻结、train-only 身份生成与重新审计。但在任何 RAAT 网络训练前，发现唯一 RAAT 候选与已存在 H0i/H1j VFT 去调频机制重叠；因此停止 RAAT。随后对 H1j 的固定分数重审显示 weak AUROC 与 AUPR 的 identity CI 都跨 0，按本 Goal 的 H1 联合门不予晋级。H0j 的 MFT 有效支持度虽通过配对物理 H0，却在 H1k 新复合源上以弱 AUROC `-0.168547`、AUPR `-0.075655` 显著反向，因此也永久停止。

没有 H0 新成功、没有 H1 安全专家、没有 H2 六标签集成、没有 H3 鲁棒性、没有 test-v3 访问。不得把“完成 Stage 0、生成训练源、CPU smoke 通过”误报为模型有效性。

## 三个候选的最终处置

| Stage 0 候选 | 状态 | 原因 | 是否允许后续调参 |
|---|---|---|---|
| C1：VFT 盲 dechirp 相位斜率 | 关闭 | H0i/H1j 已覆盖该物理机理；H1j 固定分数重审的 weak AUROC/AUPR CI 均跨 0 | 否 |
| C2：ISRJ 盲尺度/重复间隔投票 | 关闭 | 与已停止的 predicted-width / fixed scale-Hough / 周期性族直接重叠 | 否 |
| C3：RAAT 三图 | 训练前撤销 | VFT 分支是 H0i/H1j 的地图化近邻；MFT 是既有 reference delay-profile 的扩展；不能把表示扩容称为新物理机制 | 否；已生成源隔离、未使用 |
| H0j/H1k：MFT 有效延迟支持度 | 关闭 | H0j 的单 MFT 副本计数可辨识性不能迁移到弱复合 MFT；H1k 两个弱端点均显著反向 | 否 |

## 可保留的证据，而非可夸大的创新

- RawSaw 保持其冻结 SawFM label-route 证据，原样旁路，未被本轮触碰。
- ISRJ aligned-periodicity 有 H0g/H0h 的窄机制复现，但它不自动构成新统一模型或跨标签创新。
- VFT 正确参考去调频有物理 H0i 线索，并在 H1j 有 weak-AUROC 方向性信号；但 weak AUROC 与 AUPR 的 identity CI 都跨 0，且缺少独立排名/新源复制，不能接回 H2。

## 下一条研究问题（需要新的 Goal 后才可执行）

不要再寻找“把已有相位、延迟或周期特征拼成更大的专家”。新的问题必须是：

> 能否找到一个不使用真值参数、且与 VFT dechirp、MFT delay-profile、ISRJ periodicity/scale 完全不同的、可部署的局部物理可观察量，并在**单标签 H0**中先超过一个信息破坏控制？

候选形成前先执行文献/历史代码 audit；若不能给出与这三类既有机制的数学差异、正反物理预测和信息破坏控制，就不生成数据。只有新的单标签 H0 成功后，才重新讨论“至少两专家”的长期集成目标。

## 关键工件

- RAAT Stage 0 与安全修订：`docs/reports/route_a_raat_stage0_literature_overlap_audit_20260802.md`、`docs/plans/route_a_raat_h0_v1_1_safety_amendment_20260802.md`、`docs/plans/route_a_raat_h0_v1_2_matching_amendment_20260802.md`。
- RAAT 未使用源隔离收据：`outputs/tapc/route_a_raat_h0_generation/quarantine_receipt.json`。
- H1j fixed-score 重审：`docs/results/route_a_h1j_fixed_score_bootstrap_results_20260802.md`。
