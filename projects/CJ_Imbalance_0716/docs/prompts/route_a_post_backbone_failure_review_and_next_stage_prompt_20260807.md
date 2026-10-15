# Route A 上一阶段失败复盘、创新证据重审与下一阶段设计 Prompt

## 可直接交给 Codex 的专业 Prompt

```text
你是复合雷达干扰多标签识别项目的首席研究审计与实验设计专家。请基于
E:\LJQ_code\CJ_Imbalance_0716 当前正式工件，完成 Route A 在 backbone 强化科学停止后的
失败复盘、既有创新点证据重审和下一阶段可证伪方案设计。

任务不是继续堆叠网络，也不是为失败模块寻找新的调参版本。必须从实验事实出发，区分：
1. 已确认事实；
2. 有限或标签特定证据；
3. 已证伪机制；
4. 尚未验证的推断；
5. 下一阶段提案。

一、固定项目与证据边界

- 项目：E:\LJQ_code\CJ_Imbalance_0716
- Python：D:\anaconda3\envs\PY10\python.exe
- 数据根：E:\LJQ_code\干扰数据
- 当前正式 backbone 基线：B0/F2_DFMS_Full，257,511 参数
- test-v3 必须继续保持 access_authorized=false、analysis_allowed=false、used=false；不得读取其
  I/Q、标签、预测或指标。
- backbone-development-v1 已被 M1–M4 消费并进入科学停止；不得在该来源上重新选结构、损失、
  阈值、残差权重或组合失败模块。
- development-v3、H3、H5o、H0v、P1、P2 等历史来源只能作为边界证据，不得重新用于选择。
- 根 0716 数据是平衡数据。任何标签/组合长尾主张都必须使用显式派生、hash 记录的 LT manifest。
- 同一 receiver_noise_seed 的多功率视图必须属于同一数据角色。

二、必须审计的当前事实

1. B0 development-v1：macro-AUPR 0.843746，三目标弱正 AUROC 0.555864，弱召回
   0.237524，强召回 0.771434，FPR 0.195215。
2. M1 抗混叠：Δmacro-AUPR +0.004655，但 Δ弱 AUROC -0.000354，SinFM/VFT 退化。
3. M2 跨层 CSFP：Δmacro-AUPR 约 0，Δ弱 AUROC +0.000447；有效残差被压到接近 0。
4. M3 长感受野 EPB：Δ弱 AUROC +0.006936、Δ弱召回 +0.008218，但 macro-AUPR
   不升，FMB/SinFM 系统性退化；仅可视为 VFT/ISRJ 标签特定信号。
5. M4 max/top-k：Δ弱 AUROC 约 0，Δ弱召回 -0.003301，5/5 seed 下降；4/5 seed
   学到负局部残差。
6. 既有 VTSE-VBF 的 C3 未战胜同参数 C4，故标签内可见性机制未获得容量排除证据。
7. H3 证明冻结标签特定专家在同生成器新 identity 标准源上可复现，但不能替代 VTSE-VBF 的
   因果控制；H5o 联合 OOD 失败说明统一候选仍存在阈值运输和跨标签安全问题。
8. RawSaw 对 SawFM、ISRJ aligned-periodicity、MFT shared-decision、VFT aligned Ridge 具有
   不同程度的标签特定证据；这些只能作为冻结参考或基线，不得拼接成“已证明的统一创新”。

三、既有创新点必须重新拆分

请分别审视以下创新主张：

A. 双轴长尾：频次偏置（标签频次/组合共现）与标签内可见性退化是两种独立机制。
B. VTSE：针对同一标签内部 weak/strong 可见性差异的证据建模。
C. VBF：在 FPR 和强正安全约束下进行有界、标签特定证据接回。
D. RawSaw/物理专家：标签特定证据源，而非统一 backbone 创新。

对每项给出：当前证据、反证、仍缺的因果控制、可保留的最强论文表述、禁止表述。

四、失败原因分析要求

至少从以下层次分析，并建立“实验现象 -> 直接证据 -> 根因推断 -> 排除的替代解释”链条：

1. 观测可辨识性；
2. 监督目标是否提供弱分量归因；
3. shared backbone/共享统计是否造成标签冲突；
4. 局部峰值、长感受野、跨层特征是否缺少成分特异性；
5. 分类器、校准和阈值是否只是放大器而非根因；
6. 训练是否充分；
7. 合成数据与真实/HIL 分布差异；
8. 频次长尾与可见性退化是否曾被混淆。

不得将“模型可能不够大”“训练轮数可能不够”“换注意力或阈值可能有效”作为无证据整改理由。

五、下一阶段唯一允许的研究方向

下一阶段不得继续 backbone-development-v1 上的结构增强。优先设计新的、与历史 identity 零重合的
“同 receiver、同背景、成分反事实配对”观测协议：

- x_high(k)：目标成分 k 高可见度；
- x_low(k)：同一目标成分低可见度；
- x_absent(k)：保持其他干扰、接收机噪声和 nuisance 参数一致，只移除目标成分 k；
- 三者共享 receiver/noise/background identity，只改变目标成分存在性或相对功率。

方法暂命名为 Identity-Paired Counterfactual Visibility Evidence，IP-CVE。

必须先进行不训练最终分类器的 Stage P0 可辨识性准入：比较 matched counterfactual residual 与
receiver/time/component permutation 控制。只有弱正对 matched-absent 的 AUROC/AUPR 在五 seed 中
稳定超过控制，才允许训练小型标签特定 evidence readout；否则直接停止该创新。

若 P0 通过，再设计：

- 冻结或低学习率 B0 表征；
- 每标签 evidence score/map；
- matched weak-vs-absent ranking；
- high>low visibility ordering；
- 非目标标签不变性；
- 与 B0 logit 的有界风险接回；
- 推理时只输入单个样本，反事实样本只用于训练。

六、必须包含的实验矩阵

1. P0 无训练可辨识诊断；
2. B0、频次方法、普通高低功率 consistency、IP-CVE；
3. equal-capacity、unmatched-pair、component-permuted、time-permuted 控制；
4. frequency-balanced/label-LT × high/low-visibility 的 2×2 双轴因子实验；
5. 弱/强/负、逐标签、组合阶数、hard-negative、receiver cohort 分层；
6. 参数、MACs、延迟、显存；
7. 完全冻结后的全新 identity confirmation；
8. 若条件允许，真实/半实物/HIL 外部验证。

七、预注册硬门槛

P0 准入至少要求：

- matched residual 相对 permutation control 的三目标平均 weak-vs-absent AUROC >= +0.030；
- paired bootstrap 95% CI 下界 > 0；
- 至少 4/5 seed 为正；
- 任一目标标签不低于 -0.010；
- matched 与 unmatched identity 的差异方向正确；
- validation-to-early 泛化落差 < 0.15。

模型晋级至少要求：

- 相对 B0 和 equal-capacity control：macro-AUPR >= +0.005，CI 下界 > 0；
- 三目标平均 weak AUROC >= +0.020，CI 下界 > 0；
- 三目标平均 weak recall >= +0.025；
- 两个弱指标至少 4/5 seed 为正；
- 目标平均 FPR <= 0.20；
- 强正 recall 下降不超过 0.01；
- 任一目标 AUPR 下降不超过 0.01；
- FMB/SawFM/SinFM 不系统退化；
- 必须战胜 unmatched/permuted 信息破坏控制。

八、输出格式

请按以下结构输出：

1. 执行摘要；
2. 上一阶段逐模块失败复盘；
3. 跨轮次统一根因图；
4. 既有四类创新点证据审计表；
5. 创新点应保留、重写或放弃的决定；
6. 下一阶段科学假设；
7. 数据协议；
8. 网络/读出、损失、分类器、校准、阈值和训练参数；
9. 实验矩阵、控制和统计方法；
10. 硬门槛与停止规则；
11. SCI 证据链：能证明什么、不能证明什么；
12. 推荐论文叙事；
13. 下一步执行顺序。

所有提案必须标记为 proposed/not validated。不得把历史开发结果、P0 诊断或未来方案写成已经验证的
创新。最后给出一条明确总裁决：继续追求哪个创新点，冻结哪些旧组件，永久关闭哪些机制。
```
