# Route A H1k：MFT 匹配滤波有效支持度的复合源验证结果

## 决定

**停止 `correct_LFM_matched_filter_Renyi2_effective_delay_support` 作为弱 MFT 复合识别统计量。** 它在 H0j 的受控单 MFT 副本数干预中强可辨识，但在全新六类复合源的弱 MFT、组成匹配负例上稳定劣于同容量的相位置乱参考。不得把该统计量接入分类头、融合、残差、校准或阈值，也不得在 H1k 身份上调整 ROI、熵阶、匹配窗、归一化或参考。

## 数据与冻结边界

- 新 train-only 六类来源：5,040 条、63 个组合、每组合 80 条；SHA-256 `97c3a9b8c17b53f6c777d3a0cb7672f8414ffae8821631d7b7edb33cd16f2659`。
- 5,040 个唯一 receiver-noise identities，范围 `1028677743--1028682782`；fit/validation/evaluation 为 `3024/1008/1008`，每一角色的六标签正例均为 1,536/512/512。
- 与全部历史 live train 来源的 receiver identity 交集为零；test-v3 仍为 `access_authorized=false`、`analysis_allowed=false`、`used=false`。
- 评价集仅取弱 MFT 正例和其他五标签向量完全一致的 MFT 缺失负例：113 个弱正、480 个负、30 个共享上下文。

## 完全固定的比较

候选是正确已知 LFM 的圆周匹配滤波在固定 `[400,1000)` 延迟 ROI 内的 Renyi-2 有效支持度。控制仅将 LFM 的相位顺序按固定置乱替换，长度、模值、FFT、ROI 与归一化保持相同。两臂均无训练参数、无读出头、无阈值、无校准、无融合。

对固定 evaluation 行作 10,000 次类别分层、成对 bootstrap；每个抽样行始终同时保留候选和控制分数。

| 指标 | 候选 | 控制 | 候选 − 控制 |
| --- | ---: | ---: | ---: |
| 弱 MFT vs 组成匹配负例 AUROC | 0.38599 | 0.55454 | **−0.16855** |
| 弱 MFT vs 组成匹配负例 AUPR | 0.14597 | 0.22162 | **−0.07566** |
| AUROC 差 95% CI | — | — | **[−0.23240, −0.10387]** |
| AUPR 差 95% CI | — | — | **[−0.13319, −0.04212]** |

五个预注册 receiver 子折的 AUROC 差均为负：`−0.18178`、`−0.16043`、`−0.11224`、`−0.29265`、`−0.07653`。这排除了由单一子组造成的偶然反向结果。

## 为什么 H0j 通过而 H1k 反转

H0j 问的是“在只有 MFT 且其他随机量成对保持时，8 个延迟副本是否比 3 个副本产生更宽的正确匹配滤波支持度？”其结果 AUROC 1.000、相对控制 +0.358，验证了物理 sanity check。

H1k 的问题则是“弱 MFT 是否相对同组成 MFT 缺失复合样本有特异增量？”复合中其他成分与噪声会在同一全局匹配滤波 ROI 留下响应；弱 MFT 本身又最容易被强共现成分遮蔽。全局能量支持度因而测量的是“ROI 内所有 LFM 型或相关结构的扩散度”，而非 MFT 成分的可归因证据。相位置乱控制反而保留了某些复合背景变化，导致其排名更高。

因此 H0j 的作用是排除“该量在物理上完全无效”，并不等价于复合检测可用性；H1k 的反向、跨所有子折一致的结果足以关闭它的检测路线。

## 对后续研究的约束

1. 禁止以该全局有效支持度做任何 ROI、熵阶、匹配滤波、归一化、阈值、权重、读出头或残差的搜索。
2. H0j/H1k 身份均已被一次性使用，不得作为后续机制选择集。
3. 若重启 MFT，候选必须具有**副本可归因**而非全局支持度：在未知共现背景下能够把 MFT 延迟峰与其它类型的匹配滤波响应区分开；并需在新的配对 H0 后以新的复合身份做 H1。
4. H1k 失败不影响已冻结的窄 ISRJ aligned-periodicity 证据，也不改变 RawSaw SawFM 结果；它只关闭本 MFT 标量路线。

## 产物

- H0j 结果：`docs/results/route_a_h0j_mft_matched_filter_support_results_20260802.md`
- H1k 来源审计：`outputs/tapc/route_a_h1k_mft_support_generation/identity_audit.json`
- H1k 机器结果：`outputs/tapc/route_a_h1k_mft_support/route_a_h1k_mft_support_summary.json`
- H1k 计划：`docs/plans/route_a_h1k_mft_matched_filter_composite_20260802.md`
