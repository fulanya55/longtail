# Route A 双轴长尾复合识别证据差距与 Frequency Reconstruction V2 终局报告

日期：2026-08-08  
终局：**Terminal N / frequency_reconstruction_failed**  
证据范围：全新 identity、development-only；`locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`

## 1. 执行结论

本轮已完成新的 `dual-axis-frequency-development-v2`、Label/Combination IF10/20/50、五个 seed、35 个正式训练单元、完整对照和 5,000 次 receiver-identity paired bootstrap。Frequency hard gate 明确失败，因此按预注册停止 RawSaw 可见度复制、2×2 联合试验和 confirmation。

核心结果不是“没有达到显著”，而是主方向稳定反向：

- IF20 Label-LT 与 Combination-LT 上，F3 相对逐 seed/cell 的 `best(F0,F1,F2)` 平均 macro-AUPR 差为 **−0.009419**，目标为 `>=+0.005`。
- 5,000 次 paired bootstrap 的 95% CI 为 **[−0.010304, −0.008514]**，完整位于零以下。
- 10/10 个 IF20 seed-cell 均为负，5/5 个 seed 的 Label/Combo 平均差均为负。
- F3 相对 random-pair control 的平均差为 **−0.0000012**，只有 1/5 seed 为正，说明 co-occurrence-matched pairing 没有形成可辨认机制收益。
- FB safety 通过：平均差 **−0.001359**，高于 `−0.005` 下限；但安全不等于有效。

因此，当前“冻结 B0 表征后，仅靠 label-balanced 线性 cRT、标签共现匹配排序和 anchor 重训分类头”的 frequency reconstruction 路线应关闭，不能继续在同一 evaluation source 调权重、margin、pair 规则或 epoch。

## 2. 当前距离“双轴长尾复合识别”还有多远

双轴主张至少需要依次建立三条主效应和一条外部确认链：

| 证据环节 | 当前状态 | 还缺什么 |
|---|---|---|
| 数据与身份隔离 | 已验证 | 无；新 identity、角色零重叠、VH/VL 精确配对均通过 |
| 标签频次轴 H-F | **正式失败** | 候选需在 Label-IF20 与 Combo-IF20 同时稳定胜过强基线、CI>0、尾标签安全 |
| 可见度轴 H-V | 本轮未执行 | 只有新的 H-F 通过后，才能在新 identity 上复制 RawSaw 与信息破坏 controls |
| 双轴联合 H-J | 本轮未执行 | H-F、H-V 分别通过后，才允许新 2×2、joint effect 和 interaction |
| Fresh confirmation | 本轮未执行 | 冻结最终机制后使用完全新 identity，一次性验证并保留至少 70% development 效应 |

以预注册主门计数衡量，frequency gate 的 9 个判定项中只通过 2 项：FB safety，以及 F3 相对本轮 F2 的微小平均优势；其余 7 项失败。以完整双轴链衡量，目前为：

- frequency 正式主效应：0/1；
- visibility 新身份复制：0/1，未授权执行；
- joint/interaction：0/1，未授权执行；
- fresh confirmation：0/1，未授权执行。

所以当前距离 SCI 级“双轴长尾复合识别创新点”不是差一个阈值或一个消融，而是第一条 frequency 因果链尚未成立。现有结果只能支持“严谨否定了一个冻结线性分类器重构机制”，不能支持双轴模型有效性主张。

## 3. 本轮实际验证了什么

### 3.1 已验证的数据证据

- 生成 25,200 个全新 receiver identity，每个 identity 具有 SawFM 20 dB/30 dB 两个严格配对视图。
- 63 个非空六标签组合，每组合 400 个 identity；六标签总体正样本各 12,800。
- optimization/early/calibration/threshold/evaluation 分别为 15,120/2,520/2,520/2,520/2,520，角色两两零重叠。
- FB 为 2,400 identity、六标签各 1,200 正样本。
- Label-IF10/20/50 实际 IF 分别为 10、19、48.4375；Combination-IF10/20/50 实际 IF 分别为 10、20、48，均在 5% 离散容差内。
- 每个 manifest 的每个标签均同时具有正负样本；首次 Label-IF20 无 FMB 负样本的问题在任何梯度更新前被拦截并保留失败收据。

### 3.2 已验证的训练与工程证据

- 共享 backbone 为 `dfms_iq`：输入 `[B,2,3000]`、width 16、128 维 feature、dropout 0.2、线性六标签头。
- 正式完成 5 seeds × 7 regimes = 35 个训练单元；每个单元保留 checkpoint、history、evaluation prediction、pair-map hash 和 receipt。
- F3 只替换 `Linear(128,6)`，总推理参数仍为 257,511；实际训练参数为 774。
- RTX 5080 上，Conv1d/Linear 估算为 211,857,536 MAC/sample；batch-1 平均 3.363 ms，batch-64 吞吐约 12,929 sample/s。
- 所有评价均来自本轮新 evaluation role；旧 R2 sample predictions、locked test、test-v2、test-v3 均未使用。

### 3.3 已验证的科学结论

1. **Label-frequency 是真实难点，Combination-frequency 当前存在明显天花板。** ComboIF10/20/50 的各模型平均 macro-AUPR 大多在 0.996–0.999，方法差只有 `1e-5` 到 `1e-4` 量级；该场景不适合继续证明微小分类头创新。
2. **DB 仍是 Label-LT 强基线。** LabelIF20 五 seed 平均 macro-AUPR：F0=0.840182、F1 DB=0.854177、F2=0.835840、F3=0.835844。F3 没有恢复 DB 的优势。
3. **hard-pair 机制没有独立贡献。** F3 与 random-pair control 几乎完全相同；F3 与 no-rank 的差仅约 `4.9e-6`。
4. **冻结线性头无法重建被表征压缩掉的频次条件边界。** LabelIF20 中多数 head 在第 1 epoch 即达到 early 最佳，继续优化不会形成新的可分结构。
5. **失败集中在头部/关键标签，不是统一安全退化。** LabelIF20 中 F3 相对 best baseline 的平均差：FMB −0.070901、SawFM −0.032071、VFT −0.014120；SinFM 和 ISRJ略有正差，但不足以抵消核心退化。

## 4. Hard-gate 判定

| Gate | 结果 | 证据 |
|---|---|---|
| G1 IF20 macro mean `>=+0.005` | 失败 | −0.009419 |
| G2 bootstrap CI lower `>0` | 失败 | CI [−0.010304, −0.008514] |
| G3 至少 4/5 seed 正向 | 失败 | 0/5 |
| G4 三个最少标签平均差 `>=+0.010` | 失败 | −0.005187 |
| G5 FB macro `>=−0.005` | 通过 | −0.001359 |
| G6 任一 primary-label 不低于 −0.010 | 失败 | 最差 LabelIF20/FMB = −0.070901 |
| G7 IF10/IF50 各至少 3/5 非负 | 失败 | LabelIF50、ComboIF50 均 0/5 非负 |
| G8a 胜 ordinary cRT | 通过（幅度很小） | +0.000436，5/5 seed 正向 |
| G8b 胜 random-pair | 失败 | −0.0000012，1/5 seed 正向 |

任一主门失败即禁止进入 visibility/joint；本轮共 7 项失败，故终局无歧义。

## 5. 失败原因复盘

### 5.1 第一原因：F0 的 128 维冻结表征没有保留可由线性头恢复的 frequency-conditional 几何

F3 与 F2 的差极小，且 LabelIF20 heads 多在第 1 epoch 最优，说明问题不在“head 没训练够”，而在冻结 feature 上不存在足够稳定的新线性分界。分类头重训无法创造主干已经压缩掉的条件信息。

### 5.2 第二原因：标签组合 Jaccard 不是有效的困难度度量

hard-pair 与 random-pair 几乎相同，说明按其他五标签的组合相似度选择 negative，并没有在 B0 feature 空间中找到更接近决策边界的负例。组合模式相似不等于信号特征相似，也不等于当前分类器的高风险 negative。

### 5.3 第三原因：Label-LT 与 Combo-LT 的难度不对称

Combo-LT evaluation 过于容易并接近饱和，无法提供足够动态范围；Label-LT 才暴露频次失衡，但其最强基线是端到端 DB，而非冻结 cRT。这意味着下一步不能再把两种长尾简单视为同一个 head-level 问题。

### 5.4 第四原因：candidate 的收益来自一般重训，不来自预设机制

F3 虽微弱胜 F2，但不胜 random-pair，no-rank 差异也几乎为零。现有证据无法把收益归因于 co-occurrence matching、ranking 或 anchor；因此不能形成 SCI 机制创新链。

### 5.5 协议解释边界

最终 Stage-0 prereg 将 F2 operationalize 为 label-balanced head-only cRT，并把 anchor 写为归一化权重/偏置漂移；父级 Goal 文本曾写“普通 BCE”与 bias 项 0.1 系数。该差异必须披露，因此本轮对“普通 BCE cRT”和精确父级 anchor 形式的比较是有限的。与此同时，主否定结论并不依赖 F2：F3 在 10/10 IF20 cell 中都输给包含 F0/F1 的 best envelope，且 CI 完整为负，也不胜 random-pair。

## 6. 下一条长线技术方向

不建议在本轮 evaluation 上继续搜索 rank weight、margin、anchor、pair seed、head depth 或 threshold。下一条路线若继续，必须使用全新 identity source，并改变问题层级：从“冻结线性头重训”转向“受限表征适配”。

建议的下一阶段为 Frequency Representation V3：

1. 保留 B0 stem 与 stage1–3，只有 stage4/feature projection 使用 rank-4 低秩 adapter，限制容量与风险。
2. 每标签使用独立的小型 evidence projection，但保留共享 128 维 safety path；证明收益来自 label-conditioned representation，而不是扩大整个 backbone。
3. hard negative 改为 optimization feature 空间的近邻 negative，并同时约束目标标签缺失、其他标签结构相似；不再只依赖标签 Jaccard。
4. 将 Label-LT 设为主证据；重新设计更困难的 Combo-LT development，避免 0.998 AUPR 天花板。
5. 对照必须包含 DB、当前 head-only F3、random feature-neighbor、无 adapter、无 label-conditioned projection 和容量匹配 MLP。
6. 先做 representation identifiability gate：冻结 probe 或 kNN 必须证明 tail-label 局部可分性改善，再授权完整训练。
7. 仍执行 5 seeds、receiver bootstrap、逐标签安全、IF10/20/50、fresh confirmation；未通过时再次 Terminal N，不回到同一 source 调参。

## 7. 当前可发表边界

当前不具备“双轴长尾复合识别方法有效”的 SCI 主张。可以保留的可靠内容是：

- 新 identity 双轴长尾评估协议与严格证据链；
- 对 DB、ordinary cRT、co-occurrence ranking cRT 的系统负结果；
- “label-combination similarity 不能替代 feature-space hard-negative risk”的实验证据；
- “冻结共享 128 维表征上的线性解耦不足以修复 label-frequency long tail”的边界结论。

这些内容可作为后续表征适配方法的动机与强基线，但不能单独包装成已成立的新模型创新点。
