# P3n-A 跨干扰条件监督主干：结果、失败归因与证据冻结

## 1. 结论先行

P3n-A 已按冻结协议完成 B1/B2/B3、3 个模型种子的正式 development 试验。结果为：**条件对称 MFT↔ISRJ margin 不成立为当前创新机制，不进入 P3n-B，也不得在已打开的 seed770000000 上继续调 margin、损失权重或阈值。**

本轮唯一成立的结论不是“margin 还需要再调”，而是：当前主要风险已经从单纯的 ultra-close MFT 混淆，扩展为 **25 dB 可见性过渡区的普遍分数重叠**；训练样本规模、参数优先 checkpoint 和 F1 阈值又进一步放大了该风险。

证据边界：formal development；`test_generated=false`、`test_opened=false`、`locked_test_used=false`。

## 2. 冻结协议

- 数据：全新 train seed760000000、development seed770000000。
- 身份：以 `receiver_noise_seed` 为物理身份，全部机制与功率视图成组；角色间零重叠。
- 场景：ID_control、low_count、close_spacing、ultra_close、weak_target。
- 功率：25/35/50 dB。
- 模型：`mlcnn_iq_grouped_task_routed`，相同主干、参数头和训练预算。
- 训练：20 epochs，batch 63，AdamW，lr `1e-4`，weight decay `1e-4`，warmup 2 epochs，cosine。
- 参数损失：`0.1 × masked Smooth-L1`，只使用 `label=1 AND parameter_mask=1 AND finite(target)`。
- checkpoint：early role 上 oracle parameter macro-NMAE 最小。
- 阈值：calibration role 上逐类 F1 网格。
- 种子：2041/2042/2043。

对照：

- B1：原始 BCE。
- B2：MFT=1、ISRJ=0 时，对 ISRJ BCE 施加 3 倍 hard-negative 权重。
- B3：标签条件化的对称 MFT↔ISRJ softplus margin；共存和共同缺失样本不施加互斥约束。
- B0：旧 main/P1n 只作为同一 development 上的外部参考，不是等数据量因果消融。

## 3. 正式结果

| 方法 | ISRJ Recall 均值 | 最坏单元 FPR 均值 | Macro-AUPR 均值 | Oracle 参数 NMAE | Predicted-gate 参数 NMAE | 25 dB Recall | 25 dB FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 原始 BCE | 0.8589 | 0.2591 | 0.9118 | 0.4336 | 0.4263 | 0.7615 | 0.2417 |
| B2 hard-negative | 0.8619 | 0.2731 | 0.9111 | 0.4336 | 0.4266 | 0.7708 | 0.2594 |
| B3 条件 margin | 0.8658 | 0.2677 | 0.9095 | 0.4336 | 0.4268 | 0.7708 | 0.2548 |

B3 的三种子最坏单元 FPR 为 `0.20645 / 0.39032 / 0.20645`，全部高于 `0.10`。B3 相对 B1 的 ISRJ Recall 差值为 `+0.02313 / +0.00729 / -0.00979`，未达到 3/3 同方向。B3 没有任何一个冻结核心指标能够稳定优于 B2。

旧外部参考在同一 development 上为：

| 外部参考 | ISRJ Recall 均值 | 最坏单元 FPR 均值 | Macro-AUPR 均值 |
|---|---:|---:|---:|
| old main | 0.8964 | 0.1441 | 0.9642 |
| old P1n | 0.9130 | 0.1172 | 0.9661 |

该差距不能直接归因于新结构，因为旧参考与 P3n 的训练数据规模和训练协议不同；它首先暴露的是“从头训练对照不够强”的实验设计问题。

## 4. 失败门

通过：身份零重叠、Macro-AUPR 退化不超过 0.005、oracle/predicted-gate 参数 NMAE 退化不超过 1%。

失败：

1. B3 相对 B1 的 Recall 未达到 3/3 非负；
2. B3 三个种子的最坏单元 FPR 均未达到 0.10；
3. 25 dB Recall 差值存在超过 0.02 的退化；
4. miss-penalty 与 worst-combination 未稳定非劣；
5. B3 未在至少两个核心指标上稳定击败 B2；
6. 总体方向不一致。

## 5. 原因分析

### 5.1 第一原因：25 dB 可见性过渡，而非只在 ultra-close 发生

每个方法、每个种子的最高 FPR 基本都出现在 25 dB。高风险同时覆盖 ID_control、weak_target、close_spacing 和 ultra_close。说明“ultra-close MFT 被误判为 ISRJ”只是旧证据中的一个局部表现；在本轮更广的机制覆盖下，主因是低功率时真阳性与困难负样本的分数分布整体重叠。

### 5.2 第二原因：参数优先 checkpoint 与分类安全目标冲突

冻结规则以 early oracle parameter macro-NMAE 最小选择 checkpoint。九个训练轨迹中，分类 Macro-AUPR 峰值多出现在 epoch 1–2，而参数最优 checkpoint 出现在 epoch 10、15 或 18。参数 checkpoint 对应的 early AUPR 相对峰值低 `0.0130–0.0278`。

这不能直接证明分类峰值 checkpoint 在 development 上一定通过，但足以证明当前单目标选模把“联合感知”退化成了“参数优先、分类安全让步”。下一轮必须保存同一训练轨迹的参数最优和受约束联合最优 checkpoint，不能继续只看参数 NMAE。

### 5.3 第三原因：新训练数据的规模不足以支撑与旧主干的公平比较

P3n fit role 为 441 个物理身份、15 个机制功率视图，共 6615 个训练视图。旧主干来自更广的训练协议。B1/B2/B3 之间是公平的，但 B0 与 B1 不是等数据量对照。因此本轮能够否定 margin 的相对优势，却不能证明旧主干架构本身优于新训练主干。

### 5.4 第四原因：F1 阈值放大了种子风险，但不是唯一原因

三种方法的 ISRJ 阈值随种子大幅变化，约为 0.18–0.46；低阈值的 seed2042 产生最大 FPR。说明 calibration role 的逐类 F1 目标没有显式控制最坏机制功率风险。

但在已消费 development 上进行的只读诊断显示：若事后把阈值抬到最坏单元经验 FPR 恰为 0.10，B1/B2/B3 的三种子 Recall 仍只约为 `0.70–0.75`。所以仅换阈值无法恢复旧 P1n 的召回水平，表征重叠仍然存在。该诊断不得用于选择新阈值或模型。

### 5.5 第五原因：条件 margin 优化了容易样本，没有制造新信息

B3 的 margin loss 很快接近零，但 development 的安全指标没有改善，表明多数标签条件样本已容易满足 logit 排序；真正困难的 25 dB 样本仍然重叠。继续搜索 `lambda_xci` 或 margin 只是在已知表示上改变决策边界，不构成新的可辨识信息源。

## 6. 当前可以写入论文的边界

可以写：

- P1n 在旧开发证据和新身份验证中存在稳定但缩小的 Recall 收益；
- 当前 Level-4 风险主要受低功率可见性过渡和有限样本风险迁移控制；
- 普通 hard-negative 与标签条件 margin 均不能在相同数据、预算和参数任务约束下稳定解决该风险；
- 联合感知系统必须把 checkpoint 选择和最坏组校准纳入方法，而不能只报告 oracle-positive 参数误差。

不能写：

- 条件 margin 已构成创新；
- ultra-close MFT 是唯一失败原因；
- B3 优于普通 hard-negative；
- 当前系统已达到 Level 4；
- development 结果等价于 locked-test 结论。

## 7. 下一轮最小问题

下一轮不增加新专家，先回答：**失败主要来自训练覆盖不足、参数优先 checkpoint，还是即使修复二者后仍存在低功率表示不足？**

只有等数据规模、双 checkpoint、有限样本安全校准仍失败，才允许进入低功率延迟峰/时频双视图表征。这样能避免把数据量、选模和阈值问题误写成网络创新。

## 8. 完整性

- P3n 专项测试与相关回归测试：27 passed。
- 全量测试：223 passed in 8.38s。
- summary SHA-256：`E9C13159CF3951994FC45F3409B7BA4785F884C25C89F5A1B28A339930D5621A`。
- identity audit SHA-256：`993F0FECC3AF37394685A6B23F5FE775AE4A4838F885D4B0763685985694ABE9`。
- `test_generated=false`。
- `test_opened=false`。
- `locked_test_used=false`。
