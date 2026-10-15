# 路线A：SawFM stage2同身份功率Teacher–Student试验结果（2026-07-25）

## 1. 结论

本轮完成3 seeds × 3种SawFM stage2专项模型的正式试验，并在oracle晋级门通过后执行最大0.20的H2受限残差回接。锁定测试集未使用。

核心结论：

1. **SawFM弱证据可以通过正确的组合监督稳定提取。** 完整方案S2在45 dB独立development上的SawFM-vs-matched-negative AUROC为0.7972、AUPR为0.7949，validation—development绝对落差仅0.0097。
2. **hard negative是主要有效因素，teacher一致性单独不稳定。** S2相对S1的development AUROC提高0.1739，3/3 seeds为正；S1相对S0只提高0.0260，且只有1/3 seed为正。
3. **原label-LT H2在该弱可见性组合上存在明显组合域缺口。** H2在相同45 dB匹配子集上的SawFM AUROC仅约0.400，而S2专项证据头达到0.797。
4. **小残差回接不适合当前融合关系。** 受限回接只提高AUROC 0.00219、AUPR 0.00154。原因不是专项证据弱，而是H2在该子域的排序已接近反向，平均绝对残差只有约0.0058，无法改变决策顺序。
5. 本轮验证了“stage2专项证据头”有效，但尚未验证“六类最终网络”整体提升。下一瓶颈已经从特征学习转移到专家路由与分类器融合。

## 2. 数据协议审计

原label-LT清单每seed有400个SawFM阳性，但只有1个同时包含MFT/VFT/ISRJ，无法构造有效SawFM功率弱化对。因此辅助训练从同一合法`train_indices`中分层抽取：

- 28种去除SawFM后的其他标签组合；
- 每组合40个SawFM阳性；
- 每个阳性匹配一个其他标签完全相同、SawFM缺失的hard negative；
- 共1,120个阳性身份和1,120个匹配负身份；
- fit/validation为896/224 identity pairs；
- development在独立early身份中形成94～100对匹配样本；
- teacher=10 dB，students=40/50 dB，held-out development=45 dB；
- 所有功率视图按`receiver_noise_seed`配对；
- 训练与development身份重叠为0；
- locked test使用次数为0。

## 3. 模型与消融

F2完全冻结，从stage2 real/imag `[B,32,1500]`下采样并拼接为`[B,64,375]`。新增3,651参数的rank-8残差adapter和时间归一化SawFM证据头。

| 变体 | Teacher | 负样本 | 一致性 | 排序 |
|---|---|---|---|---|
| S0 | 无 | 随机Saw缺失 | 无 | 无 |
| S1 | 10 dB | 随机Saw缺失 | score+map | 无 |
| S2 | 10 dB | 相同其他标签组合 | score+map | matched ranking |

## 4. 三种子结果

| 变体 | Validation AUROC | Development AUROC | Development AUPR | 泛化落差 |
|---|---:|---:|---:|---:|
| S0 Student+RandomNeg | 0.5958 | 0.5973 | 0.5668 | 0.0398 |
| S1 Teacher+RandomNeg | 0.6450 | 0.6233 | 0.6142 | 0.0218 |
| S2 Teacher+HardNeg | **0.8061** | **0.7972** | **0.7949** | **0.0097** |

### 4.1 S2逐seed

| Seed | 最佳epoch | Validation AUROC | Development AUROC | Development AUPR | Gap |
|---:|---:|---:|---:|---:|---:|
| 2041 | 30 | 0.8810 | 0.8785 | 0.8870 | 0.0025 |
| 2042 | 30 | 0.9336 | 0.9348 | 0.9307 | 0.0012 |
| 2043 | 30 | 0.6039 | 0.5784 | 0.5670 | 0.0255 |

S2平均值高且泛化落差小，但seed 2043仍未达到0.60，说明不同F2检查点对SawFM stage2可分性仍有明显影响。当前结果应报告均值、标准差和逐seed结果，不能只报告0.797均值。

### 4.2 消融增量

| 对比 | Development AUROC增量 | AUPR增量 | 正向seeds |
|---|---:|---:|---:|
| S1 - S0 | +0.0260 | +0.0474 | 1/3 |
| S2 - S0 | +0.1999 | +0.2281 | 2/3 |
| S2 - S1 | **+0.1739** | **+0.1807** | **3/3** |

teacher/student一致性本身不是稳定贡献。组合匹配hard negative使模型必须回答“在几乎相同的复合干扰背景中，SawFM到底存在还是缺失”，这是本轮最有效的监督机制。

## 5. Oracle晋级门

S2通过全部预注册条件：

- development AUROC均值≥0.60：通过，0.7972；
- 至少2/3 seeds≥0.60：通过，2/3；
- 泛化落差≤0.15：通过，0.0097；
- 相对S0提高≥0.02：通过，+0.1999；
- 至少2/3 seeds优于S0：通过，2/3；
- development AUPR不低于S0：通过，+0.2281；
- 身份及locked-test审计：通过。

因此按预注册规则执行了受限残差回接。

## 6. H2受限残差回接

在相同45 dB匹配development子集上：

| Seed | H2 AUROC | 回接后AUROC | 增量 | 最大残差 |
|---:|---:|---:|---:|---:|
| 2041 | 0.4456 | 0.4494 | +0.00375 | 0.0449 |
| 2042 | 0.3955 | 0.3973 | +0.00180 | 0.0716 |
| 2043 | 0.3585 | 0.3596 | +0.00102 | 0.0517 |
| 平均 | 0.3999 | 0.4021 | **+0.00219** | — |

平均AUPR增量为+0.00154。残差平均绝对值约0.0058，远小于0.20上界。

失败原因：

1. H2在SawFM+扫频分量组合上的训练覆盖极少，45 dB子域排序低于0.5；
2. S2证据头输出的是一个新的有效决策边界，而不是对H2边界的微调；
3. `exp(-abs(H2_logit))`门控进一步缩小了残差；
4. 即使提高残差上限，也可能破坏H2在普通样本上的表现，因此不能直接扩大权重。

结论：下一步应验证“专家路由/替代分类器”，而不是继续使用加性小残差。

## 7. 对网络问题的更新判断

### 已验证

- F2 stage2包含可用于SawFM弱可见性识别的信息；
- rank-8 adapter足以提取该信息，不需要扩宽主干；
- 时间归一化避免了上一轮专家logit无界问题；
- 同身份teacher提供方向约束，但不是主要增益来源；
- 组合匹配hard negative是最强贡献；
- 原H2的主要问题是训练组合覆盖与分类边界，而不是stage2完全没有特征。

### 尚未验证

- 专项证据头是否能在普通功率、Saw强阳性和全部Saw缺失样本上保持校准；
- 如何无功率元数据地判断何时启用专家；
- 专家路由后六类Macro-AUPR、Hamming、FPR和弱召回是否整体改善；
- seed 2043低表现来自F2表示差异还是adapter优化差异。

## 8. 下一决策

保留F2和S2专项证据头，不继续扩大或解冻主干。下一轮应做冻结双专家融合：

1. H2负责普通六类分类；
2. S2负责SawFM弱可见性证据；
3. 使用独立校准集训练一个低容量SawFM路由门，而非直接相加；
4. 对照固定H2、专项头替代、固定凸组合和低容量路由；
5. 同时在原early角色与45 dB弱可见性角色报告Saw AUPR/AUROC/FPR；
6. 只有普通角色Saw AUPR非劣、弱可见性AUROC明显提高且全局Macro-AUPR非劣，才进入六类联合验证。

## 9. 产物

- 汇总：`outputs/tapc/route_a_sawfm_stage2_power_teacher_student/route_a_sawfm_stage2_power_teacher_student_summary.json`
- 每seed抽样与pair manifest；
- 9个模型检查点、训练历史、预测和回执；
- oracle门通过后的3份受限回接结果；
- 预注册方案：`docs/implementation/route_a_sawfm_stage2_power_teacher_student_plan_20260725.md`。
