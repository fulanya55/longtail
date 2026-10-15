# Route-A 标签长尾方法阶段 2：label-IF=2 BCE 瓶颈筛查（2026-09-01）

## 1. 试验范围

本阶段固定使用上一阶段确认的 `label_IF2` development-v2 清单，训练量 1600，实际 IF=2.0，六类为 FMB、AMA、SawFM、SinFM、MFT、ISRJ。F2_DFMS_Full、真实 I/Q、C1 初始化、AdamW、8 epoch 和数据角色均保持不变；不读取 locked test，E5 未修改。GPU 为 RTX 5080。

候选损失：

- BCE：普通 BCE 锚点；
- CB_BCE：effective-number 正样本修正，`beta=0.99`；
- ASL：`gamma_pos=0`、`gamma_neg=4`、`clip=0.05`；
- Positive-Focal：`gamma=2`。

每个候选均先完成 3 种子筛查；Positive-Focal 在 3 种子上同时改善后，追加 seed 2042、2043，形成 5 种子配对确认。

## 2. 三种子筛查结果

| 方法 | Macro-AUPR | Macro-F1 | 组合完全匹配率 | Hamming |
|---|---:|---:|---:|---:|
| BCE | 0.996791 | 0.959404 | 0.825358 | 0.034315 |
| CB_BCE | 0.996746 | 0.959725 | 0.826335 | 0.034098 |
| ASL | 0.996841 | 0.988729 | 0.933594 | 0.011285 |
| Positive-Focal | 0.997716 | 0.971162 | 0.868327 | 0.025011 |
| DRW（BCE→CB） | 0.996768 | 0.958901 | 0.823568 | 0.034749 |
| ComboSampler | 0.996673 | 0.955595 | 0.816406 | 0.037109 |

相对 BCE：

- CB_BCE 基本为零增益；
- ASL 组合率提升约 10.8 个百分点、Hamming 明显下降，但多类 FPR 增加；
- Positive-Focal 在三个筛查种子上均提高组合率、Macro-F1 和 Macro-AUPR。
- DRW 和组合均衡采样均未超过 BCE 锚点，前者基本复现 BCE，后者在一个种子上明显退化。

## 3. Positive-Focal 五种子确认

### 3.1 配对均值

| 指标 | BCE 五种子均值 | Positive-Focal 五种子均值 | 差值 |
|---|---:|---:|---:|
| Macro-AUPR | 0.996153 | 0.997803 | +0.001650 |
| Macro-F1 | 0.940715 | 0.961544 | +0.020830 |
| 组合完全匹配率 | 0.765332 | 0.830762 | +0.065430 |
| Hamming loss | 0.049642 | 0.033626 | -0.016016 |

五个配对差值均为正（Hamming 为负），但样本量只有 5，统计检验仅作为开发证据，不替代独立数据确认：

- Macro-AUPR：Wilcoxon 单侧 `p≈0.034`；
- Macro-F1：`p=0.03125`；
- 组合完全匹配率：`p=0.03125`；
- Hamming loss（更小为优）：`p≈0.03125`。

### 3.2 尾类与误报

Positive-Focal 相对于 BCE 的五种子平均变化：

- ISRJ AUPR：约 `+0.0055`；
- ISRJ recall：约 `+0.0566`；
- ISRJ FPR：约 `-0.0058`；
- MFT recall：约 `+0.0101`；
- FMB、AMA、SawFM、SinFM 未出现系统性 FPR 爆炸。

ASL 虽然组合率更高，但 FMB、SawFM、MFT 等类别的 FPR 分别约为 0.0315、0.0119、0.0117，不能通过当前的误报安全门槛，因此暂不晋级。

## 4. 当前结论

1. `label_IF2` 已构成比旧 P3 更可靠的 BCE 瓶颈；旧 P3 的名义 IF=5 实际只有约 1.4–1.5。
2. Class-Balanced BCE 没有缓解瓶颈，说明仅按标签频率修正不足以修复组合边界。
3. ASL 的收益主要来自提高整体正类响应，伴随误报增加，暂不作为候选创新。
4. Positive-Focal 是当前唯一通过“3 种子筛查 + 2 个补充种子”的候选，表现为尾类召回和组合识别同时改善，且平均 FPR 未增加。
5. DRW 和 ComboSampler 暂停扩展，不进入确认。
5. 该结果仍属于 development-v2；不能外推到 locked test，也不能在未完成独立重复前宣称最终创新成立。

## 5. 下一道门槛

下一步应在不改变 `label_IF2` 清单的前提下：

1. 固定 Positive-Focal 超参数，不再在确认种子上调参；
2. 与 BCE、CB_BCE、ASL 保持同一训练预算，补齐逐类 tail-AUPR、FPR、seen/unseen 组合差和成本审计；
3. 增加严格组合不相交的 development-v2 验证，确认收益不是已见组合重建；
4. 若 Positive-Focal 仍通过，再比较 Logit Adjustment、Deferred Reweighting 和组合感知采样；
6. 所有候选完成预注册后，才申请新的 locked-test 授权。
