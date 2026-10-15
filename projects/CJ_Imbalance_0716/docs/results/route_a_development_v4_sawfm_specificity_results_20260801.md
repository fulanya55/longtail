# Route A development-v4 SawFM 特异性因果对照结果

日期：2026-08-01

## 1. 本轮结论

`GradientContrastFixed1p5` 在新的 development-v4 上继续得到稳定的总体正增益：

- common-five macro-AUPR：`+0.00646`；
- common-five macro-F1：`+0.01163`；
- MFT/VFT/ISRJ mean weak recall：`+0.01691`；
- SawFM-present macro-F1：`+0.01621`；
- 五个种子的 target mean weak recall delta 全部为正。

但是，弱分量 Recall 没有达到预注册的 `+0.030` 效果门槛，因此本轮不是完整
复现。SawFM 特异性门槛也失败，因为 full-norm 随机方向和非配对种子方向在
5/5 种子上产生非有限 logits，LayerShuffled 在 1/5 种子上非有限，不满足
“所有控制输出有限”的预注册条件。

最终决策：`stop_and_retain_nonreplication`。本轮不允许访问任何测试集。

## 2. 最近三轮实验

| 轮次 | 数据角色 | macro-AUPR delta | macro-F1 delta | target weak recall delta | 决策 |
|---|---|---:|---:|---:|---|
| development-v3 | fresh development | +0.00624 | +0.01337 | +0.03283 | 通过 |
| test-v2 once | independent | +0.00648 | +0.01390 | +0.03048 | C 级通过，测试已消耗 |
| development-v4 | fresh development | +0.00646 | +0.01163 | +0.01691 | 效应为正但未过幅度门槛 |

## 3. 本轮科学问题

当前增益是否是 SawFM 特异、严格配对且具有结构的任务方向，而不是普通模型
平均、随机等范数扰动、参数位置置乱、其他标签方向或非配对方向？

## 4. 可证伪假设

如果 SawFM 条件配对方向是特异机制，则当前候选应：

1. 在新身份数据上复现既定效果；
2. 优于普通 midpoint；
3. 优于等范数随机和层置乱方向；
4. 优于其他标签屏蔽方向；
5. 优于非配对种子方向；
6. 所有变体输出有限且满足 FPR 边界。

第 2--5 项在所有可评价控制上获得支持，但第 1 项的弱 Recall 幅度和第 6 项
没有通过。

## 5. 数据角色与身份隔离

- 数据：development-v4 train-only；
- 样本数：25,200；
- sample seed：`[34000001,34025200]`；
- receiver_noise_seed：`[232677743,232702942]`；
- receiver identity unique：25,200；
- 数据 SHA256：
  `fbb6d43599985854ff56b634aa78db22decf6f64283e80e60bafe1e6e2f65499`；
- 与 v2 train、test-v2、v3 train 和密封 test-v3 的声明范围全部不重叠；
- identity audit：`test_files_opened=false`。

## 6. 基线和候选

正式训练端点：

- `C6_Joint`；
- `M5_GradientIsolated`；
- `MaskFMB/MaskSinFM/MaskMFT/MaskVFT/MaskISRJ`。

正式评价 12 个变体：C6、M5、fixed alpha、midpoint、随机等范数、层置乱、
五个 other-label contrast 和非配对种子 contrast。

所有端点共享 split、manifest、初始化角色、batch 顺序、训练预算、early-stop、
校准和阈值规则。全部 SawFM 共现样本均被保留。

## 7. 实现改动

新增：

- train-only v4 MATLAB generator；
- 不打开测试文件的 identity auditor；
- 标签损失屏蔽端点训练；
- 逐张量等范数随机方向；
- 同形张量循环置换和单张量内部置乱；
- other-label contrast；
- adjacent-seed non-paired contrast；
- 统一 `[-40,40]` logits 数值稳定契约；
- invalid non-finite control 的显式记录与失败门槛。

首次正式尝试在随机方向温度校准发生 overflow；统一裁剪后，full-norm random
仍产生非有限 logits。两次中断均保存为 `implementation_only` incident，未形成
正式性能门槛，也未据此缩小随机方向。

## 8. 正式定量结果

### 8.1 绝对结果

| 指标 | C6_Joint | M5 | Fixed1p5 |
|---|---:|---:|---:|
| macro-AUPR | 0.86594 | 0.86980 | 0.87240 |
| macro-F1 | 0.75172 | 0.75998 | 0.76334 |
| target weak recall | 0.29037 | 0.30362 | 0.30728 |
| target FPR | 0.19602 | 0.19849 | 0.19608 |
| SawFM-present macro-F1 | 0.70331 | 0.71697 | 0.71952 |
| SawFM-absent macro-F1 | 0.80228 | 0.80570 | 0.80987 |

### 8.2 候选相对 C6

| 指标 | Delta |
|---|---:|
| macro-AUPR | **+0.00646** |
| macro-F1 | **+0.01163** |
| target weak recall | **+0.01691** |
| target FPR | +0.00005 |
| SawFM-present macro-F1 | **+0.01621** |
| SawFM-absent macro-F1 | **+0.00759** |

每标签 AUPR 均为正：FMB `+0.00665`、SinFM `+0.00120`、MFT `+0.00597`、
VFT `+0.01164`、ISRJ `+0.00686`。

### 8.3 弱分量 Recall 的标签分解

| 标签 | Mean delta |
|---|---:|
| MFT | **+0.03293** |
| VFT | **+0.01898** |
| ISRJ | -0.00116 |

本轮弱 Recall 幅度下降主要来自 ISRJ 没有获得增益，而不是宏观排序能力消失。

## 9. 置信区间与种子一致性

候选 target weak recall delta：

- mean：`+0.01691`；
- 95% seed-bootstrap CI：`[+0.00679,+0.02703]`；
- std：`0.01283`；
- positive seeds：5/5。

逐种子：

| Seed | weak recall delta | macro-F1 delta | macro-AUPR delta |
|---:|---:|---:|---:|
| 2039 | +0.00485 | +0.01098 | +0.00499 |
| 2040 | +0.01677 | +0.00829 | +0.00525 |
| 2041 | +0.00375 | +0.01631 | +0.00915 |
| 2042 | +0.03256 | +0.00928 | +0.00707 |
| 2043 | +0.02664 | +0.01327 | +0.00586 |

方向总体有稳定正效应，但弱 Recall 的效应量对数据身份分布敏感。

## 10. 机制分析

### 10.1 可评价控制

| 方向 | weak recall delta | macro-F1 delta | 解释 |
|---|---:|---:|---|
| Fixed1p5 | **+0.01691** | **+0.01163** | 最强稳定方向 |
| ModelAverage0p5 | +0.00587 | +0.00379 | 普通 midpoint 只能解释部分增益 |
| LayerShuffled1p5 | +0.00473（4 seeds） | +0.00032 | 参数位置被破坏后效应接近消失 |
| MaskFMB contrast | +0.00527 | +0.00905 | 弱 Recall 明显低于 SawFM 方向 |
| MaskSinFM contrast | +0.01397 | -0.05090 | 弱 Recall 接近但整体识别严重崩坏 |
| MaskMFT contrast | -0.00550 | -0.00463 | 负结果 |
| MaskVFT contrast | -0.00907 | -0.00307 | 负结果 |
| MaskISRJ contrast | -0.00926 | -0.03126 | 负结果 |

候选超过最佳可评价通用控制 `+0.01105` weak recall；超过最佳 other-label
方向 `+0.00295`。这支持 SawFM 方向具有更好的整体效用，但没有通过预注册的
完整有限性门槛。

### 10.2 非有限控制

- RandomNormMatched1p5：0/5 seeds 有效；
- NonPairedSeedContrast1p5：0/5 seeds 有效；
- LayerShuffled1p5：4/5 seeds 有效。

随机方向与 SawFM 方向余弦接近 0；非配对方向余弦约在 `-0.008` 到 `0.109`。
同范数但失去学习结构或坐标配对后会失稳，说明参数方向的结构和配对很重要。
不过，这些控制无法提供完整的可比性能分布，因此不能据此宣布特异性门槛通过。

### 10.3 方向实际支持范围

checkpoint 差分审计确认，C6→M5 非零方向严格集中在：

- 最后共享 `temporal_backbone.0`：20 个浮点 state tensors；
- `classifier`：2 个浮点 state tensors。

冻结骨干 BatchNorm running statistics 和其他 state 的差分范数均为 0。原因是
C6/M5 使用相同初始化和 batch 顺序，冻结部分即使处于相同前向模式，其 buffer
更新也完全配对抵消。因此当前方向没有混入冻结骨干漂移，但共享块与分类器两部分
仍需要在 fresh development 上分离验证。

## 11. 失败原因和替代解释

1. fixed alpha 的宏观 AUPR/F1 增益在 v3、test-v2、v4 都相近，但 weak recall
   是阈值化且只统计弱正例的指标，对数据身份和标签弱分量构成更敏感。
2. v4 的收益集中在 MFT 和 VFT，ISRJ 平均略降，说明单一全局 alpha 不能保证
   三个目标标签同时改善。
3. 参数随机化和跨初始化任务向量在神经网络坐标中并非良性局部扰动；“同欧氏
   范数”不等于“同功能距离”。后续随机对照应使用结构化任务子空间或预注册的
   功能 trust region，而不是在本轮事后缩放。
4. MaskSinFM 说明提升目标 weak recall 可以通过牺牲整体任务获得，因此弱 Recall
   不能脱离 macro-AUPR/macro-F1 单独解释。

## 12. 证据等级

本轮为 `development`：

- 证明配对方向的宏观收益再次复现；
- 证明 weak recall 效应量没有稳定达到原门槛；
- 提供 SawFM 特异性的部分支持；
- 不构成新的 independent 或 external 证据；
- 不改变既有 test-v2 C 级冻结结论。

## 13. 预注册门槛

### 效果复现

8/9 checks 通过，仅 `target_weak_recall_gain_0p030` 失败；整体 gate：Fail。

### SawFM 特异性

- 超过所有可评价通用控制：Pass；
- 超过所有 other-label direction：Pass；
- 至少一组 margin `>=0.005`：Pass；
- SawFM-present 增益超过控制中位数：Pass；
- 所有控制输出有限：Fail。

整体 specificity gate：Fail。

## 14. 产物路径与哈希

- 预注册计划：
  `docs/plans/route_a_innovation_gap_and_development_v4_preregistration_20260801.md`；
- 数据 identity audit：
  `outputs/tapc/route_a_development_v4_generation/identity_audit.json`；
- runner：
  `experiments/run_route_a_development_v4_sawfm_specificity.py`；
- summary：
  `outputs/tapc/route_a_development_v4_sawfm_specificity_v3/route_a_development_v4_sawfm_specificity_summary.json`；
- receipt：
  `outputs/tapc/route_a_development_v4_sawfm_specificity_v3/run_receipt.json`；
- artifact manifest：
  `outputs/tapc/route_a_development_v4_sawfm_specificity_v3/artifact_manifest.json`。

## 15. 下一轮唯一最高优先级实验

使用新的 identity-isolated development-v5，开展“作用层与 alpha 的结构化
消融”，只回答：

> 当前稳定的 macro-AUPR/macro-F1 增益来自共享时序表示方向、分类器方向，还是
> 两者的固定组合；能否在不损伤 ISRJ 的条件下恢复 MFT/VFT/ISRJ weak recall
> 的一致增益？

预注册候选至少包括：shared-only、classifier-only、shared/classifier 分离 alpha、
固定 alpha 响应曲线，以及同一配对任务子空间内的正交等范数控制。不得继续使用
development-v4 评价新模型，不得访问任何测试集。
