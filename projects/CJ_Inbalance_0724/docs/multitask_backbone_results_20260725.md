# 无长尾多任务骨干对比结果（2026-07-25）

## 结论

在相同数据、统一 256 维适配器、统一分类头和统一六专家参数头下，`complex1d` 与 `iq_grouped` 均能稳定联合学习识别和参数提取；全量三种子结果没有形成一个在分类和参数两方面都 3/3 seeds 一致胜出的骨干。

`iq_grouped` 的三种子平均联合损失、macro-F1、subset accuracy 和参数 macro NMAE 略优，且总参数量更少，因此作为后续默认骨干；`complex1d` 的 macro-AUPR 略高，并在 12 个参数目标中的另外 6 个更优，应保留为关键对照。两者参数 macro NMAE 的均值差仅 0.00016，不能表述为参数能力有实质差异。

## 审计与协议

- 根训练集共有 25,200 条，63 种非空六标签组合各 400 条；不构造长尾。
- 固定 split seed 20260724，按组合分层为 22,680 条训练和 2,520 条验证。
- 12 个参数列为 `0,2,11,14,16,18,27,28,48,49,52,53`。
- 每个目标在开发子集中的有效阳性均为 800；负类掩码为真计数均为 0；有效目标非有限值计数均为 0。
- 参数损失只在 `label=1 AND parameter_mask=1` 计算。
- 损失：`BCEWithLogits + 0.1 * SmoothL1(z-score parameter)`；参数梯度回传骨干。
- AdamW，学习率 3e-4，weight decay 1e-4，batch 64，最多 30 epoch，patience 6。
- checkpoint 按最小验证联合损失选择；锁定测试集未使用。

## 五骨干三种子筛选

筛选每种组合使用 20 条训练、5 条验证（1260/315），最多 12 epoch。

| 骨干 | macro-AUPR | macro-F1 | subset acc | 参数 macro NMAE | 结论 |
|---|---:|---:|---:|---:|---|
| Complex1D | 0.90507 | 0.78370 | 0.26138 | **0.72710** | 晋级 |
| IQ-Grouped | **0.90929** | **0.79182** | **0.30159** | 0.73454 | 晋级 |
| ConvNeXt-Tiny-1D | 0.75422 | 0.68528 | 0.09735 | 0.80997 | 小预算下欠拟合，不晋级 |
| EfficientNet-B0-1D | 0.50741 | 0.26199 | 0.01587 | 0.87120 | 小样本 BN 验证失配，不晋级 |
| EfficientNet-B1-1D | 0.50472 | 0.29942 | 0.01587 | 0.87515 | 小样本 BN 验证失配，不晋级 |

EfficientNet 在训练指标改善的同时，验证结果停留在近随机水平，最佳 checkpoint 多在 epoch 1–2。该结果只说明它们不适合当前小样本筛选配方，不能外推为全量数据下骨干无效。ConvNeXt 参数量约 2,699 万，在相同短预算下学习显著慢于两种轻量骨干。

## 全量三种子确认

| 骨干 | 参数量 | macro-AUPR | macro-F1 | subset acc | 参数 macro NMAE | 联合损失 |
|---|---:|---:|---:|---:|---:|---:|
| IQ-Grouped | **115,138** | 0.98107 ± 0.00111 | **0.93061 ± 0.00224** | **0.67487 ± 0.00931** | **0.54097 ± 0.00998** | **0.18956 ± 0.00684** |
| Complex1D | 146,002 | **0.98128 ± 0.00112** | 0.92950 ± 0.00150 | 0.66746 ± 0.00561 | 0.54112 ± 0.01206 | 0.19189 ± 0.00278 |

训练均值常数参数基线的验证 macro NMAE 为 0.85819。Complex1D 和 IQ-Grouped 分别将其降低 36.95% 和 36.96%，说明参数头确实学到可辨识信息，而不是仅接近常数输出。

逐 seed 比较中，IQ-Grouped 的参数 NMAE 在 seed 2041/2042 更低，在 seed 2043 更高；macro-AUPR 仅在 seed 2043 更高。没有任何核心指标满足 3/3 seeds 同方向胜出，因此两骨干应视为性能近似的 Pareto 候选。

## 逐参数目标

| 目标 | Complex1D NMAE | IQ-Grouped NMAE | 较优骨干 |
|---|---:|---:|---|
| FMB JNR | **0.48226** | 0.49845 | Complex1D |
| FMB interference_bandwidth | 0.43426 | **0.33207** | IQ-Grouped |
| SawFM JNR | **0.40431** | 0.42495 | Complex1D |
| SawFM sweep_period | **0.85158** | 0.88359 | Complex1D |
| SinFM JNR | **0.39833** | 0.41335 | Complex1D |
| SinFM frequency_deviation | **0.27228** | 0.35501 | Complex1D |
| MFT JNR | 0.43703 | **0.40275** | IQ-Grouped |
| MFT num_targets | 0.83394 | **0.81450** | IQ-Grouped |
| VFT JNR | 0.50886 | **0.47415** | IQ-Grouped |
| VFT Doppler | 0.86513 | **0.85487** | IQ-Grouped |
| ISRJ JNR | 0.47480 | **0.45049** | IQ-Grouped |
| ISRJ sample_width | **0.53071** | 0.58744 | Complex1D |

两者各胜 6/12 个目标。最难目标仍是 SawFM sweep period、MFT target count 和 VFT Doppler；更换通用时域骨干没有解决这些目标的结构性难点。

## 后续建议

1. 默认采用 IQ-Grouped，原因是总体性能不劣且参数量比 Complex1D 少约 21.1%；Complex1D 保留为对照和特定参数专家候选。
2. 下一轮不要继续无差别堆更大骨干。优先在统一 IQ-Grouped 主干上，为 SawFM、MFT、VFT 三个困难目标加入与物理证据匹配的局部序列读出，再逐模块消融。
3. 若重新验证 EfficientNet，应单独测试冻结/替换 BatchNorm 或扩大筛选样本；这属于训练配方适配试验，不能与本轮“原骨干等条件比较”混在一起。
4. 当前结果全部属于开发集验证。只有在结构和超参数冻结后，才可申请一次锁定测试评估。

## 产物

- 实验入口：`cj0724/multitask_benchmark.py`
- 多任务模型与损失：`cj0724/multitask.py`
- 计划：`docs/multitask_backbone_experiment_plan_20260725.md`
- smoke：`outputs/multitask_backbone_smoke/balanced_multitask_20260725_004046`
- 三种子筛选：`outputs/multitask_backbone_screen/balanced_multitask_20260725_004121`
- 全量确认：`outputs/multitask_backbone_full_confirm/balanced_multitask_20260725_004707`
