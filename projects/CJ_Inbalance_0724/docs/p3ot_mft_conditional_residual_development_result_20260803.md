# P3o-T 开发集结果：可部署 MFT 延迟支持条件 ISRJ 残差

## 结论与证据边界

**结论：失败，停止；不生成确认集，也未读取或使用 locked test。**

P3o-T 在新的、按 `receiver_noise_seed` 隔离的 train / selection / development 三角色上完成了全部冻结试验。其目标是以可观测的复 LFM 逆延迟支持摘要，修正冻结 MLCNN 将 MFT 误报为 ISRJ 的倾向。虽然 T2 相对原始基线 T0 的源条件 ISRJ FPR 在三 seed 中均降低，幅度不足预注册门限；而且零支持残差 T1 取得了几乎相同的效果，相位扰乱控制 T3 也与 T2 接近。因此，不能把改进归因于 MFT 延迟支持，也不能作为新的物理机制或部署路线推进。

此文档仅报告独立开发证据，不能替代确认或 locked-test 结论。

## 冻结设计与数据审计

- 冻结分类骨干：`mlcnn_iq_grouped_task_routed`，基线 seed 为 2041 / 2042 / 2043；残差 seed 为 2401 / 2402 / 2403。
- 条件：T0 原冻结基线；T1 零支持等容量残差；T2 固定复 LFM 逆延迟支持残差；T3 相位扰乱、等维度支持控制。
- T2/T3 只修改 ISRJ logit。烟雾测试确认其余五个 logits 位级不变，MFT AP 的三 seed 差均为 0。
- 训练：AdamW，lr `3e-4`，weight decay `1e-4`，15 epoch，2 epoch warmup + cosine，batch 128，梯度裁剪 5，条件惩罚权重 0.25；每个残差仅按 selection ISRJ BCE 选 checkpoint。
- 数据：每角色 15 视图（5 工况 × 25/35/50 dB），每视图 2,079 样本。45 原始视图及 float32 缓存逐字段一致；角色之间身份交集为 0；与 3,011 个历史文件的最大身份交集为 0。
- 统计：以 `receiver_noise_seed` 为 cluster 的 2,000 次 paired bootstrap；源条件为 MFT=1 且 ISRJ=0。
- `test_generated=false`，`test_opened=false`，`locked_test_used=false`。

## 主要结果（冻结安全阈值）

| base / residual seed | T2-T0 源 FPR 下降 | 95% CI | T2-T1 | T2-T3 | T2-T0 ISRJ recall 变化 |
| --- | ---: | --- | ---: | ---: | ---: |
| 2041 / 2401 | +0.01250 | [0.00808, 0.01717] | +0.00076 | +0.00013 | -0.00227 |
| 2042 / 2402 | +0.01768 | [0.01263, 0.02323] | -0.00126 | -0.00088 | -0.00941 |
| 2043 / 2403 | +0.02045 | [0.01503, 0.02626] | -0.00101 | +0.00379 | -0.00505 |

相对 T0 的 T2 源 FPR 为：

| seed | T0 | T1 零支持 | T2 延迟支持 | T3 相位控制 |
| --- | ---: | ---: | ---: | ---: |
| 2041 | 0.08018 | 0.06843 | 0.06768 | 0.06780 |
| 2042 | 0.09003 | 0.07109 | 0.07235 | 0.07146 |
| 2043 | 0.10581 | 0.08434 | 0.08535 | 0.08914 |

## 门控结果

通过：身份隔离；召回跌幅不超过 0.02；最坏源单元不变差；MFT AP 不变；macro-AUPR / macro-F1 安全；预测门控 NMAE 比例不超过 1.01；最坏组合 NMAE 不变差；至少四工况与 25 dB 方向非负；locked-test 安全标记。

失败：

1. T2 相对 T0 的源 FPR 降幅未在三 seed 都达到 0.02（前两 seed 为 0.01250、0.01768）。
2. T2 相对 T1 与 T3 均未在三 seed 都达到 0.01，且 seed 2042/2043 出现相对 T1 的微弱反向差异。
3. T2-T1 和 T2-T3 的部分 paired-bootstrap 95% 下界不为正。
4. 预测标签门控的 miss-penalty NMAE 在每个 seed 均轻微上升：2041 为 1.30870→1.30964，2042 为 1.30323→1.30324，2043 为 1.30970→1.31010。

## 失败原因分析

1. **物理摘要没有带来可辨识增益。** T1 没有输入任何延迟支持，却几乎复制甚至超过 T2 的 FPR 降幅；T3 也接近 T2。这排除了“固定 32 维归一化平均/最大池化延迟支持是主要原因”的解释。
2. **残差容量与冻结全局特征已足以做通用校准。** 三分支共享冻结 MLCNN 特征和预测 MFT 门，T1 的收益表明它主要学习了对基础 ISRJ logit 的条件性校准，而非从支持图中识别 MFT 结构。
3. **支持压缩可能丢失区分信息。** 支持幅度按每样本最大值归一化并池化为 16 个 avg/max bin，抹去了绝对峰高、峰宽、局部峰顺序、峰间距与跨功率稳定性；但这只是后续假设，不能以同一身份再次调池化或超参数来验证。
4. **参数门控仍是系统级约束。** 虽然总体 predicted-gate NMAE 和最坏组合未恶化，漏检相关的 miss penalty 变差，说明单独改变 ISRJ 决策仍可能改变下游参数样本构成，不能仅凭分类 FPR 推进。

## 冻结后的下一步

不在本批身份上对支持窗口、维度、损失、门限或训练时长继续搜索。若继续，必须使用新的身份隔离 train / selection / development 数据，并采用不同机制：以**显式、可审计的延迟拓扑统计**（峰位置、相邻间距、峰宽、峰高比与跨功率一致性）为输入的受限单调校准器，而不是再训练一个共享全局特征的自由残差 MLP。新路线须包含：无拓扑统计等容量控制、拓扑置乱控制、特征消融、相对 T0/T1/T3 的预注册门与预测门控参数安全门；只有物理统计在三 seed 中带来超出容量控制的增益，才进入确认集。

## 可复核产物

- 预注册与哈希：`configs/p3ot_mft_conditional_residual_preregister_20260803.json`、`configs/p3ot_mft_conditional_residual_method_freeze_20260803.json`
- 原始审计：`outputs/p3ot_mft_conditional_residual_preflight_20260803/data_audit.json`
- 逐 seed 结果：`outputs/p3ot_mft_conditional_residual_development_20260803/base2041_result.json`、`base2042_result.json`、`base2043_result.json`
- 总结与预测：`outputs/p3ot_mft_conditional_residual_development_20260803/summary.json`、`development_predictions.npz`
