# P3o-P：复相干稀疏事件 MFT count 实验

## 结论

P3o-P 未通过预注册的严格三种子门槛，结论为 `stop_no_confirmation_data`。复相干特征有一致的平均 MAE 和 ultra-close 改善迹象，但不足以支持可靠的 MFT count 专家或确认集生成。

## 机制与对照

MFT 生成器将实数正幅度的同一复 LFM 参考信号循环移位后相加。因此，候选机制不是调峰规则：它在 P3o-M 冻结的同一 16 个严格局部 contrast 峰处保留反卷积复响应的相干信息。

- C0：四个幅度/对比度/尖锐度/位置 token 加四个零通道；
- C1：相同四个 token 加归一化 in-phase、quadrature、五样点 phasor coherence 和五样点 in-phase coherence；
- C0/C1 都是相同的 146,983 参数 ordinal CNN，分类损失关闭，MFT oracle-positive only。

训练/开发 identity 为 1800M/1810M，各 2,079 个；跨角色重叠为 0，与 402 份历史身份文件最大重叠为 0，三功率视图严格配对。未生成、打开或使用 test/locked test。

## 结果

| seed | C1 相对 C0 总体 MAE 改善 | 复合 MAE 改善 | 总体 Pearson 变化 | low-count C1 Pearson | ultra-close MAE 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2321 | +4.14% | +4.12% | +0.0138 | 0.2914 | -0.0682 |
| 2322 | +4.35% | +4.31% | +0.0180 | 0.2765 | -0.0892 |
| 2323 | +2.69% | +2.71% | +0.0090 | 0.3010 | -0.0631 |

三个 seed 的平均改善方向一致，也在 ultra-close MAE 上同向改善；但每个 seed 均未达到至少 5% overall/compound MAE 和 +0.03 Pearson 门槛，前两个 seed 的 low-count Pearson 小于 0.30，且至少一个 regime 在三个 seed 中均比 C0 恶化超过 2%。

## 运行合同修复

首次运行在模型优化前发现复相干前端错误地接收了 P3o-M 已压缩的幅度通道；第二次运行在 C1 前发现标准化位置未反变换为原始索引。两次中止输出均保留且不作为结果。修复后使用原始 I/Q、审计一致的样本索引，以及测试覆盖的反标准化位置映射，在 `contractfix2` 目录从头完成 C0/C1 三种子训练。

## 决策

不得在 1800M/1810M 上调整相干描述符、峰规则、网络、损失或训练；不得生成 1820M/1830M。后续若继续 MFT count，必须不是对相干值、局部峰几何或拒识门槛的微调，而应提出新的样本级监督或物理可辨识性数据合同，并先隔离其是否改变信息上限。

关键工件：

- `configs/p3op_complex_coherence_preregister_20260803.json`
- `configs/p3op_complex_coherence_method_freeze_20260803.json`
- `configs/p3op_complex_coherence_outcome_freeze_20260803.json`
- `outputs/p3op_complex_coherence_preflight_20260803/data_audit.json`
- `outputs/p3op_complex_coherence_development_20260803_contractfix2/summary.json`
