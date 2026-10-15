# Route A：单脉冲 MFT/VFT 可观测性约束审计

日期：2026-08-03  
状态：**预注册；仅配置审计，不训练模型。**

## 单一可证伪问题

0716 的固定单 PRI、100 us 有效 LFM 脉冲协议，是否已在配置层面使 MFT 的多副本分离与 VFT 的多普勒相干孔径受到明确限制？

## 固定输入与禁止项

- 唯一科学输入：`create_closedset_6class_config.m`，必须记录 SHA-256。
- 仅检查 test-v3 的封存 manifest 状态；不得打开其 I/Q 或标签。
- 不读取任何 MAT/HDF5 I/Q、标签、参数真值、预测、checkpoint、校准或阈值文件。
- 无网络、分类头、损失、优化器、训练轮数、阈值或统计显著性检验。

## 固定计算

1. MFT：以 `1 - (delay_max-delay_min)/pulse_width` 计算任意两个副本的脉冲支撑重叠下界。
2. VFT：以有效 LFM 脉冲而非整个 PRI 的孔径，计算 `doppler_span/(1/pulse_width)` 与端点相位累积周期数。

## 判定与边界

若 MFT 下界不低于 0.80、VFT 频移跨度少于一个有效脉冲 Fourier bin 且最大相位累积不超过一个周期，则禁止把继续扩宽单脉冲主干、调整 ROI/loss/阈值表述为新的物理机制。

该审计不证明原任务不可分类，也不构成任何模型有效性、泛化或 test-v3 结论。它只规定：后续 MFT 需重新定义 delay-aperture/minimum-separation，VFT 需重新定义 multi-PRI coherent aperture，并与原 0716/test-v3 完全隔离。
