# Route A H0r：MFT 高阶相位耦合候选准入审计

日期：2026-08-03  
状态：**不准入数据生成或训练。**

## 被考虑的单一假设

对 dechirped LFM 的多延迟副本，三频相位耦合（bispectrum/bicoherence）理论上包含
`X(f1) X(f2) X*(f1+f2)` 的三阶项；它不同于已停止的二阶 matched-delay support、
OMP、非零滞后自相关和 STFT 形态读出。一个合格的 H0r 必须证明：在保持单频幅度
谱和二阶相关量的 phase-surrogate 对照下，三阶耦合仍能识别 MFT 副本变化。

## 外部材料的当前等级

Crossref 可核对以下主题存在，但未获得可逐节验证、能直接规定本项目统计量和控制
的原始全文：

- higher-order cyclic cumulant 的相干源处理：DOI `10.1109/pimrc.2003.1259168`；
- LFM repeater jamming：DOI `10.3724/sp.j.1146.2012.01695`；
- complex-scene ISRJ perception：DOI `10.1109/radar53847.2021.10028548`。

这些雷达条目均为 `[T]` 书目主题证据，不得被解释为“文献证明双谱能解决本项目
MFT 识别”。

另已取得可访问的原始信号处理材料：Pinilla, Mishra and Sadler, *Unique Bispectrum
Inversion for Signals with Finite Spectral/Temporal Support*, ICASSP 2023,
arXiv:2111.06479。其摘要明确将 bispectrum 定义为 triple-correlation spectrum，
并针对有限谱/时域支撑讨论由 bispectrum 恢复信号。这支持“二阶谱外的三阶相位信息”
这一数学事实 `[F]`，但不支持任何雷达/MFT 性能结论。

## 与已停止机制的重叠检查

| 家族 | 与候选关系 | 结论 |
| --- | --- | --- |
| matched-filter delay support / OMP | 二阶、参考对齐的 delay 可观测量 | 数学上不等价，但必须保留正确参考/错误参考控制 |
| H0p autocorrelation complexity | 二阶 pairwise delay/lag 量 | 三阶相位耦合不由二阶谱唯一确定；但可能仍在复合背景中无归因性 |
| blind STFT morphology | 幅度/连通性容量读出 | 若直接向 CNN 输入双谱图会成为禁止的通用容量扩展 |
| H3/H4 timestamp VFT | 跨脉冲时间戳相干 | 目标与协议不同，不能作为 H0r 成功证据 |

## 二阶匹配控制合同

对固定长度记录 `x`，令 `X[k]=FFT(x)[k]`，控制为
`X_c[k]=|X[k]| exp(j phi[k])`，其中 `phi` 为预注册、共轭对称的随机相位，随后
`x_c=IFFT(X_c)`。它严格保持每个 DFT bin 的幅度与总能量；由离散 Wiener--Khinchin
关系，保持圆周二阶自相关。普通的三阶项
`X[k1]X[k2]X*(k1+k2)` 则随独立相位改变而被破坏。H0r 必须实际报告幅度谱误差、
圆周自相关误差和 candidate-control 的 bicoherence 差，不能只假定此合同成立。

## 剩余准入条件

H0r 尚不满足下列必要条件：

1. H0r 统计的样本、窗、归一化和有限样本限制必须在 source 生成前固定；
2. 必须实施上述保持能量、单频幅度谱和圆周二阶相关的 DFT 相位 surrogate；
3. 必须先报告 surrogate 的二阶匹配误差，才能把 candidate-control 差归因于三阶项；
4. 统计量、delay/frequency grid 与阈值必须在 source 生成前固定；
5. H0 必须有绝对 MFT 可辨识门与 candidate-control 门，H1 才可在新 receiver
   identities 的复合 matched-negative 合同中开始。

H0r 现**仅获准**实现该固定 no-fit surrogate 合同和单标签 source smoke；不写网络，
不访问 dev/test，不以“高阶统计”作为新的创新名称。正式 H0 必须先通过控制误差
审计，才可生成。

## 当前意义

这是一条潜在的、与既有二阶 MFT 统计不同的**候选方向**，不是已验证假设。它不能
改变 H0p/H1k/H1m 的负结论，也不能为完整 VTSE-VBF 或 test-v3 提供任何晋级理由。
