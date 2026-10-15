# 路线 A：DFMS-IQ 延迟融合多尺度复数网络技术方案（2026-07-24）

## 1. 设计目标

上一轮结果表明，修改统计池化和分类器仍不能恢复弱分量，主干网络需要重新设计。本方案将当前 PAM-IQ 的“浅层复数、早期一次性融合、深层实值”改为：

```text
深层复数多尺度编码
  + 延迟 I/Q 融合
  + Cartesian/幅度/相位三条互补融合支路
  + 无注意力全局统计分类头
```

模型名称为 **DFMS-IQ：Delayed-Fusion Multi-Scale Complex IQ Network**。

首轮只验证特征提取能力，不加入标签查询、ECA/PCAF/GC、弱正排序、功率配对或半监督目标，避免再次混杂归因。

## 2. 输入与总体结构

输入保持为 RMS 归一化复数 I/Q：

```text
x ∈ R[B,2,3000]
```

总体路径：

```text
I/Q [B,2,3000]
  → Complex Residual Block C1
  → anti-alias average downsample
  → Complex Residual Block C2
  → Complex Multi-scale Stage C3
  → Complex Multi-scale Stage C4
  → late Cartesian/magnitude/multi-lag-phase fusion
  → real-valued multi-scale refinement
  → unweighted global mean/std
  → 128-dimensional feature
  → Linear(128,6)
```

I/Q 双流一直保留到时序长度 375；所有长感受野复数分支完成后才进行实值融合。

## 3. 深层复数编码器

默认 `width=16`，通道计划为 `16 → 32 → 48 → 64`。

| 阶段 | 输入/输出 | 长度 | 结构 |
|---|---|---:|---|
| C1 | complex 1→16 | 3000 | 两层 complex Conv1d，kernel=7，残差 |
| down1 | complex 16 | 1500 | real/imag 同步 AvgPool1d(2) |
| C2 | complex 16→32 | 1500 | 两层 complex Conv1d，kernel=5，残差 |
| down2+C3 | complex 32→48 | 750 | 三个复数分支 `(k,d)=(3,1),(7,2),(15,4)` |
| down3+C4 | complex 48→64 | 375 | 三个复数分支 `(k,d)=(3,1),(7,2),(15,4)` |

C3 每个分支输出 16 个复数通道，拼接为 48，再使用 complex `1×1` 投影；C4 每个分支输出 24 个复数通道，拼接为 72，再投影为 64。每个 stage 都包含复数残差捷径。

首轮继续使用项目现有的 real/imag 对称 BatchNorm 和激活，避免同时改变融合位置、复数归一化和激活函数。严格相位等变归一化留作后续独立消融。

## 4. 延迟三分支融合

最终复数 token：

```text
R, I ∈ R[B,64,375]
```

### 4.1 Cartesian 支路

直接保留复数坐标信息：

```text
concat(R,I) [B,128,375]
  → Conv1d(128,128,1) + BN + SiLU
  → F_cart [B,128,375]
```

### 4.2 幅度安全支路

提供对相位不稳定更鲁棒的全局证据：

```text
sqrt(R²+I²) [B,64,375]
  → Conv1d(64,128,1) + BN + SiLU
  → F_mag [B,128,375]
```

### 4.3 多滞后相位支路

不再只计算单点相邻相位差，而是在深层复数特征上使用三个滞后：

```text
lags = {1,4,16}
每个 lag 计算 cos(Δφ_lag)、sin(Δφ_lag)
输入维数 = 64 × 2 × 3 = 384
```

相位支路采用低秩投影：

```text
[B,384,375]
  → Conv1d(384,32,1) + BN + SiLU
  → Conv1d(32,128,1) + BN
  → F_phase [B,128,375]
```

三路融合不是一次性 concat 压缩，而是残差相加：

```text
F = SiLU(F_cart + F_mag + tanh(gamma_phase)·F_phase)
```

`gamma_phase` 初始化为 `0.10`，避免噪声主导的相位支路在训练初期覆盖幅度和 Cartesian 主路。

## 5. 融合后时序精炼与分类器

融合后只保留一个参数高效实值多尺度残差块：

```text
128 channels
  → depthwise branches `(k,d)=(3,1),(7,2),(15,4)`
  → concat + 1×1 fusion
  → residual output [B,128,375]
```

首轮不使用注意力池化和标签查询：

```text
unweighted mean + std → [B,256]
Linear(256,128) → LayerNorm → GELU → Dropout(0.20)
Linear(128,6)
```

这样可以直接观察新主干是否提高弱正—负样本可分性，而不是通过降低阈值或增加查询预测数量获得表面召回。

## 6. 参数与训练建议

参数目标：部署参数量控制在 `230k–300k`，与 B1 的 `260,630` 尽量匹配。最终实际参数量由实现后的自动测试记录。

首轮公平筛选建议保持上一轮协议：

| 参数 | 数值 |
|---|---:|
| loss | DB Loss |
| optimizer | AdamW |
| learning rate | 3e-4 |
| weight decay | 1e-4 |
| batch size | 64 |
| epochs | 20 |
| warmup | 10% |
| gradient clip | 5.0 |
| dropout | 0.20 |
| AMP | 开启 |
| selection | macro-AUPR |
| seeds | 2041/2042/2043 |
| train samples | 2400 |
| development evaluation | 每角色最多300 |
| locked test | 禁止读取 |

## 7. 最小结构消融矩阵

| 编号 | 网络 | 目的 |
|---|---|---|
| B1 | WideCapacity/Base-DB | 容量匹配主基线 |
| P0 | 当前 PAM-IQ | 早期融合锚点 |
| F1 | DFMS-IQ，关闭多滞后相位支路 | 验证深层复数+延迟 Cartesian/幅度融合 |
| F2 | 完整 DFMS-IQ | 验证深层多滞后相位是否提供额外收益 |

如果 F1 已提高而 F2 退化，说明延迟融合有效、相位支路仍受噪声影响；如果 F1/F2 均无改善，则问题不能继续归因于 I/Q 融合时机。

## 8. 必须报告的指标

除 macro-AUPR、SawFM-AUPR、macro-F1、Hamming 和弱分量召回外，新增：

1. 弱正—负样本 AUROC/AUPR；
2. 强正、弱正、负样本的分数中位数；
3. 固定负样本 FPR 下的弱正召回；
4. FMB/SawFM/SinFM/MFT/VFT/ISRJ 逐标签弱正—负 AUROC；
5. 冻结编码器后的逐标签两层 MLP probe。

不得再用低阈值下的单独弱召回作为主干改进证据。

## 9. 首轮晋级门槛

完整 DFMS-IQ 相对 B1 同时满足：

1. `Δ macro-AUPR >= -0.005`；
2. `Δ SawFM-AUPR >= 0`；
3. `Δ weak-component recall >= +0.030`；
4. `Δ weak-vs-negative AUROC >= +0.020`；
5. `Δ Hamming <= +0.005`；
6. 至少 2/3 seeds 的弱正—负 AUROC 为正增益；
7. SinFM、MFT、VFT 任一标签 AUPR 下降不超过 0.030。

只有主干通过该门槛，下一阶段才重新接入受约束标签查询和弱正—难负排序损失。

## 10. 实现与验证状态（2026-07-24）

- 已实现 `DelayedFusionMultiScaleIQClassifier`。
- 已注册入口：`dfms_iq` 和 `dfms_iq_no_phase`。
- 实际可训练参数量：`257,511`，相对 B1 的 `260,630` 少 `3,119`，差约 `1.20%`。
- 输入 `[B,2,3000]` 时，深层复数输出为 real/imag `[B,64,375]`。
- 延迟融合及实值精炼输出为 `[B,128,375]`，最终特征为 `[B,128]`。
- PY10/RTX 5080 GPU 前向和反向传播通过，输入梯度与相位残差梯度均为有限值。
- 全量单元测试：`147 passed, 36 warnings`。
- 当前只完成网络实现与工程验证，尚未将 smoke 或随机输入结果表述为识别性能。
