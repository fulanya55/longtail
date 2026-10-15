# P1h：参考对齐 VFT Doppler 物理专家确认试验

日期：2026-07-31  
证据等级：三个全新生成 seed 的 identity-safe development-confirmation  
结论：VFT Doppler 参数专家通过准入门槛；尚未接回端到端预测标签门控系统。

## 1. 最近三轮实验

| 轮次 | 研究问题 | 主要结果 | 成功与失败 |
|---|---|---:|---|
| P1f | 置信下界路由能否优于均值路由 | lower route Recall `+0.07435`，最坏 FPR `0.04274` | 实用指标安全，但未达到 `+0.10` Recall，且 lower 没有独立安全优势 |
| P1g | 逐功率有限样本缓冲能否优于 pooled-FPR | Recall `+0.07656`；worst-FPR `0.11613→0.09274` | Level 1 局部风险校准机制通过；仍只覆盖 ISRJ 路由和 20/25 dB |
| P1h | 正确 LFM 参考和到达时延对齐能否恢复 VFT Doppler | MAE `821.38 Hz`，相对常数基线改善 `46.49%`，Pearson `0.478` | VFT 参数从旧专家失败转为通过；复杂组合仍有局部失败，不能无条件接回 |

## 2. 问题与失败根因

旧 `vft_instantaneous_frequency` 专家直接把原始复合波形的相位增量、局部幅度和幅度变化输入一维 CNN。三 seed 的结果为：

- 相对常数基线 MAE 改善：`-1.36%`；
- Pearson：`-0.015`；
- 0/3 seed 获得正 MAE 改善。

VFT 信号由已知 LFM 参考乘以 1–7 kHz Doppler 相位斜率后，再循环移动 100–500 个采样点。旧表示没有消除 LFM 二次相位，也没有估计到达时延。1–7 kHz 相对于 10 MHz 采样率非常小，直接相位增量主要被 LFM 本体和复合分量支配。

P1h 的可证伪假设是：

> 如果 Doppler 信息确实存在但被参考相位和到达时延遮蔽，那么正确参考对齐、时延搜索和 dechirp 后的局部 delay-Doppler 模糊度峰应通过专家准入门槛；错误参考和不做时延对齐应失败。

## 3. 冻结方法

### 3.1 数据

- 数据 seeds：`500000000 / 510000000 / 520000000`。
- 每个 seed 包含 20 dB、25 dB 两个配对功率视图。
- 每个 split、每个功率：63 个组合 × 20 个身份 = 1260。
- 每个功率 VFT-positive：640；两功率测试评价共 1280 个正视图。
- 三组 train/test 内部身份重叠均为 0；三个 seed 之间任意 train/test 交叉重叠均为 0。
- 只评价 `label[VFT]=1 AND parameter_mask[Doppler]=1 AND finite(target)`。
- train 只用于计算常数中位数基线；物理专家没有使用测试标签训练或校准。

### 3.2 物理专家

1. 按雷达参数构造 10 MHz、100 us、5 MHz 带宽的已知 LFM 参考。
2. 使用循环互相关在 100–500 样点范围搜索到达时延。
3. 保留相关最高的 8 个时延候选。
4. 每个候选与正确 LFM 共轭相乘完成 dechirp。
5. 在 1–7 kHz、100 Hz 步长的局部频率网格计算相干模糊度分数。
6. 最大 delay-Doppler 峰对应的频率作为 Doppler 预测。

该专家是确定性物理算子，没有神经参数、训练损失、学习率或 weight decay。

### 3.3 机制控制

- 正确参考：`K=5e10 Hz/s`；
- 错误参考 0.8 倍：`K=4e10 Hz/s`；
- 错误参考 1.2 倍：`K=6e10 Hz/s`；
- 无时延对齐：固定 delay=0。

冻结哈希：`0d4f454ff0dd417bce93ac71216ccf207d94516c0172e6f8d0b739cba4fb9128`。  
执行代码 SHA-256：`bf358d03672c99a462fe47a128c3c9c0bc764196670da9b55a25bca3e7500ba4`。

## 4. 冻结门槛

1. 平均 MAE 相对 train 中位数基线至少降低 10%；
2. 平均 Pearson 至少 0.3；
3. 3/3 独立数据 seed 的 MAE 与相关性方向均为正；
4. 复合样本子集平均 MAE 至少降低 10%；
5. 复合样本子集平均 Pearson 至少 0.3；
6. 正确参考相对最强机制控制的 MAE 至少改善 20%；
7. 正确参考在 3/3 seed 上均优于所有控制。

## 5. 正式结果

### 5.1 三 seed 聚合

| 指标 | 正确参考结果 | 门槛 | 判定 |
|---|---:|---:|---|
| MAE | `821.38 Hz` | — | — |
| 相对常数基线 MAE 改善 | `46.49%` | `>=10%` | 通过 |
| Pearson | `0.478` | `>=0.3` | 通过 |
| 三 seed MAE 改善 | `51.22% / 44.99% / 43.26%` | 3/3 正向 | 通过 |
| 三 seed Pearson | `0.525 / 0.466 / 0.441` | 3/3 正向 | 通过 |
| 复合样本 MAE 改善 | `44.74%` | `>=10%` | 通过 |
| 复合样本 Pearson | `0.460` | `>=0.3` | 通过 |
| 最强错误控制 MAE | `2279.48 Hz` | — | — |
| 正确参考相对控制 MAE 优势 | `63.97%` | `>=20%` | 通过 |

7/7 冻结门槛全部通过，`innovation_validated=true`。

### 5.2 功率结果

| 数据 seed | 20 dB MAE / 改善 / r | 25 dB MAE / 改善 / r |
|---:|---|---|
| 500000000 | `835.16 / 45.19% / 0.479` | `651.27 / 57.26% / 0.573` |
| 510000000 | `965.08 / 36.34% / 0.390` | `702.86 / 53.64% / 0.544` |
| 520000000 | `986.06 / 36.92% / 0.390` | `787.83 / 49.60% / 0.493` |

两个功率、三个数据 seed 全部保持正 MAE 改善和正相关。25 dB 一致优于 20 dB，符合更高 VFT 可见性带来更可靠模糊度峰的物理预期。

### 5.3 机制归因

错误参考和无时延对齐的 MAE 大约为 2.26–2.73 kHz，相关性接近 0，且相对常数基线普遍恶化。由此可以排除“频率网格或输出范围本身造成虚假相关”的解释。

真正有效的是正确参考相位消除与到达时延对齐的组合，而不是继续增加通用 CNN 容量。

## 6. 逐组合失败边界

纯 VFT 组合在三个 seed 上的 MAE 仅为约 `27.16 / 28.56 / 19.57 Hz`，Pearson 均接近 1。但部分复杂组合仍可能失败：

- seed 500000000 的 `FMB+SawFM+VFT`：MAE 改善 `-2.82%`；
- seed 510000000 的 `FMB+MFT+VFT`：MAE 改善 `-8.76%`；
- seed 520000000 的 `FMB+MFT+VFT`：改善仅 `4.43%`。

这些组合中的 FMB 宽带能量、SawFM 调频结构和 MFT 多个参考副本会产生竞争模糊度峰。P1h 证明 VFT Doppler 已总体可辨，但尚未证明所有组合均可安全无条件替换。

## 7. 对创新层级的影响

| 层级 | 当前状态 | 证据 |
|---|---|---|
| Level 1 局部机制 | 已完成两个 | P1g 逐功率风险缓冲；P1h 参考对齐 Doppler |
| Level 2 参数专家 | 聚合准入门槛已达到 | SawFM period 旧开发证据通过；VFT Doppler P1h 新独立数据通过 |
| Level 3 端到端系统 | 未达到 | VFT 专家尚未通过预测标签和可靠性门控接入参数系统 |
| Level 4 泛化 | 未达到 | 仅验证 20/25 dB；参数分布仍为 1–7 kHz |
| Level 5 最终证据 | 未开始 | locked test 未使用 |

Level 2 的表述必须保留边界：SawFM 与 VFT 两个专家已分别通过参数准入门槛，但 VFT 的新证据尚未接入端到端系统，MFT count 仍失败。

## 8. 下一轮最小实验：P1i

下一轮只验证一个问题：

> 使用与标签无关的模糊度可靠性指标进行 VFT 参数专家门控，能否相对无条件替换和 IQ-Grouped 参数头，在预测 VFT 标签门控下改善 Doppler NMAE，同时避免困难组合退化？

建议冻结三组：

1. IQ-Grouped 原参数头；
2. 预测 VFT 阳性后无条件使用 P1h 专家；
3. 预测 VFT 阳性且模糊度可靠性通过时才使用 P1h 专家，否则保留 IQ-Grouped。

主要门槛：预测门控 Doppler NMAE 至少改善 10%、漏检惩罚不恶化、3/3 seeds 同方向、最坏组合 NMAE 不劣于无条件替换、分类输出完全不变。可靠性规则只能在新的 train/selection/calibration 角色上确定，P1h test 不得用于选择阈值。

## 9. 产物与证据边界

- 实现：`cj0724/vft_doppler_aligned_expert.py`
- 测试：`tests/test_vft_doppler_aligned_expert.py`
- 数据生成：`matlab/generate_p1h_vft_doppler_confirmation_data.m`
- 方法冻结：`outputs/p1h_vft_doppler_aligned_confirmation_20260731/method_freeze.json`
- 聚合结果：`outputs/p1h_vft_doppler_aligned_confirmation_20260731/summary.json`
- 逐样本结果：`outputs/p1h_vft_doppler_aligned_confirmation_20260731/data_seed_*/test_per_sample.npz`
- 身份审计：`outputs/p1h_vft_doppler_aligned_confirmation_20260731/identity_audit.json`
- 全量测试：96 passed。

本轮 `test_used_once=true`、`test_fit_or_selection=false`、`confirm_used=false`、`locked_test_used=false`。P1h test 不得再用于任何门控阈值或模型选择。
