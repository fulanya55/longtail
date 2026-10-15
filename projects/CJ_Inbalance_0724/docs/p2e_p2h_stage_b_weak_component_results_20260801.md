# P2e–P2h 弱分量与复杂组合验证报告

## 结论

Stage B 已完成一次失败确认、三轮身份隔离开发和一次全新独立确认。最终 P2h 方法取得稳定的识别、参数和端到端收益，但没有通过“最坏组合严格不劣”门槛，因此不得表述为 Stage B 全面验证通过。

最终独立 confirmation 的主要结果为：

- 三个目标的最大条件 FPR：`0.070161`，通过 `<=0.10`；
- ISRJ Recall 相对同 FPR 主干平均提升：`0.085330`，三个种子分别为 `0.114583/0.072135/0.069271`；
- SawFM period 参数 predicted-gate NMAE 平均改善：`5.888%`，三个种子同方向；
- VFT Doppler 参数 predicted-gate NMAE 平均改善：`5.463%`，三个种子同方向；
- SawFM/VFT/ISRJ miss-penalty NMAE 均优于同 FPR 主干；
- 唯一失败：SawFM 最坏组合在种子 `2042` 上增加 `0.009677` NMAE，三种子平均仍略大于零。

因此，Stage B 的证据级别是：**强部分支持，但保留一个高阶组合尾部风险；`stage_b_validated=false`。**

## 数据与证据边界

所有数据均覆盖 63 个非空标签组合和 1–6 个分量规模，并对 SawFM、VFT、ISRJ 分别设置 `-5/-10/-15 dB` 相对弱化。

| 角色 | identity seed | 每条件身份数 | 用途 |
|---|---:|---:|---|
| P2e independent failure | 1500000000 | 2520 | 冻结 P2c 的首次弱分量检验 |
| P2f development-v1 | 1600000000 | 1890 | 发现阈值漂移与复杂度负迁移 |
| P2g development-v2 | 1650000000 | 1890 | 检验 CP 校准和保守门控 |
| P2h development-v3 | 1675000000 | 1890 | 最终方法拟合与晋级 |
| P2h independent confirmation | 1700000000 | 2520 | 冻结方法一次性确认 |

最终 confirmation 与原始数据、P1d、P1n、P2a、P2b、P2c、P2e 和三轮 development 身份重叠均为 `0`。九个条件共享相同有序 `receiver_noise_seed` 和标签，因此它们被视为同一身份的配对反事实视图，而不是九倍独立样本。

`locked_test_used=false`。本阶段没有访问 locked test。

## P2e 首次冻结失败

原始 P2c 方法直接迁移到弱分量复杂组合时：

- ISRJ Recall 平均提升 `9.878%`，95% CI `[9.216%, 10.553%]`；
- VFT 参数 NMAE 改善 `15.331%`；
- SawFM 参数 NMAE 仅改善 `1.542%`；
- 最大 FPR 达 `20.565%`；
- VFT/ISRJ miss-penalty 变差；
- 最坏组合门槛失败。

失败不是单纯的网络容量问题：SawFM 专家在低复杂度组合有效，在 4–6 分量出现负迁移；旧分类阈值在“非目标强分量均为 30 dB”的组合分布上漂移；保守分类阈值又降低了弱目标覆盖率。

## P2h 冻结方法

最终方法没有重新训练主干、UNet 或物理专家，只增加可观测、可审计的安全机制：

1. 使用 threshold-calibration 身份拟合一侧 `99%` Clopper–Pearson FPR 上界不超过 `0.08` 的目标阈值；
2. 参数专家只允许在预测分量复杂度 `0–3` 的候选范围内调用；
3. 使用专家与主干参数输出分歧作为可靠性条件；
4. 使用预测标签 bitmask 白名单：模式在 rule-fit 中至少有 10 个样本、总体参数 NMAE 至少改善 5%，并在三个弱化条件和三个身份折中均不退化；
5. 未进入白名单的样本回退到 MLCNN 主干参数输出；
6. threshold-calibration、rule-fit、development-validation 三种身份角色互斥。

该方法在 development-v3 上通过全部开发晋级门槛：最大 FPR `9.441%`，ISRJ Recall `+8.735%`，SawFM/VFT 参数改善 `+6.573%/+5.001%`。

## 最终独立 confirmation

### 配对 bootstrap

| 比较 | 平均差值 | 95% CI | 方向 |
|---|---:|---:|---|
| ISRJ Recall | `+0.085330` | `[+0.078400, +0.092257]` | 通过 |
| SawFM period NMAE | `-0.051228` | `[-0.060840, -0.041891]` | 通过 |
| VFT Doppler NMAE | `-0.048825` | `[-0.058437, -0.039237]` | 通过 |
| SawFM miss-penalty NMAE | `-0.013703` | `[-0.016230, -0.011118]` | 通过 |
| VFT miss-penalty NMAE | `-0.012756` | `[-0.015322, -0.010308]` | 通过 |
| ISRJ miss-penalty NMAE | `-0.024066` | `[-0.027203, -0.021013]` | 通过 |

每项使用 2000 次 receiver-noise-seed 分组 bootstrap。识别收益要求 CI 下界大于零；误差收益要求 CI 上界小于零，全部满足。

### 最坏组合失败边界

SawFM 三个种子的最坏组合差值为：

- seed 2041：`0.000000`；
- seed 2042：`+0.009677`；
- seed 2043：`0.000000`。

seed 2042 的退化来自少量被预测为可靠模式、但实际上仍含未识别共现分量的样本。专家只在相关真实组合中的极少数样本上被调用，但足以使严格最大值略高于主干。这说明“预测标签模式”仍不能完全观测隐藏干扰复杂度。

## 创新点当前可支持的表述

可以支持：

- 物理序列专家并非无条件附加，而是通过有限样本安全校准、预测组合条件和可靠性回退形成端到端联合感知；
- 在全新身份、弱分量、全部 1–6 分量规模下，ISRJ 识别、SawFM period、VFT Doppler 和 miss-penalty 均获得统计显著收益；
- 相对同 FPR 主干比较避免了用高 FPR 换取 Recall 的不公平结论。

暂不能支持：

- “所有复合组合均不退化”；
- “Stage B 全面通过”；
- “已达到 Level-4，可申请 locked test”。

## 后续门槛

Stage B confirmation 已封存，不允许在 seed 1700000000 上重新选择阈值、白名单或专家门控。若后续要消除最坏组合风险，应在新 development 身份上建立能感知隐藏共分量的干扰/分离质量指标，并使用另一套新 confirmation 身份验证。

当前先进入正交的 Stage C 参数 OOD 验证。即使 Stage C 通过，Level-4 仍需保留 Stage B 最坏组合失败，直至新的独立验证消除该风险。

## 主要工件

- 首次独立失败：`outputs/p2e_stage_b_weak_generalization_20260801`
- 最终开发冻结：`outputs/p2h_composition_safe_development_v3_20260801`
- 最终独立确认：`outputs/p2h_composition_safe_confirmation_20260801`
- confirmation 冻结预测：`outputs/p2h_stage_b_confirmation_predictions_20260801`
- 实现：`cj0724/p2f_composition_safe_development.py`
- confirmation：`cj0724/p2h_composition_safe_confirmation.py`
- 数据生成：`matlab/generate_p2e_stage_b_weak_components.m`
