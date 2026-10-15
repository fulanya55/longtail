# Route A H4：VFT 全延迟-频率相干图 P0 结果

日期：2026-08-03  
决策：**P0 未通过，停止 H4；不训练 M1/M2 图编码器。**

## 协议与边界

H4 是 H3 后唯一允许的新信息路径：不再对每一频率先做 delay-max，而是保留固定
`3×25×401` 的归一化复相干图（实部、虚部、模值）。每个 H4 identity 产生 absent、
weak VFT 8 dB、strong VFT 20 dB 三臂，五个非 VFT 背景、VFT 参数及 receiver
noise 在三臂内完全共享。

正式 P0 有 640 个新 identity triplet（1,920 views），源 SHA-256 为
`a8581453beb7938f31b61a0769b0c872b654608280b954e0dd8f0f64ee3dbea2`。源审计通过
payload、三臂反事实合同、历史训练及 H3 roles 零 receiver overlap，且
`external_used=false`、`locked_test_used=false`、`test_v2_used=false`、
`test_v3_used=false`。

P0 无可训练参数、无模型、无校准、无阈值。固定读出为：对图模值先跨频均值，再取
固定五点 delay 局部质量最大值与总 delay 质量之比。真实时间戳和 H3 冻结置换
时间戳是唯一不同项。

## 结果

| 比较 | 真时间戳 | 置换控制 | 真−置换 |
| --- | ---: | ---: | ---: |
| weak vs absent AUROC | 0.517688 | 0.507979 | +0.009709 |
| weak vs absent AUPR | 0.510842 | 0.505076 | +0.005766 |
| strong vs absent AUROC | 0.553630 | 0.517493 | +0.036138 |
| weak paired present > absent | 0.690625 | — | — |

弱 VFT 的真减置换 AUROC bootstrap 95% CI 为 `[+0.005383,+0.014041]`，虽方向为正，
但绝对弱 AUROC 的 95% CI 仅为 `[0.512087,0.523831]`。这表示该局部聚集读出确实
保留了一点真实时间顺序差异，却没有形成可用于弱 VFT 判别的足够物理证据。

## 预注册门槛

| P0 门槛 | 结果 |
| --- | --- |
| weak 真时间戳 AUROC `>=0.60` | 失败（0.517688） |
| weak 真−置换 AUROC `>=+0.060` | 失败（+0.009709） |
| 真−置换 AUROC CI 下界 `>0` | 通过（+0.005383） |
| 零训练参数与 test-v3 未使用 | 通过 |

总 gate 为 `false`。因此不能把全图输入作为“或许网络能学到”的理由：在进入训练前，
它未提供足够大的、可证伪的弱 VFT 图形物理量。禁止在 H4 source 或重复制源上搜索
delay 核宽、ROI、频率网格、图编码器、损失、JNR、训练长度、校准或阈值；M1/M2
不实现、不训练，也不访问 external/locked/test-v2/test-v3。

## 解释与总体影响

H4 与 H3 合起来排除了两个不同层级：H3 的 frequency-wise delay-max 向量加性投影
没有超过置换容量对照；H4 的完整相干图在固定局部 delay 聚集读出下也未达到最低
物理判别门。前者不能说明真时间戳无信息，后者也不否认 H0q/H1a/H1b 的全图最大
相干物理结果；但两者共同表明，现有复合背景下“将时间顺序改造成 VFT 神经局部
证据”的证据链尚未成立。

这不授权再开一轮 VFT 图网络。下一步应回到总体创新目标的缺口审计：若没有一个
与上述时间戳相干、delay-max/局部聚集及加性投影均不同的、文献支持的可证伪物理
机制，则多脉冲 VFT 只保留为物理可观测性附录；主线证据只保留窄 ISRJ 冻结复现，
并明确完整统一六标签候选仍未晋级。

## 工件

- 准入与停止规则：`docs/reports/route_a_h4_vft_beamspace_map_admission_audit_20260803.md`
- 正式源审计：`outputs/tapc/route_a_h4_vft_beamspace_p0_generation/p0_development_v1_r1_source_audit.json`
- P0 汇总：`outputs/tapc/route_a_h4_vft_beamspace_p0/summary.json`
- 图提取实现：`cjlt/models/vft_staggered_pri.py::staggered_vft_beamspace_maps`
