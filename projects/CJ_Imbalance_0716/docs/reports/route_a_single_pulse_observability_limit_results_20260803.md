# Route A：0716 单脉冲 MFT/VFT 可观测性约束审计结果

日期：2026-08-03  
结论：**原始单脉冲 0716 协议不再允许以扩宽主干、微调 ROI/loss 或阈值的方式继续搜寻 MFT/VFT 物理机制。**

## 证据角色

这是配置级、零训练审计，而不是分类实验。它只读取官方 `create_closedset_6class_config.m` 和 test-v3 的封存状态；未读取 I/Q、标签、参数真值、模型、预测、校准或阈值。

## 已固定的量化结果

| 约束 | 0716 固定配置 | 计算值 | 含义 |
| --- | ---: | ---: | --- |
| 有效 LFM 脉宽 | 100 us | 1,000 点 | MFT/VFT 的有效相干观察窗 |
| MFT 相对时延 | 30–50 us | 最大副本间距 20 us | 任意两个 100 us 脉冲副本至少重叠 80% |
| VFT 多普勒 | 1–7 kHz | 跨度 6 kHz | 仅为一个有效脉冲 10 kHz Fourier bin 的 0.6 |
| VFT 脉冲内相位积累 | 1–7 kHz × 100 us | 0.1–0.7 周 | 未提供跨脉冲相干孔径 |

这些量是必要约束，不是 Bayes 最优错误率。因此它们**不证明**原始 0716 任务不可分类，也不重写任何已有结果；但它们反驳了“仅增加容量即可恢复未观测物理信息”的后续实验假设。

## 决定

- 原始单脉冲 0716 的 MFT/VFT 不再开设容量、ROI、损失、阈值或时间戳重试。
- 如要研究 MFT，必须另行预注册有最小副本间隔与更大 delay aperture 的新采集/仿真协议。
- 如要研究 VFT，必须另行预注册多 PRI 连续相位、明确 coherent aperture 的新协议；此前的 staggered-PRI H0q/H1a/H1b 仅为该新观测协议的物理可辨识性证据，不能挽救其已失败的学习型统一模型。
- 不授权访问 H2 external、历史 locked test、test-v2 或 test-v3。

## 可复核工件

- 预注册：[route_a_single_pulse_observability_limit_audit_20260803.md](../plans/route_a_single_pulse_observability_limit_audit_20260803.md)
- 审计程序：[audit_route_a_single_pulse_observability_limit.py](../../experiments/audit_route_a_single_pulse_observability_limit.py)
- 不可覆盖收据：`outputs/tapc/route_a_single_pulse_observability_limit/summary.json`
- 配置 SHA-256：`9071b243843d30de34bc554004ed4e99d04447fafd3a517ceb328bea21a554ed`

所有收据记录 `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。
