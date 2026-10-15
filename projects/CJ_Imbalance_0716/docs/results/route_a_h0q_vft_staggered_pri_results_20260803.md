# Route A H0q：交错 PRI VFT 慢时间相位一致性结果

## 结论

**H0q 通过。** 在全新、identity-paired 的八脉冲交错 PRI VFT 物理源上，真实绝对时间戳下的固定 Doppler 相位一致性显著优于保持同一复匹配系数但置换时间戳的控制。它证明新的非均匀慢时间观测协议为 VFT 提供了可部署的物理信息。

这不是原 0716 单脉冲 VFT 的修复，也不是六标签分类、模型训练或最终 test-v3 结果。它只授权下一步设计另一组新 identities 的复合 VFT present/absent H1。

## 冻结协议

- 新源：`C:\LJQ_code\干扰数据\h0q_vft_staggered_pri_20260803`；low/high 各 640 条；
- 8 脉冲、100 us LFM、10 MHz 采样；交错脉冲间隔 `[300,331,300,331,300,331,300] us`，记录长度 24,930 点；
- receiver identities `2798677742--2798678381`，与 155,515 个历史 train identities 零交集；
- 每对共享 noise、JNR、arrival、Doppler 随机分位和时间轴；唯一干预为 low `1.0--2.5 kHz` 与 high `5.5--7.0 kHz`；
- 固定候选：8 个正确 LFM delay 复系数在真实交错时间戳上的 `1000:250:7000 Hz` 相干扫描，high-grid 最大值减 low-grid 最大值；
- 固定控制：同一系数、网格、delay window 与计算预算，仅把时间戳置换为 `[0,5,2,7,4,1,6,3]`；
- 没有可训练参数、优化器、校准、分类阈值或模型融合。

## 正式结果（5,000 次逐对 bootstrap）

| 指标 | 数值 |
| --- | ---: |
| candidate AUROC | `0.983069` |
| timestamp-permuted control AUROC | `0.335610` |
| AUROC 差 | `+0.647458`，95% CI `[+0.622980,+0.672257]` |
| candidate AUPR | `0.983616` |
| control AUPR | `0.393809` |
| AUPR 差 | `+0.589808`，95% CI `[+0.575519,+0.600384]` |
| paired high > low 比例 | `0.996875` |

所有预注册门均通过：candidate AUROC `>=0.80`、两项 candidate-control 效应及 CI 门、`>=0.75` 的逐对方向门、零拟合和 test-v3 sealed 门。

## 边界与后续门

H0q 不可用于下列任何表述：

- 不能声称原 0716 100 us 单脉冲 VFT 已可稳定识别；
- 不能声称复合背景中弱 VFT 已改善；
- 不能声称 H0q 的经典相干扫描本身是创新网络；
- 不能接入原单脉冲 C3/C4，不能访问 test-v3。

下一步 H1 必须重新生成全新 identity 的交错-PRI 复合 present/absent 源，以 VFT 存在与完全相同的五类背景为任务；候选需使用预先固定的真实时间戳相干 evidence，控制只置换时间戳，并接受 composition-matched 弱正/负样本的 paired CI 和安全门。

## 工件

- 计划：`docs/plans/route_a_h0q_vft_staggered_pri_20260803.md`；
- 生成器：`experiments/generate_route_a_h0q_vft_staggered_pri.m`；
- 来源审计：`outputs/tapc/route_a_h0q_vft_staggered_pri_generation/paired_identity_audit.json`；
- 正式摘要：`outputs/tapc/route_a_h0q_vft_staggered_pri/route_a_h0q_vft_staggered_pri_summary.json`；
- 读出：`cjlt/models/vft_staggered_pri.py`；
- 测试：`tests/test_vft_staggered_pri.py`（`3 passed`）。

所有工件记录 `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。

