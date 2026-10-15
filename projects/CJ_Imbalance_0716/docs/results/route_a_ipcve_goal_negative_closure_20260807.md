# Route A IP-CVE 自主实验 Goal：负结论收口

日期：2026-08-07  
最终状态：`mechanism_reliably_denied_research_conclusion_complete`

## 最终结论

本 Goal 已按预注册失败链完成，而不是因训练中断或预算不足停止：

1. Stage -1 完成历史重叠、B0、生成器和 test-v3 seal 审计；
2. Stage 0 完成 2,400 个新 identity 三元组、7,200 views 的严格反事实数据契约；
3. P0 使用五个正式 B0 checkpoint 完成冻结表征与信息破坏控制；
4. P0 三目标全部未通过，统一 IP-CVE 的本地单视图假设被否定；
5. P0 后扩展观测审计确认，当前可执行的 MFT、VFT、ISRJ 本地候选均与既有关闭机制重叠；
6. 当前 generator contract 不包含可直接使用的非 oracle 第二物理观测，因此没有科学上合格的新 H0 可以继续执行。

## P0 核心否定证据

- matched true 平均 weak AUROC：`0.550727`；
- 相对 strongest control：`-0.007937`，95% CI `[-0.014656,-0.002349]`；
- 正向 seeds：`0/5`；
- binding accuracy：`0.584167 < 0.60`；
- MFT/VFT/ISRJ 通过标签数：`0/3`；
- raw-IQ equal-dimension control 的平均 weak AUROC 为 `0.558620`，高于选择的 B0 global readout。

因此，P1 adapter、bounded residual、校准和阈值均未获授权，也未执行。

## 研究边界

当前证据支持的负主张是：

> 在现有单路复 I/Q、当前 B0 表征和三元组反事实协议下，matched target residual 并未形成超过信息破坏与等维 raw-IQ 控制的、可部署的统一弱成分证据。扩展单通道时间孔径可以使部分标签的物理量可见，但没有形成同时覆盖 MFT、VFT、ISRJ 的稳定学习增量。

该结论不否定未来在额外校准通道、受控 receiver setting 或真实/HIL 观测下建立新的标签归因方法；但那将是新的传感/数据问题，必须重新 Stage -1，不能视为当前 IP-CVE 的继续调参。

## 完整状态

| 阶段 | 状态 |
|---|---|
| Stage -1 admission | success |
| Stage 0 triple protocol | success |
| P0 frozen observability | fail/closed |
| P1 low-rank reconnection | prohibited/not executed |
| calibration/threshold | prohibited/not executed |
| 2x2 dual-axis | not authorized |
| confirmation | not authorized/not generated |
| locked/test-v2/test-v3 | unused |
| post-P0 local extended observation | no-run; no admissible non-overlap candidate |

## 主要产物

- `docs/results/route_a_ipcve_p0_frozen_observability_results_20260807.md`
- `docs/reports/route_a_ipcve_post_p0_extended_observation_admission_20260807.md`
- `outputs/route_a_ipcve/p0_frozen_observability/p0_summary.json`
- `outputs/route_a_ipcve/post_p0_observation_admission/admission_audit.json`
- `outputs/route_a_ipcve/post_p0_observation_admission/run_receipt.json`

全部新工件记录：`locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。

