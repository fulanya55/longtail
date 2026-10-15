# P3o-C：严格参数匹配的延迟峰与局部时频双视图正式开发结果

## 结论

P3o-C 的候选 `DF_dual_matched + C_joint + T_safe` 未通过开发晋升门槛。P3o-B 中双视图低于时频单视图的容量混杂已被消除：四个模型都在 24.4k--26.6k 参数范围内。但在新身份开发集上，DF 相对等参数局部时频强基线 F 的总体 ISRJ Recall 差为 `+0.0059 / -0.0211 / -0.0067`，25 dB AUPR 差为 `-0.0114 / -0.0330 / -0.0040`。因此没有证据支持延迟峰分支为低功率识别带来稳定增益。

DF 在 oracle 与预测标签门控的参数 NMAE 上三种子均相对 F 改善约 2--3.6%，但 miss penalty 与 worst-combination NMAE 并非每个种子都不恶化。它是“参数平均误差略改善、低功率识别及最坏组风险不稳”的权衡，不能作为安全的联合感知创新点。

## 证据边界

- 训练身份 `receiver_noise_seed=1030000000`，正式开发身份 `1040000000`；两侧各 2,079 个 identity，交叉重叠为零，且对 280 个历史身份文件的最大重叠为零。
- 每个 identity 在五个体制和 25/35/50 dB 三个功率视图上严格配对；所有 cell 使用 calibration 的 95% Clopper--Pearson 上界控制 ISRJ FPR <= 0.10。
- 未生成、打开或使用测试/locked test：`test_generated=false`、`test_opened=false`、`locked_test_used=false`。
- 全量回归测试：`243 passed`；9 条既有合成边界警告与 P3o-C 无关。
- 1050M/1060M 确认身份未使用，因为开发门槛未通过。

## 冻结方法

所有模型采用相同的多标签 `BCEWithLogits + 0.1 * masked SmoothL1`、AdamW（`lr=1e-4`、`wd=1e-4`）、2 epoch warmup、20 epoch cosine、batch 63、gradient clip 5。`C_joint` 在 early 角色中按参数 NMAE 容忍 1.01 下最大 AUPR 选择；阈值在 calibration 角色固定。未使用对比损失、伪标签或开发集调参。

| 模型 | 输入 | 参数量 |
|---|---|---:|
| T_time_matched | 原始 I/Q 时域 | 26,370 |
| D_delay_matched | 多尺度自相关的延迟峰序列 | 26,640 |
| F_time_frequency_matched | 局部 STFT 幅度与相邻帧相位增量 | 25,490 |
| DF_dual_matched | 延迟峰序列 + 局部时频融合 | 24,400 |

相对 F 的参数量偏差分别为 +3.45%、+4.51%、0%、-4.28%，均在预注册的 ±5% 范围内。

## `T_safe` 开发结果

| 模型 | ISRJ Recall（2041/2042/2043） | 25 dB Recall | 最大 cell FPR | Macro-AUPR |
|---|---|---|---|---|
| T_time_matched | 0.7737/0.7101/0.6641 | 0.5727/0.4521/0.3777 | 0.0890/0.0459/0.0391 | 0.9425/0.9328/0.9315 |
| D_delay_matched | 0.7874/0.7807/0.8140 | 0.4570/0.4648/0.5250 | 0.0547/0.0528/0.0596 | 0.8686/0.8702/0.8745 |
| F_time_frequency_matched | 0.9036/0.8929/0.9345 | 0.7860/0.7377/0.8443 | 0.0371/0.0283/0.0616 | 0.9838/0.9836/0.9820 |
| DF_dual_matched | 0.9095/0.8717/0.9278 | 0.7688/0.6612/0.8053 | 0.0606/0.0430/0.0567 | 0.9805/0.9752/0.9802 |

## 配对比较：DF 相对 F

| Seed | Recall 差 | clustered bootstrap 95% CI | 25 dB AUPR 差 | oracle NMAE 比 | 预测门控 NMAE 比 | miss 差 | worst-combination 差 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 2041 | +0.0059 | [-0.0050, +0.0163] | -0.0114 | 0.9788 | 0.9787 | -0.0036 | +0.0348 |
| 2042 | -0.0211 | [-0.0334, -0.0096] | -0.0330 | 0.9796 | 0.9718 | +0.0038 | -0.0095 |
| 2043 | -0.0067 | [-0.0158, +0.0032] | -0.0040 | 0.9650 | 0.9636 | -0.0109 | -0.0526 |

## 失败原因

1. 参数容量已匹配，仍然没有低功率识别互补性：DF 的 25 dB AUPR 三种子全部低于 F；2042 的配对 Recall bootstrap 区间完全为负。
2. DF 牺牲了主要识别分支的可分性。DF 的 Macro-AUPR 三种子也低于 F，且 25 dB Recall 均低于 F，表明简单 early fusion 不能把延迟峰作为无害的辅助信息加入。
3. 参数任务呈局部收益：两种门控 NMAE 一致改善，但 2041 worst-combination 和 2042 miss penalty 恶化，故不能按“平均参数 NMAE 更低”宣称鲁棒联合提升。
4. F 本身在 25 dB 的 2042 仍低于预注册 0.78 门槛（0.7377），说明低功率可见性仍是系统性难点，而非仅由融合设计导致。

## 冻结决定与下一假设

冻结 P3o-C，禁止在 1040M 上调整分支宽度、融合、特征、阈值、checkpoint、优化或训练轮数；1050M/1060M 不可作为确认集使用。

下一轮不能再做通用的“延迟 + 时频 early fusion”重试。若继续，应提出新的可证伪机制：例如将延迟峰仅路由至其可识别的 MFT/ISRJ 参数专家，或以不干扰 F 识别 logit 的残差式参数侧支，并把“参数收益是否能在不降低 25 dB F 识别性能和最坏组指标的条件下取得”作为预注册主问题。必须使用新的身份种子，并保留参数匹配 F 作为强识别基线。

## 产物

- `outputs/p3oc_parameter_matched_multiview_development_20260801/summary.json`
- `outputs/p3oc_parameter_matched_multiview_development_20260801/failure_audit.json`
- `outputs/p3oc_parameter_matched_multiview_development_20260801/identity_audit.json`
- `configs/p3oc_parameter_matched_multiview_preregister_20260801.json`
- `configs/p3oc_parameter_matched_multiview_method_freeze_20260801.json`
- `configs/p3oc_parameter_matched_multiview_outcome_freeze_20260801.json`
