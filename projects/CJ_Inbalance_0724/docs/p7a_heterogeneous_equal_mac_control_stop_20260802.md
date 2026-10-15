# P7a：异构任务路由的等 MAC 普通卷积对照（开发阶段停止）

## 结论

P7a **未通过开发门槛，永久停止为“异构路由本身带来参数估计收益”的候选结论**。不得以本轮结果构造确认身份池、调整 C2 通道/卷积核/训练超参数，或读取任意测试集与 locked test。

异构候选 C1 相对原始全局骨干 C0 在三个模型种子的 12 项参数宏 NMAE 上均有改善；但在 MAC 与参数量都匹配的普通全局卷积控制 C2 面前，优势不稳定，并对 SawFM 扫频周期产生三种子一致退化。因此，历史 C1 对 C0 的增益不能排除来自额外/不同的参数分支容量，不能归因为异构任务路由机制。

## 预注册和数据边界

- 预注册：`configs/p7a_heterogeneous_equal_mac_control_preregister_20260802.json`
- 方法冻结：`configs/p7a_heterogeneous_equal_mac_control_method_freeze_20260802.json`
- 新开发身份池：种子 `1220000000`，6,300 个唯一身份、63 个组合、每组合 100 个样本；仅 `closedset_6class_train.mat`。
- 审计：`outputs/p7a_heterogeneous_equal_mac_preflight_20260802/data_audit.json`。与既往身份零重叠，12 个回归目标均有限且有变化。
- `test_generated=false`、`test_opened=false`、`locked_test_used=false`。没有生成、读取或选择任何测试数据。

## 对照定义与静态成本

三种模型在完全相同的数据、划分、优化器、损失、20 epoch 训练日程和模型种子 `18041/18042/18043` 下训练：

- C0：`legacy_mlcnn_global`，原全局骨干。
- C1：`mlcnn_iq_grouped_task_routed`，异构 I/Q 分组任务路由候选。
- C2：`legacy_mlcnn_global_compute_matched_control`，非分组、普通全局 I/Q Conv1d 参数分支；它仅作为容量/计算量控制，不是候选创新。

静态前向审计选择 C2 的 67 通道版本：C1 为 459,432,704 MAC、12,890,786 参数；C2 为 459,043,130 MAC（比值 0.999152）和 12,900,137 参数（比值 1.000725）。两项均在预注册的 ±2% 范围内。静态结果：`outputs/p7a_heterogeneous_equal_mac_static_screen_20260802/result.json`。

## 开发结果

完整九个训练收据位于：
`outputs/p7a_heterogeneous_equal_mac_development_20260802/tuned_multitask_20260802_022500/tuned_multitask_20260802_024934/`。

冻结聚合结果：`outputs/p7a_heterogeneous_equal_mac_development_20260802/result.json`，判定为 `stop_no_confirmation_data`。

| 模型种子 | C1 相对 C0 的 full-12 NMAE 降幅 | C1 相对 C2 的 full-12 NMAE 降幅 | C1-C2 Pearson 差 |
|---:|---:|---:|---:|
| 18041 | 5.95% | 1.56% | +0.0261 |
| 18042 | 6.12% | 0.28% | -0.0094 |
| 18043 | 8.67% | 3.73% | +0.0243 |
| 均值 | 6.91% | 1.86% | +0.0136 |

相对 C0 的三个种子门槛通过；相对 C2 的逐种子至少 2%、均值至少 3%、逐种子 Pearson 不低于 C2 均未通过。分类安全门中 macro-AUPR 和 subset accuracy 非劣通过，但不能替代参数路由的核心收益门。

最明确的机制性失败是 `SawFM.sweep_period`：C1 相对 C2 的 NMAE 差为 `+6.27% / +9.17% / +5.28%`（正数代表 C1 更差），三种子均超过预注册的 +3% 系统性伤害界限。其余目标的局部改善（例如 FMB 干扰带宽、ISRJ 采样宽度）不足以抵消该失败。

## 解释与后续约束

P7a 证明了两点：

1. C1 对未经匹配的 C0 有稳定的开发集误差优势；
2. 在严格等 MAC、等参数的普通卷积控制下，这个优势不能稳定保留，并伴随 SawFM 周期回归伤害。

故将 P4/P7 的异构路由结果降级为“对原全局基线的经验改进”，不可写作已验证的路由机制或成本有效创新。P7a 的 C1/C2 精确构造、通道数、损失、训练日程和该身份池均冻结；本分支不再搜索。若未来提出不同的、可证伪的机制，应从新的预注册、新身份池和新的候选定义开始，并先满足等计算控制与目标安全门。
