# P3o-U 开发集结果：局部时频相位动力学条件 ISRJ 校准

## 结论与证据边界

**结论：失败，停止；不生成 confirmation，不读取或使用 locked test。**

P3o-U 用不同于 P3o-T 逆延迟支持的单 PRI 局部相位/STFT 动力学，尝试在冻结的预测 MFT 后验门控下抑制 MFT→ISRJ 误报。三 seed 中 U2 相对 U0 的源 FPR 均有小幅下降，但未达到每 seed 2pp 门，ISRJ recall 均下降约 3pp；更关键的是 U2 对时间 bin 置乱控制 U3 几乎无稳定额外收益。因此不能归因于局部时频相位结构，且安全门失败，路线冻结。

这份报告是独立 fresh-development 证据，不能替代 confirmation 或 locked-test 结论。

## 冻结设计与审计

- 骨干：冻结 `mlcnn_iq_grouped_task_routed`，base seed 2041/2042/2043；残差 seed 2501/2502/2503。
- U0：原安全阈值基线；U1：仅预测 MFT 后验的同容量校准；U2：固定局部相位/STFT 动力学；U3：固定时间 bin 置乱的同维度控制。
- U2 只向 ISRJ logit 加有界负残差，MFT 和其余五个 logits 位级不变；三 seed MFT AP 差均为 0。
- 训练：AdamW，lr 3e-4，weight decay 1e-4，15 epochs，2 epoch warmup+cosine，batch 128，clip 5，条件损失权重 0.25；只按 selection ISRJ BCE 选 checkpoint。
- 新鲜数据：train/selection/development identity seeds 为 2010000000/2020000000/2030000000；每角色 15 视图、每视图 2,079 样本。45 个原始视图和缓存均全量一致；三角色交集为 0；与 3,056 个历史文件最大交集为 0。
- 统计：以 `receiver_noise_seed` 聚类的 2,000 次 paired bootstrap。`test_generated=false`、`test_opened=false`、`locked_test_used=false`。

## 关键结果（冻结安全阈值）

| base / residual | U2-U0 源 FPR 下降 | 95% CI | U2-U1 | U2-U3 | U2-U0 ISRJ recall 变化 |
| --- | ---: | --- | ---: | ---: | ---: |
| 2041 / 2501 | +0.01338 | [0.00947, 0.01768] | +0.00783 | -0.00038 | -0.03081 |
| 2042 / 2502 | +0.01705 | [0.01288, 0.02147] | +0.01376 | -0.00051 | -0.03049 |
| 2043 / 2503 | +0.01503 | [0.01149, 0.01932] | +0.00152 | +0.00063 | -0.02999 |

| seed | U0 source FPR | U1 posterior-only | U2 phase dynamics | U3 time-bin shuffled |
| --- | ---: | ---: | ---: | ---: |
| 2041 | 0.08321 | 0.07765 | 0.06982 | 0.06944 |
| 2042 | 0.10063 | 0.09735 | 0.08359 | 0.08308 |
| 2043 | 0.10404 | 0.09053 | 0.08902 | 0.08965 |

## 门控判定

通过：身份隔离、最坏 source-cell 非恶化、MFT AP 位级不变、macro-AUPR/F1 安全、predicted-gate NMAE 安全、miss penalty 非恶化、4/5 工况和 25 dB 方向门、test/locked-test 安全标记。

失败：

1. U2 相对 U0 未在三个 seed 均达到 0.02 FPR 降幅。
2. U2 相对 U1 和 U3 均未在三个 seed 达到 0.01；U2-U3 分别为 -0.00038、-0.00051、+0.00063。
3. U2-U3 的 bootstrap 下界在前两 seed 为负，U2-U1 的第三 seed 下界为负。
4. U2-U0 的 ISRJ recall 在三个 seed 均低于 -0.02（约 -0.03）。
5. 第三 seed 的最坏组合 predicted-gate NMAE 从 1.57556 升至 1.57654，故系统安全门失败。

## 失败机制

1. **置乱控制不可区分。** U3 基本复制 U2 的 FPR 与 recall 变化，表明目前的时间 bin 动力学排列没有提供可归因的额外信息；收益更像是带预测 MFT 门的有界负校准。
2. **FPR-召回代价过大。** U2 的小幅 FPR 改善伴随约 3pp 真 ISRJ recall 损失，不符合“安全跨干扰判别”而非单阈值压低的目标。
3. **后验校准仍占主要作用。** U1 已解释部分收益；U2 对 U1 的三 seed 贡献不稳定。
4. **下游最坏组合风险未完全保持。** 整体 predicted-gate NMAE 和 miss penalty 可接受，不足以覆盖第三 seed 最坏组合的轻微恶化。

## 后续边界

不得在本轮身份上调 STFT 窗长、hop、相位窗口、残差边界、损失、阈值、epoch 或 seed。P3o-T 的逆延迟输入和 P3o-U 的局部相位时频输入均作为“强控制不可区分”的失败路线冻结。下一机制若继续，必须来自第三个不同的信息合同，并在提出前审计已有全局 ACF、分离和参数专家失败，避免变相回调；若第三个独立可观测机制仍未通过，按 V7 停止扩展 MFT→ISRJ 线并转向 VFT/SawFM 未证实路线。

## 产物

- 预注册/冻结：`configs/p3ou_local_phase_dynamics_preregister_20260803.json`、`configs/p3ou_local_phase_dynamics_method_freeze_20260803.json`
- 预检：`outputs/p3ou_local_phase_dynamics_preflight_20260803/data_audit.json`
- 逐 seed：`outputs/p3ou_local_phase_dynamics_development_20260803/base2041_result.json`、`base2042_result.json`、`base2043_result.json`
- 汇总/预测：`outputs/p3ou_local_phase_dynamics_development_20260803/summary.json`、`development_predictions.npz`
