# Route A Backbone M4/Local Statistics 结果与关闭结论

日期：2026-08-06  
正式候选：`B4_F2_DFMS_LocalStatsK16`  
证据等级：`five_seed_development_only_m4_k16_candidate`  
决策：`fail_stage2_close_candidate`

## 1. top-k 一次性选择

只读取 seed 2039 的 validation history：

| k | 最佳 epoch | validation common macro-AUPR | validation target weak AUROC |
|---:|---:|---:|---:|
| 8 | 33 | 0.970293 | 0.881318 |
| 16 | 33 | 0.970534 | 0.881601 |

按预注册主键选择 k=16。选择收据明确记录未读取 early、calibration 或 threshold 指标用于选 k；k=8 永久关闭。

## 2. 正式五 seed 配对结果

| 指标 | B0 | M4-k16 | 配对均值差 | 配对 bootstrap 95% CI | 正增益 seed |
|---|---:|---:|---:|---:|---:|
| macro-AUPR | 0.843746 | 0.843342 | -0.000404 | [-0.001750, +0.000731] | 3/5 |
| 三目标弱正 AUROC | 0.555864 | 0.555845 | -0.000018 | [-0.004302, +0.007899] | 1/5 |
| 三目标弱正 recall | 0.237524 | 0.234222 | -0.003301 | [-0.005421, -0.001475] | 0/5 |
| 三目标强正 recall | 0.771434 | 0.767293 | -0.004142 | [-0.008794, +0.001026] | 2/5 |
| 三目标负样本 FPR | 0.195215 | 0.193871 | -0.001344 | [-0.005699, +0.002258] | — |

局部 max/top-k 旁路没有改善排序，并在所有五个 seed 上降低目标弱召回。五个 checkpoint 的有效残差系数为 `-0.00941,-0.00878,-0.00936,-0.00952,+0.00843`；4/5 seed 主动学习负残差，表明当前主干的局部高响应统计总体更像混杂/强分量噪声，而不是可直接叠加的弱证据。

逐标签上，VFT 弱召回平均 `-0.01263` 且 4/5 下降，是主要损失来源；MFT/ISRJ 也没有稳定 AUROC 增益。安全标签没有系统性 AUPR 退化，但这不足以抵消核心指标失败。

## 3. 代价与证据

- 参数：290,664，较 B0 增加约 12.9%。
- Conv/Linear MACs：211,890,304，较 B0 仅增加约 0.015%；top-k 比较操作另计但不是乘加。
- batch-1 CUDA latency：4.081 ms，较 B0 本机均值增加约 0.61%。
- 最大峰值 CUDA memory：1,064,053,248 bytes。
- summary SHA256：`198796785228aee645b201e846687b82a4576b2905a757a999bab54752a5d879`
- comparison SHA256：`5b27b99e079660cf9a7d2c72961b44cdebad7c4e4af623e4ef9f1d7e7fb7c506`
- selection receipt SHA256：`b8270357f9df8980e38958c6a3c0133efc38f835599105657cf979adf53880f1`
- terminal receipt：`completed`，正式运行耗时 290.023 s。

M4 不进入随机位置 top-k 或同参数 MLP 控制，并永久关闭。test-v2、locked test、test-v3 未使用。由于 M1–M4 均未通过 Stage-2 硬门槛，流程进入预注册科学停止路径。
