# P3o-B：延迟峰与局部时频双视图的正式开发结果

## 结论

P3o-B 的预注册候选 `DF_exclusive_contrast + C_joint + T_safe` 未通过开发门槛。它在三种子中均满足最大 cell FPR <= 0.10，但总体 ISRJ Recall 为 `0.8187 / 0.7802 / 0.8623`，25 dB Recall 为 `0.5527 / 0.4684 / 0.6689`，均未满足 3/3 门槛。

同时发现一个需要保留、但不能夸大的积极现象：局部 STFT + 相邻帧瞬时频率单视图在三种子总体安全 Recall 为 `0.8936 / 0.9243 / 0.8650`，且最大 cell FPR 为 `0.0352 / 0.0459 / 0.0225`。不过其 25 dB Recall 为 `0.7593 / 0.8123 / 0.6869`，仍不能解决低功率门槛。因此它只是下一假设的强基线，不是已验证的最终创新点。

## 证据边界

- 训练身份：`receiver_noise_seed=910000000`；正式开发身份：`920000000`。
- 两侧各 2,079 个身份，交叉身份重叠为零；同一身份的 5 个体制和 3 个功率视图严格成组。
- 未生成或访问测试/locked test：`test_generated=false`、`test_opened=false`、`locked_test_used=false`。
- 全量回归：`241 passed`（9 个既有合成边界警告）。
- attempt-1 在第一个模型完成后因缺失评估损失字段而在序列化前失败；没有使用任何开发结果进行模型、阈值或 checkpoint 选择。retry-1 仅补回缺失的损失记账字段，协议和方法未变；attempt-1 产物完整保留。

## 冻结方案

| 模型 | 输入/机制 | 参数量 |
|---|---|---:|
| T_time | 原始 I/Q 时域卷积 | 21,074 |
| D_delay | 1--1024 lag 复自相关及 7/31 平滑延迟序列 | 21,314 |
| F_time_frequency | 局部 STFT（128/64）log magnitude + 相邻帧相位增量 | 25,490 |
| DF_no_contrast | 延迟序列与局部时频融合，无对比损失 | 17,194 |
| DF_exclusive_contrast | 同一双视图；只对 MFT-only 与 ISRJ-only 用原型交叉熵 | 17,322 |

所有模型使用 `BCEWithLogits + 0.1*masked SmoothL1`、AdamW (`lr=1e-4`、`wd=1e-4`、warmup=2、cosine、20 epochs、batch=63)。候选额外使用 `0.05*exclusive prototype CE`；MFT 与 ISRJ 共存样本从该项排除，主任务始终保持多标签 BCE。checkpoint 为参数误差容忍 1.01 下 AUPR 最大的 `C_joint`；ISRJ 阈值仅由 calibration 中的逐体制 x 功率 95% Clopper-Pearson 上界 <=0.10 决定。

## `T_safe` 开发集结果

| 模型 | Seed | ISRJ Recall | 25 dB Recall | 最大 cell FPR | Macro-AUPR |
|---|---:|---:|---:|---:|---:|
| T_time | 2041/2042/2043 | 0.6277/0.5803/0.6925 | 0.3229/0.2602/0.4205 | 0.0411/0.0254/0.0430 | 0.9375/0.9306/0.9349 |
| D_delay | 2041/2042/2043 | 0.7845/0.7992/0.6764 | 0.4627/0.5188/0.3108 | 0.0244/0.0499/0.0127 | 0.8738/0.8710/0.8742 |
| F_time_frequency | 2041/2042/2043 | 0.8936/0.9243/0.8650 | 0.7593/0.8123/0.6869 | 0.0352/0.0459/0.0225 | 0.9837/0.9862/0.9829 |
| DF_no_contrast | 2041/2042/2043 | 0.6865/0.8316/0.8612 | 0.3371/0.5839/0.6534 | 0.0088/0.0332/0.0420 | 0.9827/0.9758/0.9808 |
| DF_exclusive_contrast | 2041/2042/2043 | 0.8187/0.7802/0.8623 | 0.5527/0.4684/0.6689 | 0.0215/0.0147/0.0420 | 0.9813/0.9761/0.9801 |

## 失败原因

1. 独占对比的 Recall 差相对 `DF_no_contrast` 是 `+0.13220 / -0.05139 / +0.00107`。以 `receiver_noise_seed` 聚类 bootstrap 的 95% CI 分别为 `[+0.12169,+0.14311]`、`[-0.06013,-0.04249]`、`[-0.00690,+0.00876]`，没有 3/3 一致收益。
2. 候选在 25 dB AUPR 上优于延迟单视图，却在所有种子低于局部时频单视图（`-0.01726/-0.03050/-0.02446`）。因此没有证明延迟分支在该配置下提供互补信息。
3. 12 参数联合感知不稳：候选对无对比的 oracle/predicted-gate NMAE 比在种子 2041 与 2043 超过 1%，worst-combination NMAE 在三种子均恶化。
4. 双视图参数量（17.2k）低于时频单视图（25.5k）。因此本轮还不能把融合低于时频单视图归因为物理视图无用；它揭示的是 P3o-B 的容量匹配不充分。

## 冻结决定与下一假设

P3o-B 失败，禁止在 920M 上再调 width、STFT、ACF lag、对比权重、阈值、checkpoint、学习率或训练轮数；930M/940M 也不得用于确认。

下一步只能是新的 P3o-C：用全新身份，将时域、时频、双视图严格参数匹配，并把局部时频单视图作为主强基线。P3o-C 的核心可证伪问题是：在控制容量后，延迟峰序列是否仍为 25 dB 安全识别带来 3/3 的额外收益；若不能，则应放弃延迟融合，而将局部时频表征作为低功率方向的主要候选。

## 产物

- `outputs/p3ob_dualview_representation_development_20260801_retry1/summary.json`
- `outputs/p3ob_dualview_representation_development_20260801_retry1/failure_audit.json`
- `outputs/p3ob_dualview_representation_development_20260801/attempt1_codepath_failure.json`
- `configs/p3ob_dualview_representation_preregister_20260801.json`
- `configs/p3ob_dualview_representation_method_freeze_20260801.json`
- `configs/p3ob_dualview_representation_outcome_freeze_20260801.json`
