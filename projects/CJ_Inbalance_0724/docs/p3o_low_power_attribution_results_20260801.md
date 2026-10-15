# P3o-A 低功率归因实验：正式开发结果与冻结结论

## 结论

P3o-A 未通过预注册晋级门槛。扩大训练身份覆盖能够稳定提高 25 dB 的排序质量，且 Clopper-Pearson 安全校准在所有条件下可行；但它没有恢复安全阈值下的 ISRJ 召回，也未稳定保持 12 参数联合感知性能。因此，不能把“更多覆盖 + 联合 checkpoint + CP-safe 阈值”表述为低功率复合干扰联合感知的有效解法。

本轮只使用 fresh identity-safe formal development：训练池 `receiver_noise_seed=830000000`，开发集 `860000000`，训练/开发各有 2,079 个物理身份，交集为零。未生成或访问测试集：`test_generated=false`、`test_opened=false`、`locked_test_used=false`。

## 冻结设计

- 数据：5 个体制（ID_control、low_count、close_spacing、ultra_close、weak_target）x 25/35/50 dB x 63 个复合组合；同一 `receiver_noise_seed` 的全部体制和功率视图保持成组。
- 覆盖对照：Small epoch-matched（6,615 fit views, 20 epochs）、Small step-matched（78 epochs）与 Full epoch-matched（25,515 fit views, 20 epochs）。
- 模型：`mlcnn_iq_grouped_task_routed`，输入 `[B,2,3000]`，width=24、分类头=256、参数隐层=64、dropout=0.2。
- 损失：`BCEWithLogits + 0.1 * masked SmoothL1`；参数项只在 `label=1 AND parameter_mask=1 AND finite(target)` 上计算。
- 优化：AdamW，lr=`1e-4`，weight decay=`1e-4`，warmup=2 epochs，cosine，batch=63，grad clip=5，模型种子 2041/2042/2043。
- 对照因素：`C_param`（early oracle parameter macro-NMAE 最小）与 `C_joint`（参数误差不超过 `1.01*C_param` 的 epoch 中 early macro-AUPR 最大）；`T_F1` 与只替换 ISRJ 阈值的 `T_safe`。`T_safe` 要求每个体制 x 功率负例 cell 的单侧 95% Clopper-Pearson FPR 上界不大于 0.10。
- 禁用：hard-negative、条件 margin、router、蒸馏、新专家；因此本轮只归因覆盖、checkpoint 和安全校准。

实现 SHA-256：`CF0A5D765289BCD69D502B36BEFD9E29BEA2E73B48526307661512361139E288`。预注册 SHA-256：`65B0C88B882228BDFD1C8880A3EA151E9ABE1B648B4CB83BC3356926AB1B0DB7`。

## 正式开发结果

候选为预注册的 `Full + C_joint + T_safe`。下表是开发集的一次性评估，不用于再选择。

| 模型种子 | ISRJ Recall | 25 dB Recall | 最大 cell FPR | Macro-AUPR | Oracle 参数 NMAE | Predicted-gate NMAE |
|---:|---:|---:|---:|---:|---:|---:|
| 2041 | 0.7424 | 0.5595 | 0.0567 | 0.9625 | 0.3402 | 0.3386 |
| 2042 | 0.5545 | 0.3848 | 0.0215 | 0.9630 | 0.3419 | 0.3421 |
| 2043 | 0.6451 | 0.4653 | 0.0342 | 0.9627 | 0.3371 | 0.3355 |
| 均值 | 0.6474 | 0.4699 | 0.0375 | 0.9627 | 0.3397 | 0.3387 |

安全性本身通过：3/3 种子的最大观测 cell FPR 均小于 0.10，且校准中 CP-safe 阈值均有可行解。可是总体 Recall 没有达到 0.85，25 dB Recall 没有一个种子达到 0.78。

与 `Full + C_param + T_F1` 相比，候选 ISRJ Recall 的配对差分别为 `-0.18914`、`-0.33567`、`-0.24867`。以 `receiver_noise_seed` 为聚类单元的 paired bootstrap 95% CI 分别为：

- 2041：`[-0.20443, -0.17451]`
- 2042：`[-0.35464, -0.31670]`
- 2043：`[-0.26557, -0.23074]`

三条区间均严格低于零，安全阈值造成的召回下降不是随机抽样波动能够解释的。

## 归因

1. 覆盖有效但不足。Full 相对 Small epoch-matched 的 25 dB macro-AUPR 差为 `+0.06165/+0.04672/+0.05051`（3/3 正向），证明扩大身份覆盖改善了排序；然而候选 25 dB Recall 仍为 `0.5595/0.3848/0.4653`。
2. 这不是“Small 训练步数不够”。Small step-matched（78 epochs）候选 Recall 为 `0.6456/0.5844/0.5576`，没有系统性优于 Small epoch-matched。因此 Full 的收益不能归因为仅增加优化步数。
3. 低功率失败跨体制存在，而非只由 ultra_close 引起。Full 候选在 25 dB 的平均 Recall：ID_control `0.4773`、low_count `0.4523`、close_spacing `0.4662`、ultra_close `0.4747`、weak_target `0.4766`。这支持低功率可分信息不足的解释。
4. `C_joint` 不能稳定解除参数优先选模冲突。其相对 `C_param` 的 macro-AUPR 差为 `+0.000955/+0.000360/-0.000058`，第三种子方向已不满足非负门槛；同时候选的 oracle/predicted-gate 参数 NMAE、miss-penalty 或 worst-combination 至少有一个种子恶化超过预注册限制。
5. 因而不能继续在 860M 上微调阈值、checkpoint、训练轮数、学习率或损失权重，也不能使用 870M/880M confirmation。这些身份只在 P3o-A 开发门槛通过后才可使用。

## 门槛判定

通过：身份隔离、3/3 最大 cell FPR、3/3 各 25 dB 体制 FPR、Full 相对 Small 的 25 dB AUPR、CP-safe 可行性、测试/locked-test 未使用。

失败：3/3 总体 Recall >= 0.85、相对 `Full+C_param+T_F1` 的 Recall 下降不超过 0.02、3/3 25 dB Recall >= 0.78、oracle NMAE、predicted-gate NMAE、miss/worst-combination 不恶化，以及 `C_joint` 相对 `C_param` 的 AUPR 3/3 非负。

全量验证：`D:\anaconda3\envs\PY10\python.exe -m pytest -q`，结果为 `234 passed`（9 个已知的合成边界警告）。

## 冻结后的唯一下一步

进入 P3o-B，但必须作为全新的可证伪机制重新预注册并使用新身份。它应引入多尺度自相关/延迟峰序列与局部 STFT/瞬时频率双视图，仅在 MFT-only 与 ISRJ-only 样本上施加监督对比或原型分离；允许 MFT 与 ISRJ 共存，不能改写为 softmax 互斥分类。必须同时保留等参数的普通时序卷积、仅延迟视图、仅时频视图、无对比损失四项消融，并在新开发身份上重新冻结 checkpoint、校准与阈值。

## 产物

- `outputs/p3o_low_power_attribution_development_20260801/summary.json`
- `outputs/p3o_low_power_attribution_development_20260801/failure_audit.json`
- `outputs/p3o_low_power_attribution_development_20260801/identity_audit.json`
- `configs/p3o_low_power_attribution_preregister_20260801.json`
- `configs/p3o_low_power_attribution_method_freeze_20260801.json`
- `configs/p3o_low_power_attribution_outcome_freeze_20260801.json`
