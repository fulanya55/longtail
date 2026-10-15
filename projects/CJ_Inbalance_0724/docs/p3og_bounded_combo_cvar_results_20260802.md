# P3o-G：受限残差与复合组合 CVaR 的开发失败记录

## 结论

P3o-G 是一次 **实现契约失败**，不能用来支持或否定“复合组合最坏风险约束”这一研究假设。方法、1230M/1240M 身份和本次输出均已冻结；不得在这些身份上修复、重跑或调参，也不得生成其 1250M/1260M 确认集。

## 已完成且有效的协议证据

- 训练身份 1230M 与开发身份 1240M 各含 2,079 个唯一 receiver-noise identities；五种机制与 25/35/50 dB 视图均配对。
- 训练/开发身份交集为 0；与 323 个既有历史身份文件的最大交集为 0。
- P3o-G 的 F-base 与两个侧支在全部三种子中保持主干 state/buffer 与 logits 逐位一致。
- 未生成、打开或使用 test/locked test。

## 失败机制

P3o-G 的 `_per_sample_parameter_loss` 先对完整 `[B,6,2]` 张量计算 `smooth_l1_loss`，随后才用有效掩码相乘。参数协议规定 inactive targets 为 NaN；因此即使相应 mask 为 0，`NaN * 0` 仍为 NaN。结果是：

- 三个 seed 的 G0（均值残差）和 G1（组合 CVaR）均从 epoch 1 起产生 NaN 参数损失；
- 早期集的 oracle worst-combination 选择量为 NaN，所有侧支均被形式上选为 epoch 1；
- 六个侧支的 oracle NMAE、predicted-gate NMAE、miss penalty 与 worst-combination NMAE 均为 NaN，不能形成比较；
- 因而 `summary.json` 中全部 G1 比值为 NaN，P3o-G 的参数 gate 必然失败。

这与“CVaR 没有泛化能力”不同：实验没有实现有效的 masked parameter objective，故不构成机制反证。

## 次要但独立的开发 gate

F-base 在 seed 2061/2062 的最大单元 FPR 为 0.08602/0.04399，但 seed 2063 为 0.10753，故基础安全 gate 也未达到 3/3。该观察仅属于本次已消耗开发身份上的诊断，不能用于新方法选择。

## 后续最小可证伪路线：P3o-H

P3o-H 必须是独立实现并使用未使用的身份。开始生成前必须验证：

1. 逐样本损失先以 `valid` 布尔索引取出 prediction 与 target，再计算 Smooth-L1，保证 inactive NaN 永不进入损失算子；
2. synthetic batch 含 inactive NaN 时，G0/G1 loss、梯度及一次 AdamW step 后所有可训练参数均为 finite；
3. 复合 bit-mask 分组 CVaR 与均值对照具有完全相同的侧支、初值、数据顺序、训练预算与 checkpoint 规则；
4. 使用全新 train/development identities，保留相同的主干逐位不变性、最坏组合和 predicted-gate gate；不访问 locked test。

## 产物

- 预注册：[p3og_bounded_combo_cvar_preregister_20260802.json](../configs/p3og_bounded_combo_cvar_preregister_20260802.json)
- 方法冻结：[p3og_bounded_combo_cvar_method_freeze_20260802.json](../configs/p3og_bounded_combo_cvar_method_freeze_20260802.json)
- 结果：[summary.json](../outputs/p3og_bounded_combo_cvar_development_20260802_retry1/summary.json)
- 失败冻结：[p3og_bounded_combo_cvar_outcome_freeze_20260802.json](../configs/p3og_bounded_combo_cvar_outcome_freeze_20260802.json)
