# P3o-E：严格冻结频率识别的延迟残差确认结论

## 结论

P3o-E 的三种子开发通过，但在两个新身份确认中的最坏组合风险门失败，故该**延迟峰参数残差变体永久停止**。不得使用 P3o-E 的 1130M/1140M 确认身份继续调节损失、残差宽度/尺度、checkpoint、阈值或组合风险门；也不得访问 test 或 locked test。

## 证据边界

- 开发：1110M 训练身份、1120M 开发身份，三模型种子 2041/2042/2043。
- 确认：1130M、1140M 两个全新身份；各身份 2,079 identities、63 个组合、每组合 33 identities、五个困难情景和 25/35/50 dB 配对视图。
- 训练与两个确认身份互相零重叠，且对 303 个历史身份文件的最大重叠为零；两个确认身份的互相重叠也为零。
- 确认只加载开发阶段已选的 base/side checkpoint；参数 normalizer 和 ISRJ 安全阈值只由原 1110M full-fit/calibration 角色重构。1120M、test 和 locked test 均未读取。

## 通过项

六个“冻结模型种子 × 新确认身份”单元均满足：

- F backbone logits 与侧支 logits 逐位相同；
- 基线安全阈值可行、maximum cell FPR 不超过 0.10，整体及 25 dB ISRJ recall 均达预注册门；
- oracle 与 predicted-gate 全 12 参数 NMAE 比率均不超过 0.98；
- MFT、ISRJ oracle NMAE 比率均不超过 0.99；
- miss penalty 均下降，范围为 -0.0219 到 -0.0277。

平均参数改善因而不是由分类 logit 修改造成：oracle NMAE ratio 为 0.9377--0.9500，predicted-gate NMAE ratio 为 0.9357--0.9485。

## 停止原因

预注册要求每个确认单元的 worst-combination predicted-gate NMAE 相对 F 基线不恶化。1130M 上出现两个确定的反例：

| Model seed | Worst-combination NMAE delta (side - base) |
|---:|---:|
| 2041 | +0.0024 |
| 2042 | +0.0342 |

其余四个单元为非正 delta（-0.0189、-0.0064、-0.0274、-0.0614），但不能抵消任一确认单元的风险恶化。

这说明延迟峰侧支携带可转移的平均参数信息，却仍会把误差集中到困难组合；它不是“识别不变且最坏组合安全”的可靠全 12 参数创新。该失败不同于响应放大：识别路径逐位不变，失败根源是参数侧支的组合风险分配。

## 冻结产物

- 开发汇总：`outputs/p3oe_invariant_frequency_delay_parameter_residual_development_20260802/summary.json`
- 确认汇总：`outputs/p3oe_invariant_frequency_delay_residual_confirmation_20260802/summary.json`
- 确认预注册与冻结：`configs/p3oe_invariant_frequency_delay_residual_confirmation_preregister_20260802.json`、`configs/p3oe_invariant_frequency_delay_residual_confirmation_method_freeze_20260802.json`
- 状态不变量实现：`cj0724/p3oe_invariant_frequency_delay_parameter_residual.py`
