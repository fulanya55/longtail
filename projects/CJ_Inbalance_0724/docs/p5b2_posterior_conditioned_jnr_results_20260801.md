# P5b2：冻结分类后验能否替代 Oracle 组合

## 冻结结论

通过。P5b2 在完全新的 `receiver_noise_seed=840000000` 身份上，证明固定物理特征加**冻结 P4c 分类分支的六标签 sigmoid 后验**足以替代不可部署的 oracle 组合，完成 MFT/VFT/ISRJ 的扫描 JNR 估计。D1 通过全部 12 项预注册门槛。

这只放行一个后续问题：在另一批全新身份上，能否把这种小型估计器有界地附着到冻结 P4c 的 JNR 输出，而不改变分类 logits、阈值、分类决策或非 JNR 参数。P5b2 本身不是端到端提升或部署结论。

## 数据与隔离

- 预注册：`configs/p5b2_posterior_conditioned_jnr_preregister_20260801.json`，SHA256 `cd6a486472ff6a0b4c436b432642f09f20fb089c4c93eac4ed64e18d20500186`。
- 10/15/20/25/30 dB 五档、每档 3,780 个新身份、63 个组合、全部功率按 `receiver_noise_seed` 配对。
- train/early/confirm 每组合为 36/12/12，五档使用同一角色。与根数据、P4c、P4d、P4k、P5a、P5a2、P5b、P5b1 均为零身份重叠。
- D1 不接收真标签、标称功率或原始 I/Q 深度专家；它只使用 26 维固定时/频/自相关特征与对应冻结分类模型的六维 sigmoid 后验。
- D2 与 D1 同模型、同优化预算，仅将后验替换为真标签，是不可部署的 oracle 上界。

## confirm 结果（三模型种子均值）

| 方法 | MFT.JNR NMAE / r | VFT.JNR NMAE / r | ISRJ.JNR NMAE / r |
|---|---:|---:|---:|
| D0：train median 常数 | 0.84853 / 0.00000 | 0.84853 / 0.00000 | 0.84853 / 0.00000 |
| D1：物理特征 + 冻结后验 | **0.22834 / 0.95137** | **0.25114 / 0.94200** | **0.29501 / 0.92061** |
| D2：物理特征 + oracle 组合 | 0.18448 / 0.96499 | 0.19085 / 0.96323 | 0.20025 / 0.95906 |

D1 相对 D0 的 NMAE 降幅为 MFT **73.09%**、VFT **70.40%**、ISRJ **65.23%**；10 dB 降幅为 **83.63%**、**81.13%**、**77.32%**。MFT/ISRJ 的三种子均为正，相关系数超过 0.7，逐组合没有比 D0 差超过 0.03 的情况，identity-clustered bootstrap 的收益 CI 下界均大于 0。

相对 D2 oracle 上界，D1 NMAE 的倍率为 MFT 1.238、VFT 1.316、ISRJ 1.473，MFT/ISRJ 均低于预注册的 1.6 倍门槛。因此差距真实存在，却不足以否定“冻结后验可作为可观测条件”的假设。

## 创新点的证据链位置

P5b1 和 P5b2 合起来建立了可证伪的两段证据：

1. 不是纯粹不可辨识：oracle 组合下，固定物理统计与小模型已能恢复扫描 JNR。
2. 不是 oracle 幻象：以冻结多标签后验替代组合后，关键 MFT/ISRJ 信息仍保留。

尚未成立的表述是“附着后可安全改善完整复合干扰系统”。下一步 P5b3 必须在全新身份做冻结主网的 JNR-only 有界附着对照，并同时报告 oracle-gated 与 predicted-gated 参数指标及分类零变化审计。

## 产物

- 实现：`cj0724/p5b2_posterior_conditioned_jnr.py`
- 测试：`tests/test_p5b2_posterior_conditioned_jnr.py`
- 数据审计：`outputs/p5b2_posterior_conditioned_jnr_20260801/data_audit.json`
- 结果：`outputs/p5b2_posterior_conditioned_jnr_20260801/result.json`，SHA256 `e683b1420df1e910f89c203d2ead35dcc15b9a7b9958582f7750ab240244e0dd`

`test_generated=false`，`test_opened=false`，`locked_test_used=false`。
