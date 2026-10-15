# P3o-E：冻结参数侧支的开发与两套独立确认结果

## 结论

P3o-E 在 1110M/1120M formal development 的全部冻结门槛通过：识别 logits 逐位不变，三种子 oracle 与 predicted-gate 参数 NMAE 分别改善 4.5--6.2% 与 4.6--6.5%，且 miss penalty、worst-combination 均不恶化。

但独立确认不稳定：1130M 失败，1140M 通过。因此该方法不能晋升到 locked test，也不能声称为可靠的安全参数增强创新点。

## 证据边界

- development：训练 1110M、开发 1120M；两套 confirmation：1130M、1140M；每侧 2,079 identities，身份审计均为零重叠。
- 所有确认运行固定 P3o-E 方法、训练池、模型 seeds、优化、checkpoint 规则和安全校准；确认身份不参与选择。
- 未生成、打开或使用 test/locked test。

## 确认结果

| 确认身份 | 结果 | 关键证据 |
|---|---|---|
| 1130M | 失败 | logits 均相同，NMAE 三种子均改善；但 worst-combination NMAE 相对基线为 `+0.00245/+0.03416/-0.02742`，前两种子恶化。 |
| 1140M | 通过 | logits 均相同，oracle NMAE 比为 `0.9377/0.9465/0.9432`，predicted-gate 比为 `0.9357/0.9449/0.9418`，worst-combination 差为 `-0.01892/-0.00644/-0.06136`。 |

## 失败归因与冻结决定

P3o-E 修复了 P3o-D 的 BatchNorm 漂移：主干 logits 在 development 和两套 confirmation 中都逐位一致。因此失败不是分类安全漏洞，而是残差参数修正的最坏组合风险不能跨身份稳定迁移。

冻结 P3o-E/P3o-F：不得在 1120M、1130M、1140M 上调残差宽度、后验门、损失、阈值、checkpoint 或训练轮数；不得访问 locked test。后续假设必须用新的身份，且以 worst-combination 风险收缩为主机制，而不是仅优化平均 NMAE。

## 产物

- `outputs/p3oe_invariant_frequency_delay_parameter_residual_development_20260802/summary.json`
- `outputs/p3of_invariant_residual_confirmation_seed1130000000_20260802/summary.json`
- `outputs/p3of_invariant_residual_confirmation_seed1140000000_20260802/summary.json`
