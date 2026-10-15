# AMA 六类 IF3 / Anchor 阈值校准与真实 Anchor 诊断

## 阶段 1：验证集校准

现有 `route_a_ama_if3_anchor_baseline_20260903` 的 3 个种子 checkpoint 在固定 80/20 combination-stratified validation split 上重新导出了 logits。阈值和温度/偏置参数仅在 validation 拟合；development-test 只使用冻结参数做一次诊断。

| 条件 | 方法 | Val Macro-F1 | Val 组合准确率 | Test Macro-F1 | Test 组合准确率 | Val ECE |
|---|---|---:|---:|---:|---:|---:|
| IF3 | fixed 0.5 | 0.916482 | 0.610726 | 0.916220 | 0.613708 | 0.058402 |
| IF3 | per-class F1 threshold | 0.928364 | 0.654259 | 0.922248 | 0.632250 | 0.058402 |
| IF3 | combo coordinate threshold | 0.927368 | 0.658675 | 0.922181 | 0.636333 | 0.058402 |
| IF3 | temperature scaling | 0.916482 | 0.610726 | 0.916220 | 0.613708 | 0.037667 |
| IF3 | logit bias (BCE) | 0.921452 | 0.635121 | 0.919703 | 0.630917 | 0.052540 |
| IF3 | temperature + logit bias | 0.925417 | 0.646477 | 0.922044 | 0.634000 | 0.029475 |
| Anchor sampler | fixed 0.5 | 0.910153 | 0.586540 | 0.909399 | 0.589375 | 0.071486 |
| Anchor sampler | combo coordinate threshold | 0.920282 | 0.638065 | 0.916517 | 0.619417 | 0.071486 |

温度缩放只改变概率校准，不改变以 0.5 为界的二元决策，因此 F1/组合准确率不变。IF3 的组合阈值在 validation 上提升 4.80 个百分点，达到预注册的 2 个百分点诊断门槛；但逐类阈值在三个种子间存在明显波动，不能直接冻结为最终阈值。

## 阶段 2：独立真实 Anchor

从同一生成协议的九类 full-power 训练池中过滤纯 AMA 六类，构造了无重复的 Anchor：6415 个 fit 样本，IF=1.0，CF=1.0099，63 组合完整。其 validation 为 Anchor 自身的 80/20 split；另外在 IF3 validation/test 上进行了冻结参数诊断。

Anchor 自身 validation（3 seeds）：Macro-AUPR `0.978202 ± 0.000616`，Macro-F1 `0.915217 ± 0.005394`，组合准确率 `0.623764 ± 0.014961`。

在 IF3 validation split 上评估真实 Anchor（3 seeds）：Macro-AUPR `0.974862 ± 0.000812`，Macro-F1 `0.914865 ± 0.004271`，组合准确率 `0.618507 ± 0.013769`。相对于 IF3 普通采样，三种子平均差异为：Macro-F1 `+0.001617`（IF3−Anchor），组合准确率 `−0.007781`，方向不完全一致。

## 当前判定

1. IF3 的主要损失有相当部分可以由阈值/偏置校准缓解；温度缩放显著降低 ECE，但不能单独提高 0.5 阈值下的 F1。
2. 真实 Anchor 与 IF3 的性能非常接近，说明当前 IF3=3 的边际长尾效应可能小于数据域、组合结构和阈值效应。
3. 旧的均匀组合 sampler Anchor 不能继续作为因果对照；真实 Anchor 已经完成独立数据源、无重复和同语义控制，但仍需 development-v2 五种子确认。
4. 本报告仍属于 development evidence；未读取 locked test，E5 未修改。

## Receipts

- Calibration summary: `outputs/route_a_ama_if3_anchor_baseline_20260903/calibration/summary.json`
- True Anchor summary: `outputs/route_a_ama_true_anchor_20260903/summary.json`
- True Anchor on IF3 validation: `outputs/route_a_ama_true_anchor_20260903/on_if3_validation/summary.json`
