# 路线 A：Selective Unknown 决策预注册方案（2026-07-25）

## 1. 目的

验证 50 dB 下的低分 SawFM 正样本是否属于“证据不足、应输出 unknown”，而不是“确定不存在、应输出 negative”。本轮不修改任何表征、专家、融合权重或校准器参数，只学习两个决策阈值。

## 2. 冻结来源

- F2、H2、稳定 S2、C2 `alpha=0.80` 全冻结；
- A0 使用上一轮已经冻结的 `V2_VisibilityAffine` checkpoint、feature mean/std 和 source hash；
- 2043 继续使用 validation-only 选择的 S2 初始化 274302；
- 不重新训练或重新选择 checkpoint；
- locked test 不使用。

## 3. 数据角色

严格重放上一轮稳定专家校准协议：

- expert-scale：仅复用已冻结 C2 的尺度构造；
- visibility-fit：不用于本轮阈值搜索；
- power-threshold：搜索 `t_neg/t_pos`；
- ordinary-threshold：约束普通域 FPR 与 coverage；
- ordinary early 和 40/45/50 dB early-development：最终一次评价；
- 所有功率视图按 `receiver_noise_seed` 绑定；角色间 identity overlap 必须为 0。

## 4. 变体

- `U0_A0SingleThreshold`：上一轮 A0 单阈值基线；低于阈值全部判 negative。
- `U1_C2DualThreshold`：统一 Platt 校准的 C2 分数，学习一组双阈值。
- `U2_A0Selective`：A0/V2 分数，学习一组双阈值；唯一晋级候选。
- `U3_OraclePowerSelective`：A0/V2 分数，ordinary、40、45、50 各自学习双阈值；仅作为结构上限。

## 5. 决策与指标

`p >= t_pos -> positive`；`p <= t_neg -> negative`；其他为 `unknown`。

报告：

- confirmed-positive recall；
- confirmed-positive FPR；
- positive false-negative rate；
- recoverable-positive coverage=`1-positive false-negative rate`；
- unknown coverage；
- unknown positive rate；
- decision coverage；
- selective risk=已决策样本中的错误率。

## 6. 阈值搜索

- 正式网格步长 0.01；smoke 使用 0.05；
- 仅使用 ordinary-threshold 与 power-threshold；
- 统一双阈值候选必须满足：
  - ordinary、40、45、50 confirmed-positive FPR 均不超过 0.20；
  - 40、45、50 positive false-negative rate 均不超过 0.10；
  - ordinary decision coverage 不低于 0.70；
  - 三功率平均 decision coverage 不低于 0.40；
- 在可行候选中依次最大化：最差功率 coverage、平均功率 coverage、平均功率 confirmed-positive recall、普通 confirmed-positive recall、负 selective risk；
- 无可行解时该方法/种子失败，不放宽门槛。

U3 对每个域独立选择阈值，但使用相同单域 FPR/FN 约束；用于判断分域决策的 oracle 上限。

## 7. U2 正式晋级门

独立 early-development 上：

1. ordinary、40、45、50 confirmed-positive FPR 均不超过 0.20；
2. 50 dB recoverable-positive coverage 不低于 0.60；
3. 50 dB positive false-negative rate 不超过 0.10；
4. ordinary decision coverage 不低于 0.70；
5. 三功率平均 decision coverage 不低于 0.40，50 dB coverage 不低于 0.25；
6. ordinary及三功率最大 selective risk 不超过 0.15；
7. 50 dB confirmed-positive recall 不低于 0.1368；
8. ordinary confirmed-positive recall 相对 U0 下降不超过 0.05；
9. 至少 2/3 seeds 通过；
10. U2 三功率平均 coverage 不低于 U1；
11. U3 三功率平均 coverage 不低于 0.60，否则判定分域双阈值结构上限不足；
12. source/checkpoint/identity/locked-test 审计全部通过。

U3只有在独立early-development上同时满足各域FPR≤0.20、各功率正样本误判负≤0.10、coverage约束和selective risk≤0.15时，才能判为oracle selective通过。仅在threshold role存在可行解或只有coverage较高，不能触发下一阶段。

机制门与决策门必须同时通过。`recoverable-positive coverage` 不能单独作为成功指标。

## 8. 失败后的动作

- U2 失败、U3 在独立early完整通过：现有单视图分数不能无元数据推断可靠性，进入 stage2 标签条件 `exist/reliability` 双头。
- U2、U3 均失败：停止 selective/校准路线，保留 C2 排序证据，H2/Base-DB 作为二元决策基线。
- U2 通过：扩展五种子，之后才允许最后 stage 小学习率微调。

## 9. 产物

每种子保存 source/checkpoint hash、身份审计、阈值完整搜索、四变体 predictions、逐域 metrics、coverage-risk 曲线数据和 receipt；聚合 summary 保存逐种子门槛与最终决策。
