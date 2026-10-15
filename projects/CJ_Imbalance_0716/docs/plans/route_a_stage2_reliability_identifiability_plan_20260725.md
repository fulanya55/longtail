# 路线 A：Stage2 决策可靠性可辨识性筛选方案（2026-07-25）

## 1. 决策背景

上一阶段已经证明：在冻结 F2、S2、C2 与 A0 的条件下，仅依靠统一双阈值或按功率 oracle 双阈值，不能同时满足普通域覆盖率、功率域覆盖率、弱正漏检率和选择性风险约束。因此本阶段不再调分类阈值，也不解冻主干，而是先回答一个更基础的问题：

> 冻结的 Stage2 时序特征是否包含能够在独立身份上预测“A0 当前决策可能出错”的信息？

只有该问题得到肯定答案，才允许后续实现 `exist / reliability` 双头；否则停止 SawFM 决策扩展，继续保留 C2 作为排序分数、H2/Base-DB 作为二元决策基线。

## 2. 冻结来源与数据角色

- Python：`D:\anaconda3\envs\PY10\python.exe`
- 数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`
- 冻结主干：每个种子的 F2 `DelayedFusionMultiScaleIQClassifier`
- 冻结二元头：H2
- 冻结 SawFM 专家：上一阶段已选定的 S2 checkpoint
- 冻结排序与校准：C2 与 A0 visibility calibrator
- 冻结 A0 单阈值：沿用上一阶段 V2 阈值，不重新搜索
- 种子：2041、2042、2043
- 功率域：40、45、50 dB
- locked test：不使用

身份与角色继续沿用上一阶段：

1. `fit`：普通 calibration、root 配对样本、各功率 fit 配对样本；只用于拟合 probe。
2. `validation`：普通 threshold 与各功率 threshold 配对样本；只用于选择正则强度和拒识阈值。
3. `early`：普通 early-stop 与各功率 development 配对样本；只用于独立评价。

所有功率视图按 `receiver_noise_seed` 身份管理，禁止把同一身份的不同功率视图当成独立随机样本跨角色使用。

## 3. 监督目标

主任务为 `decision_error`：

```text
error = 1[A0(probability, frozen_threshold) != SawFM ground truth]
```

这是部署时可靠性头需要预测的事件。附加报告 `positive_miss`（只在真实正样本内统计）用于解释弱正漏检机制，但不单独作为模型晋级依据，避免把“已知正样本”这一 oracle 条件误当成部署能力。

## 4. 冻结特征与对照

所有 probe 都不反向传播到 F2、H2、S2 或 A0。

| 编号 | 输入 | 用途 |
|---|---|---|
| R0 | 常数错误率 | 无特征下界 |
| R1 | A0 到冻结阈值的距离 | 纯置信度基线 |
| R2 | 10 维 visibility 统计 + A0 margin | 已有统计特征基线 |
| R3 | Stage2 时序描述 + A0 margin | 检验时序图是否比统计量保留更多可靠性信息 |
| R4 | visibility + Stage2 时序描述 + A0 margin | 候选完整可靠性 probe |

Stage2 时序描述由冻结的 `[B,64,375]` 特征和 S2 evidence map 构造，不引入新可训练卷积：

- S2 evidence map；
- 通道 RMS、绝对值均值、通道标准差三条时序；
- 25 个时间分箱的 mean / max / std；
- 全局分位数；
- 一阶差分统计；
- 1、4、16、32 点归一化自相关。

R1--R4 均使用 L2 Logistic Regression。正则强度 `C in {0.01, 0.1, 1, 10}` 只在 validation 上选择；特征标准化参数只由 fit 估计；拟合时使用域平衡与类别平衡权重。

## 5. 指标与选择性决策仿真

主指标：

- `decision_error AUROC`
- `decision_error AUPR`，并报告对应错误率常数下界
- validation 到 early 的 AUROC 泛化落差
- 分域 AUROC/AUPR：root、40、45、50 dB

附加机制指标：正样本中的漏检 AUROC/AUPR。

对 R1--R4，使用 validation 选择“预测错误概率”的拒识阈值：预测错误概率高于阈值时输出 unknown。阈值必须先满足：

- 普通 validation 决策覆盖率不低于 0.70；
- 平均功率 validation 决策覆盖率不低于 0.40。

在可行候选中先最小化各域最大选择性风险，再最大化平均覆盖率。选定阈值固定后在 early 上报告覆盖率、选择性风险、正样本漏检率和负样本 FPR。

## 6. 预注册晋级门

只有 R4 在至少 2/3 个种子同时满足以下条件，才允许下一阶段训练真正的 `exist / reliability` 双头：

1. early 聚合 `decision_error AUROC >= 0.65`；
2. 相对 R2 的 early AUROC 增量 `>= 0.03`，证明时序信息有独立贡献；
3. validation 到 early 的 AUROC 落差 `<= 0.15`；
4. validation 能找到满足覆盖率约束的拒识阈值；
5. early 普通覆盖率 `>= 0.70`、平均功率覆盖率 `>= 0.40`；
6. early 各域最大选择性风险 `<= 0.15`；
7. locked test 未使用且身份审计通过。

若第 1--3 条失败，结论为 Stage2 对可靠性不可辨识，停止设计可靠性头。若第 1--3 条通过但第 4--6 条失败，说明信息存在但当前线性读出不足，只允许下一轮尝试小容量时序可靠性头，不允许解冻主干。

## 7. 产物

- 程序：`experiments/run_route_a_stage2_reliability_identifiability.py`
- 单元测试：`tests/test_stage2_reliability_identifiability.py`
- 每种子：特征审计、模型选择记录、early 预测、分域指标、选择性阈值与 receipt
- 汇总：`outputs/tapc/route_a_stage2_reliability_identifiability/route_a_stage2_reliability_identifiability_summary.json`
- 结果分析：`docs/results/route_a_stage2_reliability_identifiability_results_20260725.md`
