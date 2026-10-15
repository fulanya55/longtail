# Route A development-v5 作用层与 alpha 结构化消融预注册

日期：2026-08-01

## 1. 本轮唯一科学问题

development-v4 表明 full alpha=1.5 的 macro-AUPR、macro-F1 和 SawFM-present
增益稳定，但 MFT/VFT/ISRJ mean weak recall 只有 `+0.01691`，其中 MFT
`+0.03293`、VFT `+0.01898`、ISRJ `-0.00116`。

本轮只回答：

> C6→M5 方向的收益来自最后共享时序块、分类器，还是两者的固定组合；通过固定
> layer/alpha 分解或仅基于内部 validation 的自适应选择，能否在不损伤 ISRJ 的
> 前提下保持 macro 指标并恢复三个目标标签的一致弱分量增益？

## 2. 证据边界

- 使用全新 identity-isolated development-v5 train-only 数据；
- development-v4 不再评价任何新候选；
- 历史 locked-test、test-v2、密封 test-v3 均禁止访问；
- 不生成 test-v5；
- 本轮只产生 `development` 证据；
- 所有 SawFM-containing 样本继续保留；
- alpha/layer 网格和选择规则在生成 v5 结果前冻结。

## 3. 数据协议

- base sample seed：`44000000`；
- train sample seed：`[44000001,44025200]`；
- expected receiver_noise_seed：`[242677743,242702942]`；
- reserved test-v5 seed：`[47000001,47025200]`，本轮不生成；
- 25,200 样本、63 个非空组合；
- 与 v2/v3/v4 的 train 和全部冻结/密封测试声明区间不重叠；
- seeds：2039--2043；
- label-longtail、split、fit/validation/early/calibration/threshold 规则与 v4 相同。

## 4. 训练端点

所有端点共享初始化、batch 顺序和预算：

1. `C6_Joint`；
2. `M5_GradientIsolated`；
3. `MaskFMB`；
4. `MaskSinFM`；
5. `MaskMFT`；
6. `MaskVFT`；
7. `MaskISRJ`。

other-label 端点只用于构造结构化任务子空间控制，不参与候选选择。

## 5. 固定 layer/alpha 网格

令 `d = theta_M5 - theta_C6`，并严格分为：

- `d_shared`：仅 `temporal_backbone.0.*`；
- `d_classifier`：仅 `classifier.*`。

固定评价：

- Full：alpha `0.5, 1.5, 2.0`；alpha=1.0 由 M5 endpoint 表示；
- Shared-only：alpha `0.5, 1.0, 1.5, 2.0`；
- Classifier-only：alpha `0.5, 1.0, 1.5, 2.0`；
- Split：shared=1.5，classifier=`0.5,1.0`；
- 原候选：shared=1.5，classifier=1.5。

所有非目标 state 固定为 C6。

## 6. 自适应候选

`AdaptiveLayerAlpha` 只能在每个种子的内部 `validation` role 上，从上述固定网格
选择一个状态，不能使用 early/calibration/threshold：

1. 计算 common-five macro-AUPR；
2. 仅保留不低于该种子最佳 macro-AUPR `0.005` 的状态；
3. 在保留状态中最大化 MFT/VFT/ISRJ weak-vs-negative mean AUROC；
4. 并列时依次选择较小总 alpha、较小 classifier alpha、固定名称字典序；
5. 选择后在 early role 做一次正式 development 评价。

## 7. 结构化等范数控制

`OrthogonalTaskSubspace1p5`：

1. 将五个 other-label mask directions 分别单位范数化；
2. 固定等权求和；
3. 去除其在 SawFM direction 上的投影；
4. 重新缩放到与完整 SawFM direction 相同的全局范数；
5. 从 C6 以 alpha=1.5 应用。

该控制位于同一训练产生的任务方向子空间，避免 v4 高维高斯随机方向和跨初始化
方向的功能失稳。若正交残差范数低于 `1e-8`，本轮实现失败，不得改系数补救。

## 8. 主要指标

- common-five macro-AUPR；
- common-five macro-F1；
- target mean weak recall；
- MFT、VFT、ISRJ 各自 weak recall delta；
- target mean negative FPR；
- SawFM-present/absent macro-F1；
- 每标签 AUPR；
- 五种子配对 delta 和 95% seed-bootstrap CI；
- 每个方向的 shared/classifier/global norm。

## 9. 候选资格门槛

固定网格和 Adaptive 中的候选必须同时满足：

1. 5/5 seeds 输出有限；
2. macro-AUPR delta `>= -0.005`；
3. macro-F1 delta `>= +0.010`；
4. target mean weak recall delta `>= +0.020`；
5. MFT、VFT、ISRJ 三个 mean weak recall delta 均 `>= 0`；
6. target FPR `<= 0.20`；
7. 至少 4/5 seeds 的 target mean weak recall delta 为正；
8. SawFM-present macro-F1 delta `>0`；
9. SawFM-absent macro-F1 delta `>=-0.005`；
10. 三个目标标签 AUPR delta 均 `>=-0.010`。

## 10. 选择与晋级门槛

在所有 eligible candidates 中，按以下固定顺序选择：

1. 最大化三个目标标签 mean weak recall delta 的最小值；
2. 最大化 target mean weak recall delta；
3. 最大化 macro-F1 delta；
4. 选择较小总 alpha；
5. 名称字典序。

selected candidate 只有同时满足以下条件才晋级：

1. target mean weak recall delta `>= +0.025`；
2. 三个目标标签最小 mean weak recall delta `>=0`；
3. 相对 FullAlpha1p5 的最差目标标签 weak recall 至少改善 `+0.005`；
4. macro-F1 delta `>=+0.010`；
5. macro-AUPR delta `>=-0.005`；
6. target FPR `<=0.20`；
7. weak recall 和 macro-F1 均高于 `OrthogonalTaskSubspace1p5`；
8. 所有数据、模型和输出契约通过。

## 11. 决策

- `promote`：冻结 selected layer/alpha 规则，进入 fresh development 的通用
  PCGrad/CAGrad/MGDA/GradNorm 公平基线阶段；
- `revise`：存在 eligible candidate，但晋级门槛未过，保留层级机制证据并停止
  当前 alpha 网格；
- `stop`：没有 eligible candidate，停止参数外推改进路线，既有 test-v2 C 级
  结果仅作为固定协议内效果，不升级为可靠通用创新。

