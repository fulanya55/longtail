# Route A development-v6 通用多任务优化公平基线预注册

日期：2026-08-01

## 1. 科学问题

冻结 `FullAlpha1p5 = C6 + 1.5*(M5-C6)`，不再修改 alpha 或作用层。本轮只回答：

> FullAlpha1p5 是否显著优于简单 M5、SawFM loss weight、普通 midpoint，以及
> PCGrad、MGDA、CAGrad、GradNorm 等通用多任务优化基线？

若不能优于最强公平基线，创新声明必须降级为“特定协议下有效的配对 task-vector
技巧”，不能称为通用多任务优化创新。

## 2. 证据边界

- fresh identity-isolated development-v6 train-only；
- 不在 v5 上训练或评价新方法；
- 历史 locked-test、test-v2、密封 test-v3 禁止访问；
- 不生成 test-v6；
- 只产生 `development` 证据；
- 所有候选、算法公式、超参数和门槛在结果前冻结。

## 3. 数据与训练公平性

- base sample seed `54000000`；
- sample seed `[54000001,54025200]`；
- expected receiver_noise_seed `[252677743,252702942]`；
- seeds 2039--2043；
- 相同 F2 初始 checkpoint、split、manifest、batch 顺序、epochs、patience、
  learning rate、weight decay、gradient clip、early-stop、calibration 和 thresholds；
- trainable scope：`temporal_backbone.0` 与 classifier；
- task 1：common-five BCE mean；
- task 2：`0.2 * SawFM BCE`；
- 所有 SawFM 共现样本保留。

## 4. 正式方法

1. `C6_Joint`：`L_common + 0.2 L_saw`；
2. `M5_GradientIsolated`：`L_common`；
3. `LossWeight0p1`：`L_common + 0.1 L_saw`；
4. `ModelAverage0p5`：C6/M5 midpoint；
5. `PCGradTwoTask`：对 common 和 scaled SawFM 两任务做对称冲突投影后求和；
6. `MGDATwoTask`：解析求两任务最小范数凸组合，并固定乘 2 使等权点与 C6 梯度尺度一致；
7. `CAGrad0p4TwoTask`：两任务 simplex 上固定 101 点求解，`c=0.4`，使用 conflict-averse 合成方向；
8. `GradNormTwoTask`：初始权重 `[1,1]`，alpha=`1.5`，weight lr=`0.025`，每步归一化权重和为 2；
9. `FullAlpha1p5`：冻结 task-vector 候选。

所有梯度方法使用 common 与 `0.2*SawFM` 两个任务，不引入逐标签额外搜索。

## 5. 机制与成本记录

- common/Saw gradient cosine、冲突 batch 比例；
- MGDA 权重；
- CAGrad simplex 权重；
- GradNorm task weights；
- nonfinite、gradient clipping；
- 每方法 train seconds；
- FullAlpha1p5 总训练成本按 C6+M5 两个端点之和计算；
- 推理成本全部相同。

## 6. 候选效果门槛

FullAlpha1p5 相对 C6：

1. macro-AUPR delta `>=-0.005`；
2. macro-F1 delta `>=+0.010`；
3. target mean weak recall delta `>=+0.025`；
4. 三个目标 mean weak recall delta 均 `>=0`；
5. target FPR `<=0.20`；
6. positive target weak delta 至少 4/5；
7. SawFM-present delta `>0`；
8. SawFM-absent delta `>=-0.005`；
9. 三个目标 AUPR delta 均 `>=-0.010`；
10. 5/5 seeds 输出有限。

## 7. 最强基线选择

排除 `FullAlpha1p5` 后，基线只有满足以下条件才进入 strongest-baseline 排名：

- 5/5 输出有限；
- macro-AUPR delta `>=-0.005`；
- target FPR `<=0.20`；
- SawFM-absent macro-F1 delta `>=-0.010`。

固定排序：

1. 最大化三个目标 mean weak recall delta 的最小值；
2. 最大化 target mean weak recall delta；
3. 最大化 macro-F1 delta；
4. 名称字典序。

## 8. 方法优势门槛

候选相对 strongest baseline 必须同时满足：

1. macro-AUPR delta `>=-0.003`；
2. macro-F1 和 target weak recall 均不得低于 strongest baseline `-0.003`；
3. 至少一个主要指标优势达到：macro-F1 `+0.005` 或 target weak recall `+0.005`；
4. 上述达到 `+0.005` 的指标，其五种子 paired bootstrap 95% CI 下界 `>0`；
5. target FPR 不高于 strongest baseline `+0.01`；
6. SawFM-present 和 absent 均不劣于 strongest baseline `-0.005`；
7. 所有证据边界通过。

## 9. 决策

- `promote`：候选效果门槛和方法优势门槛均通过，进入分布/架构鲁棒性阶段；
- `downgrade`：候选本身有效，但不能优于 strongest baseline，停止“优化方法创新”
  声明，只保留配对 task-vector 技巧；
- `stop`：候选效果门槛也失败，停止该创新路线的后续开发。

