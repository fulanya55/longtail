# 两条路线协议纠错与下一阶段实验方案

日期：2026-07-22  
状态：已执行完成  
环境：`D:\anaconda3\envs\PY10\python.exe`  
数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`  
locked test：禁止读取

## 1. 目标

本阶段不增加 prototype、参数头、distribution alignment 或复杂网络。只回答两个问题：

1. 在真正的标签频次长尾下，多标签 DB Loss 是否优于 ASL 和现有 CB-ASL？
2. 修复未标注有序截断和过强 weak augmentation 后，小权重伪标签是否能不破坏 B0_CONT？

## 2. 路线 A：真实 label-LT 损失基线

| 项目 | 设置 |
|---|---|
| 模型 | T0，单视图，不使用功率、蒸馏、ordering、prototype |
| protocol | label-LT |
| 初始 optimization pool | 1024 |
| achievable IF | 3，允许标签目标相对误差 15%；组合计数由整数优化求解 |
| development eval | 300 |
| epochs / patience | 20 / 20 |
| seeds | 2026、2027、2028 |
| loss | ASL、CB-ASL、DB Loss |
| selection | macro-AUPR |

判定指标：macro-AUPR、固定阈值 macro-F1/Hamming、每标签 AUPR，以及训练清单中频次最低标签的 AUPR。DB Loss 只有在三 seed 平均 tail AUPR 提升且 macro-AUPR 不低于 ASL 超过 0.01 时才保留。

## 3. 路线 B：协议纠错后的最小半监督矩阵

### 3.1 固定修复

- `max_unlabeled_samples` 使用确定性组合分层抽样，不再截取排序后的前 N 个索引；
- weak augmentation 噪声改为 18～30 dB；
- 正、负伪标签损失分别归一化，防止数量更多的一侧淹没另一侧；
- 同一 B0 best checkpoint 初始化所有 continuation；
- 不启用 feature consistency、prototype、dynamic threshold 或 distribution alignment。

### 3.2 实验矩阵

| 变体 | 说明 |
|---|---|
| B0_CONT | 纯监督 continuation |
| B1_U005 | B1，`lambda_u=0.05` |
| B1_U010 | B1，`lambda_u=0.10` |

共同设置：20% 标注，1024 个组合分层未标注样本，12 continuation epochs，5-epoch 等效 ramp，逐标签 calibration 阈值，同一 seed 2026。

### 3.3 离线审计

隐藏真值只在训练完成后用于统计，不参与阈值、损失或 checkpoint 选择。逐 epoch 保存：

- 每标签正/负伪标签 precision、recall、coverage；
- 总体正/负伪标签数量；
- Teacher 每标签概率分位数；
- 标注/未标注的标签先验、组合数量和平均标签基数。

保留条件：

1. 每标签正、负伪标签 precision 均不低于 0.90；
2. 任一标签/任一侧 coverage 不超过 30%；
3. 相对 B0_CONT，macro-AUPR 下降不超过 0.005；
4. Hamming 不恶化超过 0.01。

## 4. 边界

- 本阶段为四类 development pilot；
- 不读取 locked test；
- Power-LT、combo-LT 与 label-LT 结论分开表述；
- 若路线 B 的 B1_U005 仍明显退化，停止伪标签训练，转向无标签自监督预训练或冻结 backbone 的两阶段方案。

## 5. 执行记录

- 路线 A 固定标签头尾顺序为 `FMB -> SinFM -> MFT -> ISRJ`，确保三个 seed 的尾类均为 ISRJ；
- 路线 A 结果：`outputs/tapc/route_a_label_lt_losses/route_a_label_lt_loss_summary.json`；
- 路线 B 结果：`outputs/student_b/route_b_protocol_correction/seed_2026/route_b_protocol_summary.json`；
- 路线 B 隐藏真值没有进入训练、阈值生成或 checkpoint 选择，只对初始 B0 与最终 best teacher 做离线审计；
- 完整结论见 `docs/results/two_route_protocol_correction_results_20260722.md`。
