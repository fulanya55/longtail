# Route A：标签内可见性时序专家（VTSE）接回试验

## 创新假设

stage2/stage3 筛选表明，F2 final temporal 统计对 MFT/VFT/ISRJ 弱正—负样本
具有 0.7606 AUROC，而普通 C6 分类边界只有 0.6107。由此提出：六类复合干扰
中的主要长尾不是单一标签频次长尾，而是**标签内可见性长尾**。普通 DB/BCE
使用强正和弱正共同拟合一个边界，强正主导后会掩盖弱证据。

VTSE 将标签存在分类与标签内弱可见性识别解耦：冻结 C6 负责全局标签存在，
独立专家只学习“弱正 vs 标签缺失负样本”，再以受限方式修正不确定样本。

## 网络与数据

- 主干、C6 头全部冻结。
- temporal map `[B,128,T]` 提取 mean/std/max/top-8 mean，得到 512 维。
- 三输出线性 VTSE，对应 MFT/VFT/ISRJ；直接复用已通过五种子可辨识性门槛的
  T_FinalTemporal probe。
- VTSE 训练使用完整 optimization role；强正样本被 mask，仅用弱正和负样本。
- optimization 内 validation 用于固定 expert score 的方向、中心和尺度。

## 安全接回

对目标标签：

`z=(expert_score-class_balanced_center)/robust_scale`

`uncertainty=4*sigmoid(base_logit)*(1-sigmoid(base_logit))`

`residual=0.35*tanh(z)*uncertainty`

`final_logit=base_logit+residual`

非目标标签完全不变。该设计保证：

- 单次修正硬限制在 ±0.35 logit；
- C6 越自信，专家作用越小；
- expert score 低的难负样本得到负修正，弱正样本得到正修正；
- 不额外学习类别 bias，不在 early/calibration/threshold 上训练专家。

## 评估协议

- 五种子 2039--2043，development-v2。
- calibration role：每变体独立标量温度。
- threshold-selection role：FPR<=0.20 下最大化弱召回。
- early role：报告五标签 macro-AUPR/F1、目标弱召回、弱 AUROC、FPR、强正召回。
- test-v2 与历史 locked test 禁止读取。

## 晋级门

VTSE 相对冻结 C6 必须全部满足：

1. 五标签 macro-AUPR >= -0.005；
2. macro-F1 >= +0.010；
3. MFT/VFT/ISRJ 平均弱召回 >= +0.030；
4. 目标弱正—负 AUROC >= +0.015；
5. 目标平均 FPR <= 0.20；
6. 目标强正召回不劣于 -0.010；
7. 至少 4/5 种子弱召回为正增益；
8. SawFM-present 五标签 macro-F1 不劣于 -0.005；
9. 最大残差不超过 0.350001。

通过后，VTSE 成为 development-v2 候选创新模型，随后冻结全部组件并仅执行一次
test-v2；失败则保留“标签内可见性可辨识性”发现，但重新设计接回函数。

