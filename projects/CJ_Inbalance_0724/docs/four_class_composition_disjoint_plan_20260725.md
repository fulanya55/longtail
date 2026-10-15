# 四类组合不相交机制验证计划（2026-07-25）

## 目标

在 `{FMB, SawFM, MFT, ISRJ}` 四类任务上，先验证以下机制是否能够提高未见标签组合的泛化，而不把结果外推为六类结论：

- 标签关系约束；
- 组合均衡的正负原型；
- 匹配数量的困难负样本损失；
- 原型正负分离；
- 同身份25/30/35 dB视图一致性。

骨干固定为此前分类特征提取表现较强的 `legacy_mlcnn`，推理结构保持统一的 `256 -> 4` 分类头。原型、关系和一致性均只在训练阶段使用。

## 数据边界

- 只读取 `power_sweep_9level_v1` 中25、30、35 dB目录下的 `closedset_6class_train.mat`。
- 不读取任何 `closedset_6class_test.mat`。
- 删除所有含SinFM或VFT的身份，而不是只丢弃对应标签列。
- 剩余6000个身份、15种非空组合、每组合400个身份。
- 三个功率视图通过原始样本索引和 `receiver_noise_seed` 对齐，同一身份的三个视图必须属于同一数据角色。

## 首轮组合留出

首轮使用fold 0，完全未见组合为：

1. FMB；
2. SawFM+MFT；
3. SawFM+ISRJ；
4. FMB+MFT+ISRJ；
5. FMB+SawFM+MFT+ISRJ。

其余10种组合内部每组合拆为280/40/40/40个身份，分别用于训练、早停、阈值校准和seen-confirm。未见组合每种400个身份全部只用于最终开发评估。

## 对照

- `baseline`：MLCNN + BCE，训练使用三个功率视图。
- `prototype_relation`：相同MLCNN和分类头，在训练时增加原型、关系、困难负样本、正负原型分离和跨功率一致性损失。

## 指标与边界

主指标为按身份平均三个功率概率后的 unseen-combination subset accuracy，同时报告：

- seen/unseen subset gap；
- Macro-AUPR、Macro-F1、Hamming loss；
- 四类逐类AUPR、recall和FPR；
- 逐未见组合exact accuracy；
- 25/30/35 dB逐功率结果；
- 同身份跨功率概率标准差。

首轮若增强模型没有改善unseen subset，或通过牺牲ISRJ recall取得表面提升，则停止直接迁移到六类。四类通过也只能证明机制具备继续迁移验证的资格，不能声称解决了六类VFT组合shortcut。
