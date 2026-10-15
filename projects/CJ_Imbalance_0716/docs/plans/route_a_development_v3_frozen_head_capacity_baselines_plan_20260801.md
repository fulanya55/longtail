# Route A development-v3 冻结表征分类头容量基线预注册

## 问题

F3 的 ISRJ收益是否可以由“冻结强 Focal表征后重新拟合分类边界”或“增加普通 MLP非线性容量”解释，而无需周期结构专家？

## 冻结对象

- 每个 seed使用已选定的 `Focal-36` checkpoint；
- 完全冻结其128维 `forward_features` 表征；
- 冻结 RawSaw路由；
- F3保持冻结参考；
- development-v3角色、校准、阈值和 test-v3封存边界不变。

## 分类头

- H0：原 Focal-36，冻结参考；
- H1：冻结表征 + 重新初始化 `Linear(128,6)`，BCE；
- H2：冻结表征 + 重新初始化 `Linear(128,6)`，DB Loss，两阶段解耦基线；
- H3：冻结表征 + 六个独立 `128→4→1` GELU MLP，DB Loss，普通非周期容量基线。

所有新分类头只接收共享128维特征，不接收周期统计、原始时序位置、JNR、标签组合或 F3分数。

## 训练

- AdamW，初始 LR `1e-3`，weight decay `1e-4`；
- batch 128，60 epochs完整 cosine，最低比例0.01；
- 损失仍为 common-five + `0.2 * SawFM`；
- 相同 seed使用同一初始化/批顺序家族；
- 按 validation common-five macro-AUPR选择最佳 checkpoint，target weak AUROC破同分；
- 保存特征归一化统计、checkpoint、history、校准、阈值、预测和哈希。

## 判定

- 在 H0--H3 中按 mean macro-AUPR选择最强普通分类器基线；
- 若 H2/H3达到 F3 的 ISRJ AUPR、弱召回或弱 AUROC，则收缩“周期证据必要”主张；
- 否则 F3相对最强普通头须满足：ISRJ AUPR为正、弱召回 `>=+0.030`、弱 AUROC `>=+0.010`、弱召回和弱 AUROC至少4/5正向、FPR `<=0.20`；
- 本轮只证明分类边界/普通容量不能解释收益，不代替 block-shuffle周期消融或外部/OOD验证。
