# Route A development-v3 等总预算两阶段解耦基线预注册

## 问题

现有 `Focal-36 -> Frozen Linear BCE` 使用 36 epoch 表征训练后再训练分类头，因总训练预算超过单阶段 Focal-36，不能作为最终公平的 two-stage 基线。该实验只检验“等总预算的解耦是否已能解释当前候选的收益”，不修改任何物理专家。

## 固定协议

- 数据、五个种子、F2 初始化、partial shared scope、RawSaw SawFM 路由、Platt 校准和 `FPR <= 0.20` 阈值协议：完全继承 development-v3 主流基线；
- 两阶段：Focal 表征训练 24 epoch；冻结 `forward_features`；随机初始化 `Linear(128,6)`，BCE 训练 12 epoch；总计恰为 36 epoch；
- 表征阶段与分类头阶段均运行完整预算，不允许早停缩短预算；validation macro-AUPR 仅用于各自阶段 checkpoint 选择，target weak AUROC 仅破同分；
- 优化器、批量、正则和 RawSaw 超参数沿用已完成的公平基线；不搜索损失、学习率、阶段比例或阈值；
- 对照为同一 F2 初始化、同一数据角色、同一总 36 epoch 的单阶段 `Focal-36`。

## 记录

每个 seed 保存两阶段 checkpoint、history、特征归一化、校准器、阈值、预测、角色哈希和总 epoch receipt。test-v3 必须保持 sealed。

## 解释规则

该基线只回答“解耦训练是否足以解释弱分量收益”。若它优于或接近统一候选，应收缩“物理专家是唯一来源”的主张；若它弱于候选，也仍不能单独证明物理模块，必须与模块删除、容量和破坏对照共同报告。
