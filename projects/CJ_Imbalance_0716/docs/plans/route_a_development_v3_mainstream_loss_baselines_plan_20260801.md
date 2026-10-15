# Route A development-v3 主流损失公平基线预注册

## 科学问题

在相同 F2 初始化、partial-scope、训练预算、RawSaw 路由、校准和阈值协议下，传统标签频次重平衡或难例损失能否达到冻结 F3 周期融合头的 ISRJ 弱证据收益？

本轮只比较损失，不同时改变分类器结构。两阶段解耦和普通 MLP将在下一项独立预注册中执行。

## 数据与冻结边界

- development-v3 SHA256：`b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`；
- 种子：2039--2043；
- 完全复用现有 `role_indices.npz`，不重新划分；
- test-v3保持封存，禁止访问；
- RawSaw使用已冻结 checkpoint，对所有损失候选只替换 SawFM logit；
- F0 C4与 F3只作为冻结参考，不重新训练或调参。

## 共同网络和训练预算

- 初始化：同 seed `F2_DFMS_Full`；
- trainable shared scope：`temporal_backbone[-1]`，53,376 参数；
- trainable classifier：`Linear(128,6)`，774 参数；
- AdamW，classifier LR `5e-4`，shared LR `5e-5`，weight decay `1e-4`；
- batch 64；最多 12 epochs；patience 4；gradient clip 5；
- 所有候选使用相同 train seed `seed+73000` 和相同 batch 顺序；
- 总损失：common-five loss + `0.2 * SawFM loss`；
- 选模：validation common-five macro-AUPR，target weak AUROC只作同分破局。

## 损失候选

- L0：BCE；
- L1：pos-weight BCE，复用已冻结 C6 checkpoint作为协议锚点；
- L2：Focal，`gamma_pos=0, gamma_neg=3`；
- L3：ASL，`gamma_pos=0, gamma_neg=3, clip=0.05`；
- L4：Distribution-Balanced Loss，使用项目固定参数；
- L5：Class-Balanced BCE，effective-number `beta=0.99`；
- L6：Logit Adjustment BCE，fit role先验、`tau=1`，训练时将 `log(p/(1-p))` 加到 logits，推理使用原始 logits；
- R0：冻结 C4同容量无可见性参考；
- R1：冻结 F3对齐周期融合参考。

除 pos-BCE复用已完成的同协议 C6外，其余候选均从同一 F2状态重新训练。禁止针对单个损失搜索 gamma、clip、beta、tau或学习率。

## 统一评价

- 对每个候选加载同 seed冻结 RawSaw，并只替换 SawFM logit；
- 每个候选独立使用 calibration role做六标签 Platt校准；
- threshold role在逐标签负样本 FPR不超过0.20下最大化弱召回；
- early role只评价；
- 保存六标签 macro-AUPR/F1、每标签 AUPR、弱/强召回、FPR、弱 AUROC、SawFM有/无子组。

## 判定

1. “最强传统损失”只按五种子 mean macro-AUPR选择，target weak AUROC破完全同分；
2. 若 F3 相对最强传统损失的 ISRJ AUPR、弱召回和弱 AUROC均为正，且弱召回至少 +0.030、弱 AUROC至少 +0.010、4/5种子为正，则支持“传统损失不足而周期专家提供额外证据”；
3. 若任一传统损失达到或超过 F3，则必须把它提升为新的公平基线，并收缩创新主张；
4. 无论结果如何，不调整 F3或损失超参数；
5. 本轮不代替最终统一模型的 macro晋级门，也不授权 test-v3。

## 产物

保存每方法 checkpoint、history、calibration、thresholds、predictions、metrics、训练时间、参数量、数据/role/F2/RawSaw哈希、逐种子receipt、汇总与正式报告。

