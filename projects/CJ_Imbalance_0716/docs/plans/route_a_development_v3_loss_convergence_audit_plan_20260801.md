# Route A development-v3 传统损失收敛审计预注册

## 科学问题

主流损失公平基线在 12 epochs 时全部把最佳点落在预算上限。需要检验 F3 的 ISRJ 专项优势是否只是传统损失欠训练造成，而不是周期专家提供了不可由训练时长替代的额外证据。

## 冻结边界

- 只使用 development-v3，SHA256 固定为 `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`；
- 种子固定为 2039--2043，完全复用现有 role indices；
- test-v3 保持 `access_authorized=false, analysis_allowed=false, used=false`；
- F2 初始化、partial-scope、batch 顺序、RawSaw checkpoint、校准和阈值协议全部不变；
- 不搜索 gamma、clip、DB 参数、学习率或阈值目标。

## 审计对象

只延长 12-epoch 结果中表现最强且相邻的三种传统损失：

- DB Loss；
- ASL，`gamma_pos=0, gamma_neg=3, clip=0.05`；
- Focal，`gamma_pos=0, gamma_neg=3`。

原 12-epoch checkpoint 只作为冻结参考，F3 只作为冻结候选参考。

## 训练协议

- 每个方法仍从同 seed 的原始 F2 状态重新开始，不从 12-epoch checkpoint 续训；
- AdamW；classifier LR `5e-4`，shared LR `5e-5`，weight decay `1e-4`；
- batch 64，最大 36 epochs，patience 8，gradient clip 5；
- train seed 仍为 `seed+73000`，保证前 12 epochs 的 batch 顺序与原实验一致；
- 总损失仍为 common-five loss + `0.2 * SawFM loss`；
- 保存逐 epoch history、最佳 checkpoint、训练时间和哈希。

## 评价与判定

1. 每个长预算模型独立使用 calibration role 做 Platt 校准，并在 threshold role 的逐标签 `FPR<=0.20` 下最大化弱召回；
2. 按 mean macro-AUPR 在三个长预算损失中选最强基线，target weak AUROC只破完全同分；
3. 汇报每个方法相对原 12-epoch版本的变化和最佳 epoch；
4. 若最佳 epoch仍为36，判定该方法“未证明平台化”，不得宣称传统基线充分收敛；
5. 若最佳长预算基线达到或超过 F3 的 ISRJ AUPR、弱召回或弱 AUROC关键门槛，则提升它为新基线并收缩 F3 主张；
6. F3 相对最强长预算基线的目标门槛保持不变：ISRJ AUPR为正、弱召回 `>=+0.030`、弱 AUROC `>=+0.010`、两者至少4/5种子为正、ISRJ FPR `<=0.20`；
7. 本审计不授权 test-v3，也不代替两阶段分类头和同容量 MLP 对比。

## 证据解释

- 若长预算基线平台化且 F3 门槛仍通过：增强“结构专家贡献不能由传统损失与训练时长解释”的证据；
- 若长预算明显改善但 F3仍通过：保留 F3，同时以长预算方法作为后续新基线；
- 若 F3门槛失败：优先承认传统训练不足，停止夸大周期专家贡献；
- 若仍未平台化：继续将收敛性列为证据缺口，不允许开启 test-v3。
