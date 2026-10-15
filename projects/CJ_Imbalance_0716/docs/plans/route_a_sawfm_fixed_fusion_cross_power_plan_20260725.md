# 路线 A：SawFM 固定融合跨功率确认方案（2026-07-25）

## 1. 目标

上一轮已经证明冻结的 S2 SawFM 专家可以显著改善 45 dB 弱可见度识别，且固定凸融合优于 9 参数动态路由。本轮不再训练路由、不按种子选择 alpha，只验证一个可复现的问题：统一 `alpha=0.80` 能否在独立身份、多个功率档和不同复合组合上稳定优于 H2。

本轮通过前，不解冻 F2/H2/S2，不做 locked-test 评价。

## 2. 冻结结构

- F2 DFMS-IQ 主干：冻结。
- H2_PosBCE 六类分类头：冻结。
- S2_TeacherHardNeg stage2 SawFM 专家：冻结。
- 专家 Platt 映射：直接读取上一轮路由 fit 数据得到的每种子参数，不重新拟合。
- SawFM 融合：`z = (1-alpha) * z_H2 + alpha * z_expert_cal`。
- 其余五类 logits 完全保持 H2。

## 3. 对照组

- `C0_H2`：原 H2。
- `C1_A075`：固定 alpha=0.75，敏感性下界。
- `C2_A080`：固定 alpha=0.80，唯一晋级候选。
- `C3_A085`：固定 alpha=0.85，敏感性上界。

不得按种子、功率或最终指标重新选择 alpha。

## 4. 数据角色

- 种子：2041、2042、2043。
- 普通六类 calibration、threshold-selection、early-development 角色保持原 split。
- 跨功率评价身份：每种子的 early-development role 内，SawFM+至少一种扫频干扰的正样本，与其他标签组合完全一致但 SawFM 缺失的 hard negatives。
- 功率档：35、40、45、50 dB。
- 35/45 dB：S2 训练未使用的功率档，属于 held-out-power 验证。
- 40/50 dB：S2 训练使用过功率条件，但评价身份完全独立，属于 unseen-identity 验证。
- 同一 `receiver_noise_seed` 的功率视图只作为同一身份的重复条件评价，不作为独立样本扩充统计量。
- locked test 始终冻结。

## 5. 指标

### 普通六类域

- macro AUPR、macro F1、Hamming loss。
- SawFM AUPR、SawFM negative FPR、SawFM 弱正召回。
- 温度仅由 calibration role 拟合；阈值仅由 threshold-selection role 在 negative FPR≤0.20 下选择。

### 跨功率匹配域

- 每档 SawFM AUROC、AUPR。
- 使用普通域阈值计算 SawFM positive recall、weak-positive recall 和 hard-negative FPR。
- 相对 H2 的逐种子、逐功率增益。
- 按复合标签数量、扫频共存类型分层的 AUROC/AUPR。
- 四档功率 AUROC 的均值、最差值和标准差。

## 6. 晋级门槛

统一 alpha=0.80 只有同时满足以下条件，才允许进入最后一个多尺度 stage 的小学习率联合微调：

1. 四档中至少三档的三种子平均 AUROC 高于 H2；
2. 每一档至少 2/3 种子 AUROC 高于 H2；
3. 四档平均 AUROC 增益不低于 0.15；
4. 四档平均 AUROC 不低于 0.65，最差功率档 AUROC 不低于 0.58；
5. 四档平均 AUROC 的跨功率标准差不高于 0.08；
6. 普通域 macro AUPR 相对 H2 不低于 -0.003；
7. 普通域 SawFM AUPR 相对 H2 不低于 -0.005；
8. 普通域 SawFM FPR 增量不高于 0.02；
9. 至少 75% 的可统计组合/规模分层 AUROC 不劣于 H2；
10. 身份交叉为 0，locked test 未使用，检查点和预测审计完整。

若性能门槛通过但种子 2043 或部分组合仍失稳，下一步不立即解冻，而是先对 S2 做多种子 teacher-student 稳定性修复。若全部通过，联合微调使用主干最后 stage 学习率≤`1e-5`、专家/融合头学习率≤`1e-4`，并保留冻结 C2 为并行对照。

## 7. 产物

保存每种子/每变体的普通域校准、阈值、预测、指标；每功率的正负分数、阈值预测、弱掩码；逐功率与逐分层结果；split/identity/checkpoint 哈希；run receipt；三种子总汇总、晋级判断和结果技术文档。
