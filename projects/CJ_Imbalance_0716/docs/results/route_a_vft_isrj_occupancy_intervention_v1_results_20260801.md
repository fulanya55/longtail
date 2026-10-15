# Route A：VFT--ISRJ 宽度×副本受控干预结果（v1）

## 结论

预注册的“**通用 ISRJ 占空/重置**会单调推高 VFT 假阳性”假设**未通过**，因此不
授权构建或训练一个泛化 occupancy 专家。但实验给出了更精确且一致的机制结论：

> 在同一 receiver identity、同一组合、同一其余分量和噪声流下，ISRJ **窗口宽度**
> 从 10--15 us 变为 22--30 us，会稳定、大幅推高冻结 Q4 的 VFT 缺失/ISRJ 存在
> hard-negative 概率和 FPR；单独把副本数从 3--4 变为 6--8 则方向相反。

因此现有 VFT OOD 失败不能再笼统归因为“ISRJ 越密越混淆”。更可证伪的下一假设是
**窗口宽度导致的局部时频片段/相位窗口状态**与 VFT ridge 发生混淆；副本数增加可能
破坏该类 VFT-like 连续结构。该解释是由本次受控生成器干预支持的机制推断，不是已验证
的可部署分类器。

## 预注册合同与边界

| 项目 | 结果 |
|---|---|
| 新鲜基础 seed | `178000000` |
| 制度 | `W10_15_R3_4`、`W22_30_R3_4`、`W10_15_R6_8`、`W22_30_R6_8` |
| 每制度 | 63 个组合 × 20 = 1,260 条；仅 train-named 数据 |
| 成对控制 | 四制度 receiver id 与标签完全一致；ISRJ 缺失记录的 32 条 I/Q 探针逐字节一致 |
| 历史 train 身份 | 全部零交集 |
| 评价 | 五个固定模型 seed 的完整 Q0--Q5 冻结重建；分析仅使用 Q4 |
| 训练/校准/阈值 | 均未发生；Q4 每 seed 阈值跨制度完全相同 |
| 历史 locked test/test-v2/test-v3 | 均未访问；`test_v3_used=false` |

hard-negative 固定为 `VFT=0 AND ISRJ=1`，每制度 320 条。区间是 receiver 配对
bootstrap（每模型 seed 2,000 次）与五个固定模型 seed 的汇总描述，不将模型 seed
误称为新增独立训练。

## 成对干预结果

参考制度是 `W10_15_R3_4`。FPR 为冻结 Q4 阈值下的 VFT FPR。

| 干预 | VFT 概率均值差，5 seed 汇总 95% CI | FPR 差，5 seed 汇总 95% CI | 正方向 seed | 判读 |
|---|---:|---:|---:|---|
| 宽度：`W22_30_R3_4 - ref` | `+0.19546` `[+0.16287,+0.21861]` | `+0.32125` `[+0.25438,+0.36125]` | 5/5 | 强、稳定的宽度效应 |
| 副本：`W10_15_R6_8 - ref` | `-0.03607` `[-0.04590,-0.02631]` | `-0.05688` `[-0.07375,-0.03875]` | 0/5 | 与通用 occupancy 方向相反 |
| 联合：`W22_30_R6_8 - ref` | `+0.17695` `[+0.13727,+0.21664]` | `+0.30250` `[+0.21875,+0.39313]` | 5/5 | 宽度效应仍在，但低于纯宽度干预 |

纯宽度干预的五个逐 seed FPR 差为 `+0.34063, +0.35938, +0.35000, +0.18750,
+0.36875`；联合高宽度/高副本为 `+0.28125, +0.36563, +0.26875, +0.15313,
+0.44375`。这排除了单个冻结模型或阈值漂移造成的表象。

## 对预注册门槛的判定

| 门槛 | 结果 |
|---|---|
| 宽度概率差 `>=0.020` 且至少 4/5 正 | 通过，`+0.19546`、5/5 |
| 副本概率差 `>=0.020` 且至少 4/5 正 | **失败**，`-0.03607`、0/5 |
| 联合 FPR 差 `>=0.050` 且至少 4/5 正 | 通过，`+0.30250`、5/5 |
| 成对合同、冻结阈值、test-v3 密封 | 通过 |

整体 `occupancy_causal_screen_passed=false`。这是有效的机制否证，不是候选性能失败；
本轮不改变 `H1_B2_MFTSharedDecision_VFTRidgeFusion_v2`，不重新分析历史 OOD 以挑选
模型，也不授权 test-v3。

## 后续动作

停止“惩罚 ISRJ 总占空/副本数”的路线。若继续，需要先在**新的、未被本轮使用的**
身份隔离 development 数据上预注册一个 `ISRJ window-state` 可识别性屏幕：固定的
宽度/相位重置状态特征必须在 VFT 缺失/ISRJ 存在 hard negatives 上可分，并与等容量
非物理特征比较。只有该屏幕通过，才允许在另一份新鲜 development 数据中以有界、
等容量残差训练窗口状态专家；不得用本轮四制度数据拟合该专家。

## 产物

- 方案：[route_a_vft_isrj_occupancy_intervention_v1_plan_20260801.md](../plans/route_a_vft_isrj_occupancy_intervention_v1_plan_20260801.md)
- 生成器：`experiments/generate_route_a_vft_isrj_occupancy_intervention_v1.m`
- 数据审计：`outputs/tapc/route_a_vft_isrj_occupancy_intervention_v1_generation/identity_parameter_pair_audit.json`
- 冻结汇总：`outputs/tapc/route_a_vft_isrj_occupancy_intervention_v1_frozen/route_a_vft_isrj_occupancy_intervention_v1_frozen_summary.json`
- 分制度逐 seed 预测：`outputs/tapc/route_a_vft_isrj_occupancy_intervention_v1_frozen/<regime>/seed_<seed>/Q4_VFTAlignedRidgeFusion/predictions.npz`
