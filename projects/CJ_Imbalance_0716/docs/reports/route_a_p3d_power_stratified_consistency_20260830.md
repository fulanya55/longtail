# Route-A P3d：分层功率一致性门控筛查（2026-08-30）

## 状态

**completed（3-seed 探索）；部分支持，未冻结。** 在 P3c 五种子结果显示低、高功率收益方向相反后，本轮分别只对低功率（10 dB）或高功率（50 dB）配对分支施加一致性约束，验证功率层梯度是否冲突。

## 固定协议

- F2_DFMS_Full/dfms_iq，参数 257,511，真实 I/Q `[B,2,3000]`；30 dB 拟合 4,096、验证 2,048。
- train-only 未见身份功率开发集，每功率 256；未打开 test/locked test。
- AdamW `1e-4/1e-4`，batch=16，2 epochs，梯度裁剪 5.0，阈值 0.5，seeds 2039/2040/2041。
- `λ=0.1`，`high_only` 只约束 50 dB 与 30 dB 的输出，`low_only` 只约束 10 dB 与 30 dB 的输出；普通 BCE 作为同 split 对照。

## 结果

### high-only

| 方法 | 35 dB 组合率 | 40 dB 组合率 | 45 dB 组合率 | 50 dB AUPR |
|---|---:|---:|---:|---:|
| BCE | 0.6823 | 0.3047 | 0.3164 | 0.8781 |
| high-only consistency | **0.7174** | **0.3477** | **0.3190** | **0.9047** |

但 10 dB AUPR 为 0.8563 → 0.8533，20 dB 为 0.9214 → 0.9164。

### low-only

| 方法 | 10 dB AUPR | 15 dB AUPR | 20 dB AUPR | 25 dB AUPR | 50 dB AUPR |
|---|---:|---:|---:|---:|---:|
| BCE | 0.8559 | 0.8632 | 0.9210 | 0.9932 | 0.8761 |
| low-only consistency | **0.8585** | **0.8658** | **0.9233** | **0.9936** | 0.8729 |

low-only 的 10 dB 组合率由 0.3919 提升到 0.3997，25 dB 由 0.5039 提升到 0.5104；但 50 dB AUPR 下降约 0.0031。

## 结论

高、低功率两侧的收益方向相反，说明简单把所有功率视图放进同一个一致性项会产生梯度干扰。high-only 和 low-only 各自在目标功率层有效，但都牺牲另一侧，尚未满足“最弱功率不下降、两个困难层同时改善”的晋级条件。该结果支持继续研究**预先固定的分层功率门控/双分支有界正则**，不支持直接冻结单侧模型。

## 失败边界

1. 本轮仅 3 seeds、2 epochs，不能替代最终五种子确认；
2. 门控由功率视图角色预先指定，不是由 test 调节；
3. 未使用任何专家网络，避免绕过 P4 门控；
4. 当前数据仍无法提供同 receiver 的换组合反事实对，P4 严格困难负样本仍未解决。

## 工件

- [执行脚本](E:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_p3c_power_consistency.py)
- [high-only 汇总](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p3d_high_only_20260830\p3c_power_consistency_summary.json)
- [low-only 汇总](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p3d_low_only_20260830\p3c_power_consistency_summary.json)
- [角色清单（high-only）](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p3d_high_only_20260830\role_manifest.json)

## 下一步

设计一个固定的双分支分层正则：低功率分支只作用于 10–25 dB，高功率分支只作用于 35–50 dB，并设置独立上限与 gate saturation 审计；先用 3 seeds 做最小筛查，若两侧同时不降再补齐 5 seeds。P4 严格反事实数据补全仍优先于 P5 深度专家构造。

