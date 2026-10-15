# ECCV 2024 DR-Loss 公平基线结果

## 结论

在相同 F2 初始化、可训练范围、36 epoch 预算、RawSaw 路由、Platt
校准角色和阈值角色下，官方 DR-Loss 默认配置（`gamma1=gamma2=1`）稳定
劣于 Focal-36。它不能解释 VTSE-VBF 的弱分量收益，也不能作为统一候选的
替代方案。

| DR-Loss 减 Focal-36 | 平均差 | 95% bootstrap CI | 正向 seed |
|---|---:|---:|---:|
| 六类 macro-AUPR | -0.00849 | [-0.01026, -0.00703] | 0/5 |
| 六类 macro-F1 | -0.01092 | [-0.01339, -0.00862] | 0/5 |
| MFT/VFT/ISRJ 平均弱召回 | -0.01294 | [-0.02057, -0.00649] | 0/5 |
| 三目标平均弱 AUROC | -0.00792 | [-0.00938, -0.00663] | 0/5 |
| ISRJ AUPR | -0.01305 | [-0.01868, -0.00903] | 0/5 |
| ISRJ 弱 AUROC | -0.00700 | [-0.00985, -0.00415] | 0/5 |
| 三目标平均 FPR | +0.00059 | [-0.00258, +0.00398] | 2/5 |

ISRJ 弱召回只有 1/5 个 seed 为正，且区间跨零；它不能抵消排序与整体性能的
一致下降。该结果支持的范围有限：DR-Loss 的标签级分布鲁棒目标未解决此数据中
同一标签内强正—弱正可见度的不均衡；它并不单独证明 VTSE-VBF 的外部泛化。

## 公平性与边界

- 官方来源：ECCV 2024 DR-Loss 的公开 `dr_loss.py`，只使用 logits 与标签；
  本实现保留原始 `gamma1=gamma2=1` 公式，并将其按既有共同五类加
  `0.2 * SawFM` 的公平训练契约聚合。
- 每个 seed 从同一 `F2_DFMS_Full` 初始化，只训练同一 partial scope；Focal-36
  为已存储的同预算对照，未重新调参。
- 数据 SHA-256：`b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`；
  seeds：`2039, 2040, 2041, 2042, 2043`。
- 计划：[route_a_development_v3_dr_loss_baseline_plan_20260801.md](../plans/route_a_development_v3_dr_loss_baseline_plan_20260801.md)
- runner：`experiments/run_route_a_development_v3_dr_loss_baseline.py`
- receipt：`outputs/tapc/route_a_development_v3_dr_loss_baseline/route_a_development_v3_dr_loss_baseline_summary.json`

全程 `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。
该基线不改变候选、不授权 development-v13 或 test-v3。
