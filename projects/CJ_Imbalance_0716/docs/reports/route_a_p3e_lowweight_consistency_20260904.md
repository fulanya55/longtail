# Route-A P3e：双侧低权重功率一致性筛查（2026-09-04）

## 状态

**completed（3-seed 探索）；部分支持，未冻结。** 在 P3c/P3d 观察到低、高功率梯度冲突后，将同时作用于 10 dB 和 50 dB 的一致性权重从 0.10 降至 0.05，其他设置不变。

## 结果（3 seed 均值）

| 方法 | 10 dB AUPR | 20 dB AUPR | 25 dB AUPR | 35 dB 组合率 | 40 dB 组合率 | 45 dB AUPR | 50 dB AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| BCE | 0.856974 | 0.921065 | 0.992888 | 0.677083 | 0.300781 | 0.948079 | 0.879141 |
| BCE+consistency λ=0.05 | 0.855819 | 0.920670 | **0.992963** | **0.688802** | **0.311198** | **0.954745** | **0.882603** |

验证组合率均值为 BCE 0.993815、λ=0.05 为 0.994466；Macro-F1 为 0.998987 → 0.999093。低功率 10 dB 组合率由 0.395833 降至 0.393229，说明减小权重只能缓解 AUPR 损失，不能消除低功率边界退化。

## 判定

低权重方案在中高功率有小幅、较一致的收益，但 10–20 dB 仍不满足功率鲁棒门控；3 seeds 也不足以冻结。继续扫描 λ 会把同一开发集用于反复调参，当前先停止该调参线，保留 λ=0.05 作为中高功率候选记录。

## 工件

- [执行脚本](E:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_p3c_power_consistency.py)
- [汇总 JSON](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p3e_both_lowweight_20260904\p3c_power_consistency_summary.json)
- [角色清单](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p3e_both_lowweight_20260904\role_manifest.json)
- [终端回执](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p3e_both_lowweight_20260904\terminal_receipt.json)

## 下一步

不再调节一致性权重。P4 matched-HN 已在严格组合组留出下停止；若要继续形成创新点，应重新设计可解释的低/高功率双分支机制并预先固定规则，或补齐带反事实索引的数据后重新注册 P4。P5 专家模块保持禁止启动。

