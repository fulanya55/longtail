# Route A MFT 残差上界移除消融结果

## 结论

移除额外 `0.75` 上界（设 cap=`1.0`，保留 `tanh` 的自然界）并未破坏 development-v3 的 MFT guard 安全门，且在构造集上取得略强的困难负样本抑制。但该差异不足以支持把“0.75 上界”写成创新机制，也不能据此修改已冻结 candidate-v2。

## cap=1.0 相对 B2 的结果

| 指标 | cap=0.75 | cap=1.0 |
|---|---:|---:|
| MFT hard-FPR 差 | -0.02833 | -0.03167 |
| MFT hard AUROC 差 | +0.01340 | +0.01756 |
| 目标弱召回差 | -0.00319 | -0.00385 |
| 目标弱 AUROC 差 | +0.00620 | +0.00820 |
| macro-AUPR 差 | +0.00078 | +0.00101 |

两种配置均使用同一已冻结峰计数专家、B2 logit、校准器与阈值，且五个 seeds 均降低 hard-FPR、非 MFT logit 完全不变。

## 解释边界

cap=1.0 对 cap=0.75 的差异只在已反复使用的 construction-v3 中得到，且当前 shared-decision candidate-v2 已以 cap=0.75 冻结并完成 v12 评估。根据预注册边界，不能用本结果回调候选。可支持的结论仅是：**残差的非正方向与父级决策契约是关键安全结构；0.75 的具体缩放不是经过可靠独立证明的创新点。**

- 预注册：[route_a_mft_residual_cap_removal_ablation_plan_20260801.md](../plans/route_a_mft_residual_cap_removal_ablation_plan_20260801.md)
- 复用实现：`experiments/run_route_a_h1_b2_mft_peak_guard_v3.py --residual-cap 1.0`
- 汇总：`outputs/tapc/route_a_mft_residual_cap1_ablation/route_a_h1_b2_mft_peak_guard_v3_summary.json`
- `locked_test_used=false`；`test_v3_used=false`。
