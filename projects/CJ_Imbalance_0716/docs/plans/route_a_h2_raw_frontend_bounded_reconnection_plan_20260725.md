# 路线 A：H2 与 Raw SawFM 前端有界接回方案（2026-07-25）

## 目标

Raw-only 前端已经在五个 split seeds 上通过独立验证。本轮冻结 H2 与 Raw frontend，仅验证其分数能否安全接回六类决策：改善 SawFM 弱正排序与 50 dB 召回，同时限制普通域、hard-negative FPR 和整体六类指标损失。

## 冻结来源

- F2/H2：seeds 2041、2042、2043 的冻结 checkpoint；
- Raw frontend：对应 seed 的 `N0_RawOnly/best_frontend.pth`；
- 其他五类 logits：逐样本完全保持 H2 原值；
- locked test：不使用。

## 数据角色

- ordinary calibration：仅拟合每种方法的 temperature；
- ordinary threshold + power threshold：选择接回强度与 SawFM threshold；
- ordinary early + power development：独立评价；
- power scale 只作二级稳健性审计，不参与最终 early 指标；
- 所有功率身份按 `receiver_noise_seed` 分组，角色间无重叠。

## 方法

Raw logit 先用 fit 角色做单变量 Platt 标准化为 `z_raw`，不使用 early。

| 编号 | SawFM logit |
|---|---|
| B0 | `h2` |
| B1 | `h2 + α*tanh(z_raw)*exp(-abs(h2))`，α=0.05 |
| B2 | 同上，α=0.10 |
| B3 | 同上，α=0.20 |
| B4 | hard rescue：仅当 `h2<g` 且 `sigmoid(z_raw)>=r` 时添加 `+0.20` |

B4 网格：`g∈{-1.0,-0.5,0.0}`，`r∈{0.70,0.80,0.90}`。所有方法只修改 SawFM 一列。

## 候选选择

每种方法在 threshold 角色为 SawFM 搜索阈值 `0.01:0.01:0.99`，候选必须满足：

- ordinary SawFM negative FPR `<=0.20`；
- 40/45/50 dB hard-negative FPR 均 `<=0.20`；
- ordinary macro-AUPR 相对 B0 不低于 `-0.005`；
- ordinary macro-F1 相对 B0 不低于 `-0.02`。

可行候选按 mean-power recall、50 dB recall、ordinary SawFM AUPR、较小残差依次选择。

## 晋级门

最佳新方法至少 2/3 seeds 同时满足：

1. development mean-power recall 相对 B0 `>=+0.10`；
2. development 50 dB recall 相对 B0 `>=+0.15`；
3. development 各功率 hard-negative FPR `<=0.20`；
4. ordinary early SawFM negative FPR `<=0.20`；
5. ordinary early macro-AUPR 相对 B0 `>=-0.005`；
6. ordinary early macro-F1 相对 B0 `>=-0.02`；
7. 其他五类概率和预测逐项完全不变；
8. locked test 未使用、身份审计通过。

未通过则 Raw frontend 保留为独立机制结果，不接回 H2。

## 产物

- 程序：`experiments/run_route_a_h2_raw_frontend_bounded_reconnection.py`
- 测试：`tests/test_h2_raw_frontend_bounded_reconnection.py`
- 输出：`outputs/tapc/route_a_h2_raw_frontend_bounded_reconnection/`
- 结果：`docs/results/route_a_h2_raw_frontend_bounded_reconnection_results_20260725.md`
