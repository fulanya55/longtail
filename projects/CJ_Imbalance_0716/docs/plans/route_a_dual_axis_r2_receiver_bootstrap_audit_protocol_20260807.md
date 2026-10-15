# Route A R2 receiver-identity 配对 bootstrap 审计协议（2026-08-07）

## 1. 审计目的与证据边界

本审计只读取已经冻结的 R2 development evaluation predictions、cell summary 与 receipt，不训练模型、不重选 checkpoint、不重拟合校准器、不重选阈值，也不访问 locked test、test-v2 或 test-v3 的样本、标签、预测和指标。

审计回答两个问题：

1. 五 seed 汇总的点估计能否由逐 identity 的冻结预测独立复算；
2. `MFV_B0_DB_RawSaw - best(MF_B0_DB, MV_B0_RawSaw)` 的 receiver-level paired bootstrap 95% CI 是否支持独立组合增量。

该审计属于 `[A] development statistical audit`，不能升级为 confirmation/test 证据。

## 2. 固定输入与抽样单位

- 正式根目录：`outputs/route_a_dual_axis_next/r2_formal_primary`；
- seeds：`260811, 260812, 260813, 260814, 260815`；
- cells：`FB-VH, FB-VL, LT-VH, LT-VL`；
- 每个 cell 的 evaluation identity 数：2,520；
- 抽样单位：`sample_ids` 对应的 receiver/sample identity；
- 同一 bootstrap replicate 对所有 seed、cell、variant 使用同一组 identity multiplicity，保持完全配对；
- bootstrap repetitions：5,000；
- RNG seed：`26082401`；
- 区间：percentile 95% CI。

必须先验证 20 个 cell 的 identity 顺序、targets、完成状态和 test-use flags 一致且合法，否则审计中止。

## 3. 固定 estimand

复算并 bootstrap：

- `Delta_F_LT`：五 seed、LT-VH/LT-VL 上 `MF-M0` macro-AUPR 的平均差；
- `Delta_F_FB`：五 seed、FB-VH/FB-VL 上 `MF-M0` macro-AUPR 的平均差；
- `Delta_V_AUROC`：五 seed、FB-VL/LT-VL 上 `MV-M0` SawFM weak-vs-negative AUROC 平均差；
- `Delta_V_recall`：相同单元上的 weak recall 平均差；
- `MV_FPR`：相同单元上的候选 SawFM negative FPR；
- `Delta_V_strong`：相同单元上的 strong recall 平均差；
- `Delta_FV_macro`：五 seed、LT-VL 上 `MFV-max(MF,MV)` macro-AUPR 平均差；
- `Delta_FV_weak_recall`：五 seed、LT-VL 上 `MFV-max(MF,MV)` SawFM weak recall 平均差；
- 非 SawFM 安全性：LT-VL 上 MFV-M0 的逐标签 AUPR 差。

`best(MF,MV)` 在每个 bootstrap replicate、每个 seed 内重新取两者最大值。这是保守定义，禁止在看到结果后固定较弱对照。

## 4. 固定门槛与失败动作

沿用正式 addendum，不修改门槛：

- frequency-only：`Delta_F_LT >= 0.005`、至少 4/5 seed 为正、`Delta_F_FB >= -0.005`；
- visibility-only：weak AUROC `>=0.020`、weak recall `>=0.050`、候选 FPR `<=0.20`、strong recall 差 `>=-0.01`；
- combined：macro-AUPR `>=0.005` 且对应 CI 下界 `>0`，或 weak recall `>=0.020` 且对应 CI 下界 `>0`；
- 非 SawFM：任一标签五 seed 平均 AUPR 差不得低于 `-0.01`。

若 frequency-only 或 visibility-only 任一失败，保持 `controls_admitted=false`，不运行以正向双轴主张为前提的 MEC、MTS 和信息破坏消融。若 combined 失败，禁止声称独立组合增量。审计结束后进入可靠负结论和论文范围收缩整理，不得在同一 development source 上改参重试。

