# 路线 A：Raw SawFM 独立分类器全分布审计方案（2026-07-25）

## 1. 目标

上一阶段证明 Raw frontend 不能作为 H2 的小幅残差。本轮不再融合，而是将其视为独立 SawFM 分类器，验证：

1. 在普通六类全分布上能否区分 SawFM present/absent；
2. 在统一 FPR 约束下能否同时覆盖普通域与 40/45/50 dB；
3. 是否只会识别训练时使用的“含扫频背景 SawFM”，而忽略不含扫频背景的 SawFM；
4. 概率校准是否足以支持独立阈值决策。

## 2. 冻结项与数据角色

- Raw frontend：seeds 2041/2042/2043 的 `N0_RawOnly` checkpoint；
- H2：对应 seed 的冻结基线；
- 所有网络参数冻结；
- ordinary calibration：拟合校准器；
- ordinary threshold + power threshold：选择统一阈值；
- ordinary early + power development：独立评价；
- locked test：不使用；
- power identity：按 `receiver_noise_seed` 隔离。

## 3. 方法

| 编号 | 分数与校准 |
|---|---|
| I0 | H2 SawFM logit + ordinary Platt |
| I1 | Raw logit + ordinary Platt |
| I2 | Raw logit + ordinary Isotonic |

校准器只使用 ordinary calibration 的完整 SawFM present/absent 标签。阈值网格 `0.01:0.01:0.99`，必须同时满足：

- ordinary threshold negative FPR `<=0.20`；
- 40/45/50 dB threshold hard-negative FPR 均 `<=0.20`。

可行候选按 mean-power recall、ordinary recall、50 dB recall、较高阈值顺序选择。

## 4. 指标

普通 early：

- AUROC、AUPR、Brier、10-bin ECE；
- SawFM recall、negative FPR；
- 含 swept 背景 SawFM recall/AUROC；
- 不含 swept 背景 SawFM recall/AUROC。

Power development：

- 40/45/50 dB recall、hard-negative FPR、AUROC/AUPR；
- mean/worst recall。

## 5. 晋级门

Raw 最佳校准器至少 2/3 seeds 同时满足：

1. ordinary early AUPR 不低于 H2 `-0.02`；
2. ordinary early negative FPR `<=0.20`；
3. ordinary early recall相对 H2 `>=+0.05`；
4. no-swept SawFM recall `>=0.60`；
5. power development mean recall `>=0.70`；
6. 50 dB recall `>=0.60`；
7. 各功率 hard-negative FPR `<=0.20`；
8. Brier `<=0.20`、ECE `<=0.10`；
9. locked test 未使用、身份审计通过。

若排序和召回通过但校准失败，只允许继续做校准研究；若 no-swept 子群失败，则 Raw frontend 只能定位为“扫频复合背景 SawFM 专家”，不能替换普通 SawFM 头；若总体失败，则停止独立决策路线。

## 6. 产物

- 程序：`experiments/run_route_a_raw_frontend_independent_classifier_audit.py`
- 测试：`tests/test_raw_frontend_independent_classifier_audit.py`
- 输出：`outputs/tapc/route_a_raw_frontend_independent_classifier_audit/`
- 结果：`docs/results/route_a_raw_frontend_independent_classifier_audit_results_20260725.md`
