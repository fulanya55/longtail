# Route A H0r：MFT 三阶相位耦合零拟合结果

日期：2026-08-03  
决策：**H0r 失败并停止；不进入 H1、网络、融合或 test-v3。**

## 固定合同

H0r 在 640 个新 seed 配对中比较 3-copy 与 8-copy 单标签 MFT。固定读出是 8 个
等长段、DFT bins `(7,11)` 的 segmented bicoherence；控制为逐记录 full-record DFT
随机相位 surrogate。该控制严格保持每个 DFT bin 幅度与圆周二阶自相关，故只破坏
三阶相位耦合。无训练参数、无阈值、无校准、无复合背景。

## 结果

| 项目 | 数值 |
| --- | ---: |
| candidate 3-vs-8 AUROC | 0.430652 |
| surrogate-control AUROC | 0.529189 |
| candidate − control AUROC | -0.098538 |
| paired bootstrap 95% CI | [-0.142379, -0.055622] |
| paired 8 > 3 fraction | 0.426563 |
| 最大 DFT 幅度相对误差 | `6.10e-16` |
| 最大圆周二阶自相关相对误差 | `3.14e-16` |

控制保持门全部通过，说明相位 surrogate 合同按预期工作；然而三阶统计的绝对排序
反向且相对控制显著更差。它不能作为 MFT 副本数量的可归因物理证据。

## 停止边界

不得调整 bins、segment 数、copy count、JNR、seed 规模、surrogate、统计、网络、
损失、校准或阈值。H0r 不生成 H1、不接入 VTSE-VBF、不访问 external/locked/test-v2/
test-v3。该失败强化了当前结论：不能仅因三阶量在数学上不同于二阶相关，就假定其在
本 LFM-MFT 合同中具有目标可辨识性。

## 工件

- 预注册：`docs/plans/route_a_h0r_mft_higher_order_identifiability_20260803.md`
- 汇总：`outputs/tapc/route_a_h0r_mft_higher_order/summary.json`
- smoke：`outputs/tapc/route_a_h0r_mft_higher_order_smoke/summary.json`
- 实现：`cjlt/models/mft_higher_order.py`、`experiments/run_route_a_h0r_mft_higher_order.py`

所有工件记录 `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。
