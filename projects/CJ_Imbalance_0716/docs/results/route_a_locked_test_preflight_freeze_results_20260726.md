# 路线 A：locked-test 前冻结审计结果（2026-07-26）

## 1. 结论

`RawSaw+H2Others`已完成最终测试前preflight，状态为`ready_for_authorization`。审计只读取development产物、配置和checkpoint，没有读取locked test。

本轮发现并修复了一个关键可复现性缺口：原五种子实验保存了Raw阈值和预测，但没有独立保存Raw Platt系数与截距。现在五个seed的Platt参数均已从原ordinary calibration角色确定性重放并持久化；正式summary内容和SHA256完全不变，证明没有修改模型、指标、阈值或晋级结论。

## 2. 执行内容

1. 在`run_route_a_raw_platt_five_seed_label_routing.py`中增加每seed `calibration.json`；
2. 使用PY10重放五种子development；
3. 验证正式summary SHA256仍为：
   `1C456FF579F0FA2EFCCBA202152F5F2CA117CA215A7D84131447D4086D491EFC`；
4. 新增`freeze_route_a_raw_saw_label_routing.py`；
5. 对五个seed逐项验证F2/H2/Raw checkpoint、split、identity、temperature、H2 threshold、Raw Platt和Raw threshold；
6. 生成冻结推理包、preflight审计和receipt。

## 3. 五种子冻结参数

| seed | Raw协议 | Raw threshold | Platt coefficient | Platt intercept | H2 temperature |
|---:|---|---:|---:|---:|---:|
| 2039 | new_split_R0 | 0.33 | 0.857727 | -0.350080 | 3.327313 |
| 2040 | new_split_R0 | 0.40 | 0.856262 | -0.366893 | 3.015163 |
| 2041 | original_N0 | 0.35 | 0.556981 | -0.523297 | 2.685783 |
| 2042 | original_N0 | 0.39 | 0.693503 | 0.625512 | 2.689069 |
| 2043 | original_N0 | 0.41 | 0.681986 | 0.216720 | 3.100362 |

每个seed的Raw Platt只使用300个ordinary calibration样本。H2均保存6个逐标签threshold。所有校准文件均为`locked_test_used=false`。

## 4. Preflight门禁

| 检查 | 结果 |
|---|---|
| 正式summary SHA256未变化 | 通过 |
| 五个F2 checkpoint哈希一致 | 通过 |
| 五个H2 checkpoint哈希一致 | 通过 |
| 五个Raw checkpoint哈希一致 | 通过 |
| Raw Platt参数完整 | 通过 |
| Raw统一threshold完整 | 通过 |
| H2 temperature与六标签threshold完整 | 通过 |
| identity overlap | 0 |
| confirmation门 | 4/5或以上，通过 |
| label routing门 | 5/5，通过 |
| locked test访问 | false |

## 5. 冻结推理规则

- 输入：`[B,2,3000]`，complex RMS normalization；
- class order：`[FMB, SawFM, SinFM, MFT, VFT, ISRJ]`；
- SawFM：Raw frontend → 固定Platt → 固定SawFM threshold；
- 其他五类：F2 → H2 → 固定temperature → 固定逐标签threshold；
- 不使用ensemble；
- 不使用learned fusion；
- 不输入真实功率、JNR或生成元数据；
- 最终测试不能重新训练、拟合校准器或选择阈值。

## 6. 产物与哈希

- 计划：`docs/plans/route_a_locked_test_preflight_freeze_plan_20260726.md`
- Development程序：`experiments/run_route_a_raw_platt_five_seed_label_routing.py`
- Preflight程序：`experiments/freeze_route_a_raw_saw_label_routing.py`
- 测试：`tests/test_route_a_raw_saw_freeze_preflight.py`
- 冻结包：`outputs/tapc/route_a_raw_saw_locked_test_preflight/frozen_inference_bundle.json`
- 冻结包 SHA256：`F5FB713FE0C8EF351C6CDD42304E5D1541C955D27B28CBBFDB5B8B03B55B9489`
- 审计：`outputs/tapc/route_a_raw_saw_locked_test_preflight/preflight_audit.json`
- 审计 SHA256：`B53350B746E6BA83B6462521027699C03CA2F1692FD22F9AE0D75623535E754A`
- Receipt：`outputs/tapc/route_a_raw_saw_locked_test_preflight/run_receipt.json`
- Receipt SHA256：`CA73373B444B61404ABEB3A21398C2C9E064E446E32A1B1255469DD0B83D0FB3`

## 7. 下一决策

当前工作已完成到“等待明确授权”。除非用户明确批准一次性locked-test执行，否则不得读取测试数据，也不得创建以测试结果为依据的新阈值或新模型。

