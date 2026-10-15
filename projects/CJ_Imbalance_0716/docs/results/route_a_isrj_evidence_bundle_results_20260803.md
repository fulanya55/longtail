# Route A：冻结 ISRJ aligned-periodicity 证据包结果

日期：2026-08-03  
状态：完成的只读证据归档，不是新模型实验。

## 结果

已生成机器可读证据包：

`outputs/tapc/route_a_isrj_evidence_bundle_20260803/evidence_bundle.json`

该构建器只读取 H0g/H0h 的冻结摘要、已经完成的共标签只读审计，以及 `test-v3` 密封清单元数据。没有读取任何测试 I/Q，没有训练、选择、校准或阈值拟合。

| 新身份外部条件 | B2 − B0：ISRJ AUPR | weak AUROC | weak recall | FPR | seed 一致性 |
| --- | ---: | ---: | ---: | ---: | --- |
| H0g：25--30 us、9 replicas | `+0.119267` `[+0.093757,+0.145030]` | `+0.099646` `[+0.089815,+0.110694]` | `+0.071252` `[+0.063139,+0.080776]` | `+0.002419` `[-0.003952,+0.008629]` | weak AUROC 5/5；weak recall 5/5 |
| H0h：proper complex AR(1), rho=0.80 | `+0.022420` `[+0.015772,+0.028525]` | `+0.080371` `[+0.074622,+0.086824]` | `+0.102703` `[+0.096577,+0.109910]` | `-0.004677` `[-0.011129,+0.000242]` | weak AUROC 5/5；weak recall 5/5 |

每个摘要、共标签审计和密封清单都被记录为 SHA-256。证据包同时确认：

- `selection_or_refit_performed=false`；
- `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`；
- `test_v3.access_authorized=false`、`analysis_allowed=false`、`used=false`；
- 共标签审计只支持“排序增益并非局限于单一已报告的 MFT/VFT 共标签”；它**不**支持“任意上下文均具有阈值安全性”。

## 可写入论文的受限表述

在本合成协议、两类新 receiver-noise identity 条件及冻结 B0/B2 工件范围内，aligned-periodicity 增量相对 B0 一致提高 ISRJ 弱分量排序与召回。该结论有参数外推与相关接收机噪声两条独立外部复现。

## 不可写入论文的表述

- 不可称为完整 MFT/VFT/ISRJ 统一模型已经推广；
- 不可称为所有 MFT/VFT 共现上下文均具低 FPR 或高弱召回；
- 不可称为 final locked-test 结果；
- 不可据此重新开启已停止的 MFT/VFT 分支。

## 验证

`D:\anaconda3\envs\PY10\python.exe -m pytest -q tests/test_route_a_isrj_evidence_bundle.py`

结果：`2 passed`。

## 关联文件

- 准入审计：`docs/reports/route_a_mft_vft_architecture_admission_audit_20260803.md`；
- 构建器：`experiments/build_route_a_isrj_evidence_bundle.py`；
- H0g：`docs/results/route_a_h0g_isrj_parameter_ood_v2_frozen_results_20260802.md`；
- H0h：`docs/results/route_a_h0h_ar1_noise_frozen_results_20260802.md`；
- 共标签审计：`docs/results/route_a_isrj_frozen_colabel_audit_results_20260803.md`。

