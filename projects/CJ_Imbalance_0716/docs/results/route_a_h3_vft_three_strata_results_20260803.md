# Route A H3：VFT 三层可见性与交错 PRI 可辨识性结果

日期：2026-08-03  
证据角色：新的 identity-isolated development-only 机制验证；非 external、非历史 locked test、非 test-v2、非 test-v3。

## 结论

H3 模型比较**未通过 H3 的预注册模型门槛**。按停止规则，不改变时间戳排列、ROI、专家宽度、JNR、epoch、学习率或损失重跑 H3；该物理可观测量存在，但当前目标专属可训练投影未证明其优于等容量置换控制。

## 物理零拟合前置证据

- weak-vs-absent：真实时间戳 AUROC `0.779026`，置换 AUROC `0.570946`，差值 `+0.208079`，配对 bootstrap 95% CI `[0.18936791875314946, 0.2268627025462963]`。
- strong-vs-absent：真实时间戳 AUROC `0.833417`，置换 AUROC `0.602957`。
- 零拟合物理门槛：`True`；训练参数数为 `0`。

## 五种子模型对照

| 对比 | 均值差 | 95% bootstrap CI | 正向种子 |
|---|---:|---:|---:|
| T1 真时间戳 − T2 置换：weak AUROC | `-0.001365` | `[-0.0018787202380953216, -0.0008635282816832124]` | `0/5` |
| T1 真时间戳 − T2 置换：weak AUPR | `-0.001061` | `[-0.0014641808990296656, -0.000640843252647283]` | `0/5` |
| T1 真时间戳 − T0 基线：strong AUROC | `+0.000864` | `[-0.002439216427311708, 0.003640125031494046]` | `3/5` |

## 预注册门槛

- `t1_minus_t2_weak_auroc_mean_ge_0p030`: `False`
- `t1_minus_t2_weak_auroc_5of5_positive`: `False`
- `t1_minus_t2_weak_auroc_ci_lower_gt_0`: `False`
- `t1_minus_t2_weak_aupr_mean_ge_0p025`: `False`
- `t1_minus_t2_weak_aupr_ci_lower_gt_0`: `False`
- `t1_minus_t0_strong_auroc_mean_ge_minus_0p010`: `True`
- `t1_minus_t0_strong_auroc_every_seed_ge_minus_0p020`: `True`
- `t1_t2_equal_parameter_count`: `True`
- `all_runs_ranking_only_and_test_free`: `True`

## 可审计工件与边界

- 15 份训练工件、checkpoint 与不可变 development predictions 已由 `outputs/tapc/route_a_h3_vft_three_strata_artifact_audit.json` 哈希审计；T1/T2 参数量严格相等。
- 汇总：`outputs/tapc/route_a_h3_vft_three_strata_development_summary.json`；零拟合物理结果：`outputs/tapc/route_a_h3_vft_three_strata_observable/summary.json`。
- 校准：无；阈值选择：无；所有结论为 ranking-only development 证据。
- `external_used=false`、`locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。
