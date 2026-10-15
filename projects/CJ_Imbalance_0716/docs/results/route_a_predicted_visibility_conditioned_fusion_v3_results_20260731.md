# Route A 预测可见度条件融合 development-v3 结果

## 1. 结论

E1 aligned predicted visibility 相对 E0 shuffled visibility 存在小而一致的排序信息贡献，但未达到
预注册的分类晋级门槛，因此不晋级，不访问 test-v3，并按停止规则终止 final-temporal 后处理线。

## 2. 五种子均值

| 方法 | macro AUPR | macro F1 | 弱召回 | 弱正-vs-负 AUROC | 强召回 | FPR | ISRJ FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 F2+C6 | 0.85050 | 0.73409 | 0.29937 | 0.58902 | 0.79720 | 0.18812 | 0.19081 |
| C4 linear all-positive | **0.88128** | **0.77043** | 0.31192 | 0.59568 | **0.81290** | 0.19032 | 0.19339 |
| D2 visibility auxiliary | 0.87940 | 0.76786 | 0.31584 | 0.59479 | 0.80325 | 0.19097 | 0.19468 |
| E0 shuffled visibility | 0.87939 | 0.76803 | 0.31490 | 0.59493 | 0.80373 | 0.19048 | 0.19226 |
| E1 aligned visibility | 0.88115 | 0.77034 | **0.31758** | **0.59803** | 0.81109 | **0.19000** | 0.19339 |

## 3. 因果对照差值

E1 相对 E0：

- macro AUPR `+0.00176`，95% CI `[+0.00073,+0.00267]`；
- macro F1 `+0.00231`；
- 弱召回 `+0.00268`，95% CI `[-0.00155,+0.00752]`，仅 3/5 种子为正；
- 弱正—负 AUROC `+0.00310`，95% CI `[+0.00089,+0.00584]`；
- 强召回 `+0.00736`；
- FPR `-0.00048`。

E1 相对 C0：macro AUPR `+0.03065`、macro F1 `+0.03625`、弱召回 `+0.01822`、
AUROC `+0.00901`，5/5 种子弱召回为正。弱召回和 AUROC 仍低于总目标的 `+0.030/+0.015`。

## 4. 失败原因与证据边界

真实样本级 predicted visibility 的确比打乱对照更有用，说明物理预测不是纯容量效应；但其
效应主要体现为小幅排序和强正恢复，无法形成足够大的弱召回提升。D2 的 ISRJ 可见度预测本身
较弱，且 final-temporal 全局统计对低功率局部证据的可控性有限。继续调严重度中心、温度或
融合学习率将变成事后搜索，违反预注册停止规则。

该结果可以作为“预测可见度具有小但可检测的样本级贡献”的 development 机制证据，不能
作为统一模型已经证明有效的证据。下一步转向同 receiver identity 多功率训练视图，从 stage2
训练小型配对 adapter，使可见度方向在更早层形成，而不是继续做 final-temporal 后处理。

所有运行均为 `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。

产物：`outputs/tapc/route_a_predicted_visibility_conditioned_fusion_v3/route_a_predicted_visibility_conditioned_fusion_v3_summary.json`。

