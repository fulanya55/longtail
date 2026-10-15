# Route A presence + visibility 辅助双头 development-v3 结果

## 结论

D2 连续功率差辅助监督未通过同容量因果门槛，不晋级、不访问 test-v3。低秩结构 D1 本身与
线性 C4 接近，但 D2 相对 D1 的分类改变量接近零，说明“可预测物理变量”不等于“普通多任务
损失会自动改善分类边界”。

## 五种子均值

| 方法 | macro AUPR | macro F1 | 弱召回 | 弱正-vs-负 AUROC | 强召回 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| C0 F2+C6 | 0.85050 | 0.73409 | 0.29937 | 0.58902 | 0.79720 | 0.18812 |
| C4 linear all-positive | **0.88128** | **0.77043** | 0.31192 | **0.59568** | **0.81290** | 0.19032 |
| D1 low-rank no-aux | 0.87927 | 0.76765 | 0.31496 | 0.59472 | 0.80244 | 0.19038 |
| D2 low-rank visibility-aux | 0.87940 | 0.76786 | **0.31584** | 0.59479 | 0.80325 | 0.19097 |

D2 相对 D1：弱召回 `+0.00088`、AUROC `+0.00007`、macro AUPR `+0.00013`；均远小于
预注册的 `+0.010` 机制门槛。D2 相对 C0 的弱召回为 `+0.01648`、AUROC `+0.00577`，
也未达到最终门槛。所有测试边界标志均为 false。

## 解释与下一证据

冻结 Ridge 探针已证明 gap 信息存在，但 D2 没有证明该信息被 visibility head 学到并有效影响
presence。下一步先对已冻结 D1/D2 checkpoint 做 post-hoc development 诊断：比较 visibility
head 对 gap 的 Spearman 和弱/强 AUROC。若 D2 明显学到但分类不变，后续使用“预测 gap 显式
控制有界残差”的同容量对照；若 D2 未学到，则停止当前双头实现并检查损失尺度/adapter 梯度。

产物：`outputs/tapc/route_a_presence_visibility_aux_v3/route_a_presence_visibility_aux_v3_summary.json`。

## Post-hoc 冻结诊断

对已冻结 D1/D2 checkpoint 的 early 角色进行可见度头诊断后，D1 无辅助头接近随机：三标签
平均 Spearman `-0.0466`、弱/强正 AUROC `0.4754`；D2 明显学到了物理量：平均 Spearman
`0.4218`、AUROC `0.7147`。其中 MFT/VFT 的 AUROC 分别为 `0.8196/0.7390`，ISRJ 为
`0.5855`。

因此失败原因已经收敛为：辅助头确实学到了可见度，但该信息只停留在旁路输出，没有显式
进入 presence 决策。下一轮不提高辅助损失权重，而是冻结 D2，比较样本对齐的“预测可见度
条件融合”和严格同容量的“打乱可见度条件融合”。诊断产物：
`outputs/tapc/route_a_presence_visibility_aux_v3_diagnostic/route_a_presence_visibility_aux_diagnostic_v3.json`。

