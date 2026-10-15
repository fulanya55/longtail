# Route A 物理可见度差冻结探针 development-v3 结果

## 结论

五随机种子全部通过预注册门槛。冻结 F2 final-temporal 512 维统计能够可靠恢复 MFT、VFT、
ISRJ 的连续相对功率差，因此当前失败不能归因于特征中完全不存在可见度信息；主要缺口是
分类目标与融合器没有有效利用该连续信息。

## early 角色五种子结果

| 标签 | Spearman | MAE/dB | R2 | 弱-vs-强正 AUROC |
|---|---:|---:|---:|---:|
| MFT | 0.8793 | 2.008 | 0.7330 | 0.9427 |
| VFT | 0.7613 | 2.667 | 0.5122 | 0.8748 |
| ISRJ | 0.6999 | 3.397 | 0.3203 | 0.8288 |
| 三标签平均 | **0.7802** | — | — | **0.8821** |

五个种子均通过平均 Spearman `>=0.30`、平均 AUROC `>=0.65` 和至少 2/3 标签通过的
种子内门槛。`test_v3_used=false`。

## 证据边界与下一步

该结果是 development-v3 的机制可行性证据，不是分类方法提升证据。下一步允许进入
presence + continuous-gap 双头训练，但必须设置完全相同参数量、优化器和选择规则的
no-aux 对照；推理时不得使用真实功率或弱正标记。

产物：`outputs/tapc/route_a_visibility_gap_frozen_probe_v3/route_a_visibility_gap_frozen_probe_v3_summary.json`。

