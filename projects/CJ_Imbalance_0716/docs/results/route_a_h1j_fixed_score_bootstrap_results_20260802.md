# Route A H1j：固定分数身份 bootstrap 重审

## 决定

**H1j 的 VFT 正确去调频标量不满足本 Goal 的可靠 H1 晋级门，不进入 H2、安全集成或 test-v3。**

这是一项 read-only 重审：重算 H1j 已有的正确/phase-permuted 参考分数与已保存的零初始化一维 BCE 头，未改数据、参考、脉冲窗口、匹配规则、模型、训练、校准、阈值或融合。

## 固定来源与边界

- H1j train-only source SHA-256：`33de4102ccd871a9d45dec97974a81fc6890597b9da3fc28cabce5d3ad5d3070`，5040 唯一 receiver identities，`998677743--998682782`。
- 评估对象完全沿用原冻结 view：weak VFT positives `107`、VFT-absent matched negatives `480`；strong positives `389`、negatives `496`。
- 原五个 optimizer seed 的 candidate/control 头均为正单调读出，因此五个 seed 保持完全相同的 AUROC/AUPR 排序；这不是五个独立排名复现。
- 本次 CI 为 10,000 次 paired bootstrap。输出记录中称为 `paired_hierarchical_bootstrap`；由于 seed 排名完全一致，解释时只把它视为 receiver-identity 不确定性，绝不当作五份独立 source 复现。
- `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。

## 固定分数结果：正确参考减 phase-permuted 对照

| 端点 | 点估计 | 95% bootstrap CI | Goal H1 判定 |
|---|---:|---:|---|
| weak VFT AUROC | `+0.048715` | `[-0.036530,+0.132545]` | **失败：下界不大于 0** |
| weak VFT AUPR | `+0.017996` | `[-0.028691,+0.069734]` | **失败：下界不大于 0** |
| strong VFT AUROC | `+0.219400` | `[+0.168014,+0.271910]` | 未见强正伤害 |

## 失败原因

1. 正确参考确实改善了弱样本的排序方向，但弱 AUROC 与弱 AUPR 的身份不确定性都跨 0，不能证明该增益在固定复合源上可靠。
2. 五次训练只改变一维正单调读出的尺度和偏置，不改变排序；将它们视为五个独立 seed 成功会造成伪重复。
3. 即使 AUROC 端点为正，H1 仍需 AUPR、弱 AUROC、strong safety 共同通过，不能只用一个较有利指标晋级。

## 停止规则

停止正确去调频相位增量的任何窗口、相位置换、读出、训练预算、阈值、校准或融合调参。该结果保留为“存在 VFT 物理线索但尚未形成可靠复合弱证据”的阴性 H1 记录。新的 VFT 研究问题必须是一个在 Stage 0 中与 H0i/H1j 明确非重叠的物理观测，而不能是该标量或 RAAT 地图化变体。

机器可读收据：`outputs/tapc/route_a_h1j_fixed_score_bootstrap/route_a_h1j_fixed_score_identity_bootstrap_summary.json`。

### 收据版本说明

同目录的 `route_a_h1j_fixed_score_bootstrap_summary.json` 是一次临时统计草稿：它把完全相同的五个 optimizer 排序重复作为层级 seed 抽样，不能用于不确定性结论。该草稿已撤销；本报告只引用上方的 `paired_identity_10000` 收据。
