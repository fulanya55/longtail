# Route A H1+B2 周期难负分数解耦结果

## 判定

development-v3 门禁仅通过 5/10，整体失败。冻结输出上的低容量重堆叠不足以完成难负解耦。

D1 相对原 B2 的 ISRJ AUPR 增加 0.001959、弱召回增加 0.000181、强召回增加 0.003982，macro-F1 仅下降 0.000131；但核心排序指标弱正—负样本 AUROC 下降 0.007592，95% bootstrap CI 为 `[-0.015134,-0.001353]`。相对同结构 D0 控制组，弱 AUROC 同样下降 0.007236，说明退化确实来自周期难负加权，而不是分类头容量。

D1 平均 ISRJ FPR 为 0.201613，高于 0.20 门禁；五个种子的最差受支持子群 FPR 分别为 0.2531、0.2938、0.2781、0.2844、0.3031，全部未形成稳定安全边界。五种子中没有达到“至少四个弱 AUROC 正增益”的要求。

## 对照结果

| 指标 | D0 弱正加权堆叠头 | D1 难负解耦头 | D1-B2 | D1-D0 |
|---|---:|---:|---:|---:|
| macro-AUPR | 0.900551 | 0.900867 | +0.000327 | +0.000316 |
| macro-F1 | 0.796065 | 0.795991 | -0.000131 | -0.000074 |
| ISRJ AUPR | 0.906733 | 0.908630 | +0.001959 | +0.001898 |
| ISRJ 弱召回 | 0.415953 | 0.418283 | +0.000181 | +0.002329 |
| ISRJ 强召回 | 0.891775 | 0.894550 | +0.003982 | +0.002775 |
| ISRJ 弱 AUROC | 0.672672 | 0.665435 | -0.007592 | -0.007236 |
| ISRJ FPR | 0.196452 | 0.201613 | +0.006774 | +0.005161 |

## 失败原因

fit role 中按 `B2-B1` 最高四分位选出的所谓周期难负，其切点在五种子均为负值（-0.7781 至 -0.1829）。这说明 B2 与 B1 两个已经训练完成的输出差值不是一个稳定、纯净的周期证据轴；它混合了两套分类头的尺度和边界差异。对该差值加权会优先改变概率排序，却不能稳定定位真正的物理周期假阳性。

因此，不能继续在 B1/B2 最终输出上搜索权重、分位点或交互特征。问题必须前移到 310 维周期专家本身：使用专家自己的 fit-role 分数识别 composition-matched ISRJ 负样本中的高分难负，并在相同 HGB 容量下重新训练；先要求 matched-view AUROC/AUPR 改善，再允许重新接回 H1 融合头。

## 证据边界与产物

- 数据：development-v3，种子 2039—2043。
- 非 ISRJ logits 最大绝对变化为 0；test-v3 未使用。
- 预注册：`docs/plans/route_a_h1_periodicity_hard_negative_score_decoupling_plan_20260801.md`。
- 实现：`experiments/run_route_a_h1_periodicity_hard_negative_score_decoupling.py`。
- 测试：`tests/test_route_a_h1_periodicity_hard_negative_score_decoupling.py`，1 passed。
- 输出：`outputs/tapc/route_a_h1_periodicity_hard_negative_score_decoupling`。
- 汇总 SHA256：`697B38A4F89BE2C78EB59C71DE620456E6C1972011A35F4A63202F1193445E96`。

