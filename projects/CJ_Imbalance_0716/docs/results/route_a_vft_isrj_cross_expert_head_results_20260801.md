# Route A VFT-ISRJ 跨专家分类头：否定结果

## 结论

该修正失败，不能接入统一候选，也不能据此生成 development-v13。它没有解决 ISRJ 参数 OOD 下 VFT 的来源混淆，反而在三个已冻结诊断集上稳定损害了 VFT 的排序。

## 固定设计

三个逻辑回归头均为 5 个输入、6 个可训练标量：父 VFT logit、VFT 专家 logit、二者差值、ISRJ logit，以及差值与 ISRJ 概率的交互项。对照头以两个非语义非线性项替代 ISRJ 信息；shuffle 对照在角色内打乱 ISRJ logit。所有头均固定 `C=0.1`，弱 VFT 正样本与 VFT-negative/ISRJ-positive 难负样本权重均为 4，无超参数搜索。

## 结果

| 数据 | VFT AUPR | VFT 弱 AUROC vs capacity | VFT 弱 AUROC vs shuffled | VFT 弱召回 | VFT FPR | 裁决 |
|---|---:|---:|---:|---:|---:|---|
| construction-v3 | -0.01622 | -0.02318 | -0.01613 | -0.00952 | 0.18903 | 失败 |
| development-v12 诊断 | -0.01683 | -0.02254 | -0.01693 | -0.00750 | 0.19376 | 失败 |
| ISRJ parameter-OOD | -0.01276 | -0.00558 | +0.00011 | -0.00195 | 0.35944 | 失败 |

虽然 macro 指标在部分数据上仍为正，例如 OOD macro-AUPR 为 `+0.00972`，但这是以 VFT 特异性退化换来的，不能作为机制成功证据。OOD 中 ISRJ-present/VFT-negative FPR 达到 `0.45875`，也超过预注册的 `0.25` 上限。

## 原因与后续约束

ISRJ 分类 logit 是过度压缩的上下文变量，不能表达“该时频脊属于 VFT 还是长时密集 ISRJ”的原始物理来源；在训练角色中赋予 ISRJ 难负权重后，头部直接压低了部分真 VFT 证据。因此不允许继续调权重、阈值或正则化。后续若要继续 VFT-OOD 路线，必须先在原始物理特征级证明来源可分，而不是再叠加低维决策 logit。

## 证据与边界

- 预注册：[route_a_vft_isrj_cross_expert_head_plan_20260801.md](../plans/route_a_vft_isrj_cross_expert_head_plan_20260801.md)
- 汇总：`outputs/tapc/route_a_vft_isrj_cross_expert_head/route_a_vft_isrj_cross_expert_head_summary.json`
- `locked_test_used=false`；`test_v3_used=false`。
