# Route A：频次组损失不能直接替代标签内可见性建模

日期：2026-08-03  
证据角色：解析性损失机制审计；非训练、非性能比较、非测试结果。

## 结论

在同一标签且 logit 相同的正样本之间，BCE、pos-weight BCE、Focal、ASL、
CB-BCE、Logit Adjustment 和训练频次组 SoftF1 不会因样本被命名为 weak 或
strong 而改变损失或梯度。它们没有接收 visibility mask，因此不能直接实现
VTSE/VBF 所需的标签内 weak-positive/strong-positive 分层目标。

Distribution-Balanced Loss（DB）有一个重要但不同的边界：在相同标签、相同
logit 的两个正样本具有不同**其他共现标签**时，DB 的梯度可以不同；本审计示例
中为 `-0.061724` 与 `-0.050034`。这是组合/频次条件，而不是对未输入的
weak/strong 可见性分层的条件化。交换两样本的 weak/strong 名称后，DB 的完整
目标和完整梯度仍严格不变。

## 解析控制

构造两个 label-0 正样本，二者 logit 都为 `0.20`，仅赋予不同的 weak/strong
名称。所有被审计损失均只接收 logits、标签和训练频次/先验；visibility mask
从不作为函数输入。逐项检查如下。

| 损失 | weak/strong 互换后的 loss 相等 | 完整 logit 梯度相等 | 同标签同 logit 的两个正样本梯度相等 |
|---|---:|---:|---:|
| BCE | 是 | 是 | 是 |
| pos-weight BCE | 是 | 是 | 是 |
| Focal | 是 | 是 | 是 |
| ASL | 是 | 是 | 是 |
| DB | 是 | 是 | 否，仅受共现标签影响 |
| CB-BCE | 是 | 是 | 是 |
| Logit Adjustment | 是 | 是 | 是 |
| FrequencyGroupSoftF1 | 是 | 是 | 是 |

所有预注册解析检查通过，且 `iq_opened=false`、`labels_opened=false`、
`training_performed=false`、`locked_test_used=false`、`test_v2_used=false`、
`test_v3_used=false`。

## 与已有五种子结果的合并解释

development-v3 的公平训练已经显示 DB 是普通损失中 macro-AUPR 最强者，但它没有
复现周期性 ISRJ 专家的弱证据收益；DR/RAL 也未提供可晋级的统一弱分量增益。该
解析审计补齐的不是新的性能数字，而是因果解释：这些损失可以处理标签频次、正负
不对称或组合共现，却不包含“同一标签内哪个正样本弱可见”的直接监督接口。

因此，`FrequencyGroupSoftF1` 不作为一个新的 VTSE-VBF 候选训练，也不以 TALC
或任何未获准入的外部方法名称报告。若未来把它作为补充频次基线，必须在新的身份
隔离开发源上预注册、同预算运行，并明确其仅检验频次层；它不能填补 MFT/VFT
的物理可辨识或 C3/C4 容量因果缺口。

## 允许与禁止的表述

允许：标签频次/组合重加权与标签内可见性分层是不同机制；现有频次损失不直接
编码 weak/strong strata。  
禁止：从此审计推断频次损失绝不会间接改善弱样本，或将单次解析结果当作最终性能
比较。间接效应须由新的 identity-isolated 五种子训练验证。

## 工件

- 预注册：[route_a_frequency_group_visibility_invariance_audit_20260803.md](../plans/route_a_frequency_group_visibility_invariance_audit_20260803.md)
- 机器审计：`experiments/audit_route_a_frequency_group_visibility_invariance.py`
- 收据：`outputs/tapc/route_a_frequency_group_visibility_invariance/summary.json`
- 单元测试：`tests/test_route_a_frequency_group_visibility_invariance.py`
