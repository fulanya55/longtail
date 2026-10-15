# Route A H1+B2 全局子组安全阈值结果

## 判定

development门8/9通过，整体失败。全局安全阈值把最坏early子组FPR控制到0.25以内，平均ISRJ FPR从0.194839降至0.156774，且不改变AUPR和弱AUROC；但B2相对H1的弱召回增益只保留39.36%，未达到80%门槛。

五种子安全阈值为0.4031、0.4268、0.4905、0.4197、0.4287。安全候选弱召回0.362075，相对原B2下降0.056027；强召回只下降0.020041，macro-F1只下降0.000386。说明统一提高阈值主要损伤弱正，而高风险实际只集中在部分上下文。

## 结论

“全局阈值过宽”不是完整根因。正确整改不能继续抬高所有样本的阈值，而应使用推理时可观察的共现预测作为风险条件，只对高风险上下文采用更严格阈值。

本轮不得晋级，也不得在OOD-v1上事后选择阈值。下一轮预注册 `FMB predicted-positive` 条件双阈值，并继续只使用development-v3 threshold role选择。

## 产物

- 预注册：`docs/plans/route_a_h1_periodicity_subgroup_safe_threshold_plan_20260801.md`；
- 实现：`experiments/run_route_a_h1_periodicity_subgroup_safe_threshold.py`；
- 输出：`outputs/tapc/route_a_h1_periodicity_subgroup_safe_threshold`；
- 汇总SHA256：`D0EBF90A9057BAA060663DB26B1420FCA2207B954CA6F0C2A7E6869AB85C7A21`；
- test-v3未使用。
