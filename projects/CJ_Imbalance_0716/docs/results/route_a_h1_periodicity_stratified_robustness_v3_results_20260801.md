# Route A H1+B2 development-v3 分层稳健性结果

## 判定

分层门8/10通过，整体 `passed=false`。B2相对H1的弱AUROC在受支持分层中92.86%为正，相对block-shuffle为96.43%正向，说明周期增益广泛存在；失败项是高JNR支持度不足和最坏组合阶数FPR 0.255超过0.25。

最坏一般子组再次为 `FMB present`，FPR 0.265625，与参数OOD-v1的0.25625方向一致。风险在development-v3和外部数据上独立复现，不能作为偶然波动忽略。

## 通过证据

- 弱AUROC正向分层比例：0.9286；
- 弱召回非负分层比例：0.9667；
- 相对block-shuffle弱AUROC正向比例：0.9643；
- 两个极弱相对功率层平均弱召回增益：+0.08108；
- 三个sample-width层满足正向要求；
- 五个receiver cohort满足4/5以上正向；
- 非ISRJ输出完全一致；
- test-v3未使用。

## 未通过与风险

- JNR只有10--15、15--20 dB两层具有最低弱正支持，不能声称覆盖四层；
- 最坏组合阶数FPR 0.255；
- `FMB present`最坏一般子组FPR 0.265625；
- 这些风险表明当前总体FPR约束阈值不能保证共现条件下的最坏风险。

## 下一步

冻结网络、H1、周期专家、融合头和Platt校准器，仅在development-v3 threshold role选择一个全局ISRJ安全阈值，同时约束总体负样本FPR与预注册共现/组合子组FPR。阈值不能使用early或OOD-v1选择。修改后使用新receiver身份的OOD-v2验证，OOD-v1降级为机制诊断数据，不再作为新阈值的独立晋级证据。

## 产物

- 预注册：`docs/plans/route_a_h1_periodicity_stratified_robustness_v3_plan_20260801.md`；
- 分析器：`experiments/analyze_route_a_h1_periodicity_stratified_v3.py`；
- 输出：`outputs/tapc/route_a_h1_periodicity_stratified_v3`；
- test-v3未使用。
