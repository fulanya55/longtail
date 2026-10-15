# Route A MFT/VFT 共享决策契约：否定结果

## 结论

共享父级校准和 `0.50` 有界残差确实保留了 MFT 的单调安全性，但不能修复 VFT 在 ISRJ 参数 OOD 下的错误来源归因。该机制未通过三个冻结诊断数据集的共同门槛，停止，不进入 fresh development-v13。

## 机制证据

MFT 在 construction-v3、development-v12 和 ISRJ-OOD 上均降低高阶难负 FPR，且弱召回代价受限，说明“父级决策契约 + 负残差”对于 MFT 是正确的安全设计。

但 VFT 的有界残差只获得约 `+0.003` 到 `+0.005` 的弱 AUROC 增益，低于 `+0.010` 门槛；VFT FPR 在 construction-v3/v12 已升至约 `0.218–0.221`，在 ISRJ-OOD 达 `0.3768`。这说明问题不在于校准器或阈值位移，而在于 VFT 物理专家本身无法拒绝长时、密集 ISRJ 产生的相似时频结构。

## 约束

不得继续围绕同一 `0.50` 残差限幅调参。MFT 共享决策策略保留为当前候选 v2 的独立、已验证组件；VFT 的 OOD 问题必须用新的原始物理判别证据重新提出可证伪假设。

## 证据与边界

- 预注册：[route_a_dual_target_decision_contract_plan_20260801.md](../plans/route_a_dual_target_decision_contract_plan_20260801.md)
- 汇总：`outputs/tapc/route_a_dual_target_decision_contract_diagnostic/route_a_dual_target_decision_contract_diagnostic_summary.json`
- `locked_test_used=false`；`test_v3_used=false`。
