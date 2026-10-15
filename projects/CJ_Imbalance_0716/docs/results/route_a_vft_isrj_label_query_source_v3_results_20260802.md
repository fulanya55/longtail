# Label-query token source-v3：失败并停止

五个正式 seed 的冻结 F2、385 参数 VFT query-head、385 参数无条件 mean-head 和 Global128 Ridge 结果已完成。query 相对 Global128 early AUROC 差为 `+0.01826,+0.00391,-0.04299,-0.03480,-0.00475`，均值 `-0.01207`，仅 2/5 为正；query 相对同参数 mean-head 均值 `-0.01101`，仅 1/5 为正。

因此全部主要门槛失败：没有相对 global 的 +0.010/4-of-5 增益，也没有相对无条件对照的 +0.015/4-of-5 增益。该结果否证“当前冻结 F2 token 的 VFT 标签条件 softmax 查询可稳定恢复被 global pooling 抹去的 VFT--ISRJ 证据”。

停止动作：不把 query 接回统一候选；不调 query 温度、维度、学习率、训练轮数或早停；不将 source-v3 用于任何后续结构选择；不访问 test-v3。此前 token 对时间置乱敏感的事实仍只说明顺序信息存在，并不足以支持该标签条件查询机制。

五个 receipt 位于 `outputs/tapc/route_a_vft_isrj_label_query_source_v3_per_seed/`。全部 `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。
