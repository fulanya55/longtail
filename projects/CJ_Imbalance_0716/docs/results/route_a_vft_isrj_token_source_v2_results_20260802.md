# Frozen F2 token screen：部分机制支持，未晋级

在新鲜 source-v2（seed 181000000、63×80、五种子严格角色）上，冻结 F2 的 12-bin temporal token Ridge 与 per-record 时间置乱等容量对照比较显示：Token12 相对置乱 early AUROC 平均 `+0.13226`，5/5 为正，表明 token 顺序保留真实判别信息；但 Token12 相对 Global128 平均仅 `+0.01109`，正方向 3/5（逐 seed：`-0.04941,-0.00127,+0.01909,+0.04536,+0.04169`），未满足预注册的至少 4/5 正方向门槛。

结论：不把当前固定 bin token Ridge 接回候选，也不授权 test-v3。该结果仅支持“时间顺序信息存在”，不能证明它足以稳定超过全局 F2 表征。后续若继续，应在新的 development 数据上比较**标签条件的低容量 token 查询**与完全同参数的无条件 token pooling，而不是增加 token bin 数或在 source-v2 上调 alpha/bin 数。

全部五个正式单-seed receipt 位于 `outputs/tapc/route_a_vft_isrj_token_source_v2_per_seed/`；`locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。
