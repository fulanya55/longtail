# P6a-v1 身份池无效记录

P6a-v1 的模型、损失和所有训练超参数均**没有运行**。在数据生成、重分块和全历史身份审计后，发现其生成种子 `1030000000` 与已消费的 `p3oc_parameter_matched_train_seed1030000000_*` 身份池冲突。

- 新池：6,300 个唯一 `receiver_noise_seed`；
- 与每个 P3oc 条件池的重叠：2,079 个 identity；
- 结论：`all_excluded_overlap_zero=false`，该池不得用于训练、选择、阈值拟合、确认或测试；
- P6a-v1 仅保留为数据生成失败审计，不构成任何方法结果；
- 没有读取 locked test，`test_generated=false`、`test_opened=false`、`locked_test_used=false`。

依据预注册停止规则，后续只能以新的生成种子重新建立身份池，且网络、特征、损失、门槛和模型随机种子不得调整。

审计工件：`outputs/p6a_physical_film_full12_20260801/data_audit.json`。
