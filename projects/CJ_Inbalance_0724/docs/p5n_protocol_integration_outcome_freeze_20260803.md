# P5n 协议集成不兼容：执行前冻结

P5n 的三套 train-only 开发身份池已完整生成并通过零重叠审计，但在任何 CUDA 模型推理开始前发现协议—执行器不兼容，因此本轮没有性能结果。

P5n 预注册意图复用 P5m 执行器；但该执行器固定验证 `P5l_preregister`、`P5l_outcome_freeze`、`P5l_runner` 和 `P5l_generator` 字段，而 P5n 以正确的 P5m 来源字段描述其前序中止。运行将导致配置键错误。数据已经生成后，修改协议或替换执行器都会破坏“方法冻结在数据前”的边界，因此两者均不执行。

三套池均只含五个配对功率 view 的 train 数据；所有审计均通过，且 `outputs/p5n_verified_idle_window_shared_dense_development_20260803` 在 GPU 执行前不存在。没有生成、打开或读取任何独立/locked test。

P5n 数据全部隔离。下一次 P5 后继必须在数据生成前，通过一个会实际调用自身父哈希校验的协议集成测试；不能再只检查 JSON 字段或局部 helper。
