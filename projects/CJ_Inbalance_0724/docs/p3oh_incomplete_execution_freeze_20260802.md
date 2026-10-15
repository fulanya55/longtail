# P3o-H：不完整执行封存

P3o-H（finite-valid bounded compound-CVaR residual）不能形成任何性能结论。

在新身份训练/开发池（1270M/1280M）上，seed 2071 与 2072 已完成三臂训练；seed 2073 的 F-base 训练结束后，原单进程 runner 在启动侧支前退出，未写 `summary.json`，也没有异常回执。因而不存在完整三种子比较，不能把已经产生的 checkpoint 或局部开发输出解释为 CVaR 有效或无效。

该身份池及其输出已消费并永久封存：不修复、不重跑、不调参、不生成确认或测试数据。`test_generated=false`、`test_opened=false`、`locked_test_used=false`。

后续 P3o-I 使用全新 1330M/1340M 身份池和同一有限掩码机制，但采用“每个模型种子一个独立 Python 进程 + 成功/异常回执”的耐久执行协议。P3o-I 是新的独立证据池，不能与 P3o-H 的不完整输出合并。

可复核文件：

- `configs/p3oh_finite_masked_combo_cvar_preregister_20260802.json`
- `outputs/p3oh_finite_masked_combo_cvar_development_20260802/protocol.json`
- `outputs/p3oh_finite_masked_combo_cvar_development_20260802/identity_audit.json`

