# P5m 授权独占窗口：中止回执与结论冻结

## 结论

P5m 未产生可用的性能结果，路径冻结为 **环境干扰中止**。不得从本次运行报告延迟、加速比、输出等价性或 P5 成本结论；也不得在 P5m 的三套开发身份上重跑或调参。

## 已完成且有效的前置证据

- 三套全新 train-only 身份池已生成：每套五档功率、每档 3,780 个身份；15 个 MAT 文件均原子提交，无残留 partial 文件。
- 三套池分别通过数据审计：各自检查 2,556 个本地历史训练文件，均为 0 个不可读文件、0 个身份重叠，且 `valid_for_frozen_inference=true`。
- 启动时 P5h 守卫没有发现外部 Python CUDA 作业。
- 已尝试 `nvidia-smi -c EXCLUSIVE_PROCESS`；驱动返回权限不足，GPU 仍为 `Default`。这是受预注册允许的回退分支，而不是性能结果。

## 中止事实

在第一套池的解析角色特征收集结束后、任何模型结果行或 `result.json` 写入前，P5h 守卫检测到外部 CUDA Python 进程：

`python -m cj0724.run_p3ou_local_phase_dynamics --device cuda`（PID 54480）。

执行器立即抛出异常并只保留 `pre_result_abort.json`。回执中的 `development` 为空，`development_passed=false`，并明确记录 `test_generated=false`、`test_opened=false`、`locked_test_used=false`。

## 证据边界与下一步

P5m 的三套身份池在该次成本试验尝试后全部隔离，不能用于 P5 重跑。若要继续 P5，只允许新建预注册的后继试验，使用全新身份，并满足以下任一条件：实际成功的 `EXCLUSIVE_PROCESS` 会话，或有外部可验证的无其他 CUDA 作业窗口。此前不构造确认池、不进入 locked test，也不改变 P5e 冻结候选。

## 可复核工件

- 预注册：`configs/p5m_authorized_window_shared_dense_preregister_20260803.json`，SHA-256 `f58aa667e5c8042f1af5225b5d6e4308fd42ce31477a72dcf9b33147702bea2d`
- 方法冻结：`configs/p5m_authorized_window_shared_dense_method_freeze_20260803.json`，SHA-256 `8b9d0fd0bc2c3a24c02e8274d9cc79a9cdca41715e18c975972996901e76d9be`
- 中止回执：`outputs/p5m_authorized_window_shared_dense_development_20260803/pre_result_abort.json`，SHA-256 `5c19dff2683c31e8575b8493295775b755c700c5fac902b08bc9aa669b4ef0ce`
- 结果冻结：`configs/p5m_authorized_window_shared_dense_outcome_freeze_20260803.json`
