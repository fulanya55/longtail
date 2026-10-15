# P5o 异常终止：无可用性能结果

P5o 在协议泛化校验、方法冻结、三套 train-only 身份池生成及零重叠审计之后启动。运行进程随后消失，但其预先创建的输出目录为空：既没有正常完成时的 `result.json`，也没有可捕获 Python 异常时应留下的 `pre_result_abort.json`。

因此，本轮的唯一结论是执行基础设施发生了无法由现有本地日志归因的异常终止；没有输出等价性、批延迟、GPU guard 或分类性能结果，不能支持或反驳 shared-forward 的成本假设。针对结束时段的 Application Error、Windows Error Reporting 与相关 System 事件查询未检出可明确归属于 P5o 的 Python/CUDA 故障。

三套 P5o 身份池（3910000000、3920000000、3930000000）全部隔离，不得重跑、挑选、确认、校准或用于测试。未生成、打开或读取任何独立/locked test。

后续如继续 P5，只能建立一个新的、独立预注册的 checkpointed 执行器：在高耗时阶段前落盘启动凭据，并在每个 pool 完成后落盘 checkpoint；先以不涉及性能结论的恢复行为测试验证它，再生成全新的身份池。P5o 本身永久停止。
