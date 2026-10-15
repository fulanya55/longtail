# P5p：迁移相关的首池异常终止，无性能结果

P5p 在原 `C:\LJQ_code\CJ_Inbalance_0724` 路径启动，并留下 launch、environment 和 `pool_000_started` 三类不可变回执。随后项目的 C/D 位置被移除，保留项目迁移至 `E:\LJQ_code\CJ_Inbalance_0724`。当前原进程不存在，且没有 pool-completed、abort、restore 或 result 回执。

因此 P5p 没有输出等价性或延迟结论。项目迁移与进程消失存在时间关联，但现有证据不足以把迁移写成已证明的操作系统终止原因。

P5p 相比 P5o 已取得一个基础设施成果：异常不再表现为空目录，至少可以确认运行通过了初始 guard 并进入首池。但 checkpoint 仍以整池为单位；解析特征和五模型全部完成前没有可恢复科学单元，粒度仍过粗。

三个 P5p 开发身份全部隔离，不重跑、不恢复、不用于选择或确认。下一轮只能修改路径可移植性和 checkpoint 粒度，不得改变 P5e candidate、模型、特征、batch、计时次数、输出等价门或延迟门。
