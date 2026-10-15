# P5u 运行可观测性观察记录（非终态）

记录时间：2026-08-06T01:10:22Z

## 观察范围

本记录只描述正在运行的 P5u 开发执行现场，不构成结果、失败、outcome freeze 或任何模型性能结论。不得据此调整、重启或替换当前冻结执行。

## 现场事实

- 执行根：`outputs/p5u_direct_supervision_20260806/execution`。
- 已完成 30 个解析特征块；当前仅写入 `pool_000_model_3041_started.json`。
- `pool_000_model_*_completed.json` 数量为 0；`supervisor_terminal.json`、`result.json` 和 `attempt_000_abort.json` 均不存在。
- executor PID 54160 为原 direct supervisor 的子进程，启动于 2026-08-05T17:57:23Z；观察时已运行约 7.22 小时，累计 CPU 时间约 26775.2 秒。
- 先前线程级采样显示一个工作线程在 10 秒采样窗口内消耗约 10 秒 CPU；进程线程数、工作集与私有提交稳定，说明没有证据显示进程已退出或死锁。
- `execution.full_pool_examples=3780` 的含义是每个新身份池的基础身份数。`PairedPowerOracleDataset.__len__()` 将该身份索引在 10/15/20/25/30 dB 五个配对功率视图上展开，因此实际 full-pool DataLoader 行数为 `5 x 3780 = 18900`；已完成的 `30 x 630 = 18900` 个解析位置与此一致。该展开是配对功率评估的既有设计，不得将五个视图视为独立身份或统计重复。
- 随后的线程级 CPU 差分复核中，主工作线程在 15 秒窗口增加 14.984 秒 CPU。一次进程级 I/O/CPU 快照差分未报告增量，但与线程级计数矛盾，不能作为停滞证据；线程级连续 CPU 增长支持“仍在实际计算”。

## 风险分类与行动约束

当前分类为 `running/uncertain — execution observability and cost risk`：单个冻结模型单元长时间没有中间或完成回执，无法产生 P5u 的等价性、延迟或门槛结论。

后续成本报告必须同时报告基础身份数（3780）、配对功率视图数（5）和实际 full-pool 执行行数（18900）；不能把 3780 误表述为实际前向行数，也不能把 18900 误表述为 18900 个独立样本。

P5u 预注册没有以运行时长触发失败/重跑的规则。因此在其出现原子完成、abort 或 supervisor terminal 回执前：

1. 不得将其称作通过或失败；
2. 不得修改 batch、DataLoader、样本数、warm-up、计时重复数、模型或数据；
3. 不得启动确认池、P5-final、P5-U、P5-CD、P5-LT 或 locked test；
4. 仅可继续只读审计 PID、GPU 守卫、资源和回执。
