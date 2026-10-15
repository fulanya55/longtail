# P4i 双 CUDA Stream 紧凑路由失败审计

P4i 只改变执行调度：MLCNN 分类分支和 width=4 IQ-Grouped 参数分支从串行执行改为两个持久 CUDA stream 并行执行，网络、权重和输出定义不变。

并行与串行在 batch=1 和 batch=64 下的分类 logits、12 参数输出均逐元素完全一致，最大绝对差为 0。

## 结果

| 模式 | batch=1 中位延迟 | batch=1 P95 | batch=64 吞吐 |
|---|---:|---:|---:|
| 共享 MLCNN 串行 | 0.9917 ms | 1.1482 ms | 22,411 samples/s |
| compact route 串行 | 1.2532 ms | 1.4144 ms | 20,749 samples/s |
| compact route 双 stream | 1.3270 ms | 1.4807 ms | 23,017 samples/s |

双 stream 在 batch=1 下没有加速，反而比 compact 串行慢 5.89%，相对共享 MLCNN 延迟增加 33.81%，未达到不超过 20% 的门槛。batch=64 下吞吐达到共享基线的 102.70%，且没有额外峰值显存增长。

## 原因

batch=1 时 compact IQ 分支很小，可与主 MLCNN 分支重叠的有效 GPU 工作有限，stream 协调和同步开销超过被隐藏的计算；batch=64 时 kernel 规模增大，并发才产生正收益。因此双 stream 是批吞吐优化，不是通用的低时延优化。

按预注册停止双 stream 路线，不搜索更多 stream 数或同步方式。下一轮对共享基线和 compact route 同时使用固定形状 CUDA Graph，公平验证减少 kernel launch 开销后，近等 MAC 是否能够转化为近等 batch=1 延迟。

本轮未生成数据、未训练模型、未使用 locked test。
