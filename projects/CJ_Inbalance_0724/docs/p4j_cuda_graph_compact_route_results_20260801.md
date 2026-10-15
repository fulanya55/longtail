# P4j CUDA Graph 紧凑异构路由结果

P4j 对共享 MLCNN 与 width=4 紧凑异构路由采用相同的固定形状 CUDA Graph 捕获与重放协议，输入复制计入延迟，确保成本比较公平。

## 结果

- 分类 logits 和 12 参数输出在 batch=1、64 下均与 eager 逐元素完全一致；
- 紧凑路由参数量仅比共享 MLCNN 增加 0.052%，MAC 增加 0.343%；
- batch=1：共享 Graph 0.3147 ms，紧凑路由 Graph 0.3270 ms，只增加 3.93%；
- batch=1 P95 只增加 3.11%；
- 紧凑路由 Graph 相对自身 eager 延迟下降 74.04%；
- batch=64 吞吐为 23,922 samples/s，保留共享 Graph 的 96.04%；
- Graph 内存增量门通过，紧凑模型的最大测得增幅为 13.08%。

所有预注册门槛通过。P4h 的单样本延迟失败主要来自 eager 固定启动开销，而不是 MAC；P4i 的双 stream 不能消除该开销，CUDA Graph 则能够把近等 MAC 转化为近等时延。

该结论只适用于固定输入形状。动态 batch 或信号长度必须配置有限的捕获 bucket，或回退到 eager。

下一步冻结 width=4 与 CUDA Graph 部署方式，在全新身份上运行三模型、三种子精度对照。本轮未生成数据、未训练、未使用 locked test。
