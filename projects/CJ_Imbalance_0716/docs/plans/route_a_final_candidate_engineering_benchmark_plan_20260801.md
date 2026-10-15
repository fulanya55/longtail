# Route A 最终候选工程开销基准方案

## 目的与边界

量化最终候选 `H1 + B2 aligned periodicity` 相对 H1 的增量推理开销，不改变权重、校准器或阈值，不访问 test-v3。

## 固定协议

- seed 2039 development-v3 early role 前 512 个样本；只用于性能测量。
- GPU 批量 64；2 次 warmup、10 次正式重复。
- 分别计时：310D 特征提取（含 GPU 计算及回传）、HGB 预测、4 维融合头。
- 每次 GPU 计时前后同步；报告 median、p95、每样本时延和吞吐。
- 静态报告 Focal-36 主干、H1 线性头、RawSaw 的 state-dict 参数量与文件大小，以及周期 HGB、融合头的序列化大小/系数数量。
- 保存 Python、PyTorch、CUDA、GPU、输入规模、哈希和原始重复值。

该结果是单机工程证据，不用于准确率选择，也不替代多硬件部署测试。

