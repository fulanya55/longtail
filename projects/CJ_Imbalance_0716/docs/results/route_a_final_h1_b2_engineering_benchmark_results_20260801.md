# Route A 最终 H1+B2 工程开销结果

## 结论

在 RTX 5080、PyTorch 2.11.0+cu128、批量 64、512 个 development-v3 样本上，B2 相对 H1 的周期专家增量推理中位耗时为 12.758 ms，即 0.02492 ms/样本、约 40,132 样本/秒。p95 批耗时为 15.040 ms。增量峰值 CUDA allocated/reserved 分别为 27,215,872 / 46,137,344 bytes。

相同样本和批量在 CPU 上的增量中位耗时为 43.968 ms，即 0.08587 ms/样本、约 11,645 样本/秒。

增量耗时分解：

| 模块 | 中位批耗时 | p95 | 中位 ms/样本 | 吞吐 |
|---|---:|---:|---:|---:|
| 310D FFT/ACF 特征 | 9.082 ms | 10.384 ms | 0.01774 | 56,377 samples/s |
| HGB 预测 | 3.262 ms | 4.029 ms | 0.00637 | 156,957 samples/s |
| 4D 融合头 | 0.600 ms | 0.744 ms | 0.00117 | 853,689 samples/s |
| B2 增量合计 | 12.758 ms | 15.040 ms | 0.02492 | 40,132 samples/s |

## 参数与存储

- Focal-36 主干 state-dict：391,759 参数，1,708,680 bytes。
- H1 六类线性头：774 参数，5,053 bytes。
- RawSaw：5,521 参数，28,817 bytes。
- B2 逻辑融合头：4 个系数 + 1 个截距，共 5 个可训练标量；模型文件 879 bytes。
- 310D 周期 HGB：821,142 bytes；归一化文件 2,745 bytes。

B2 不新增可反向传播的神经主干参数；主要代价来自一个约 0.82 MB 的 CPU HGB 和 FFT/ACF 特征计算。该结构与“低容量物理专家增量”的创新叙述一致。

## 证据边界

这是 seed 2039、单台 RTX 5080、预热后批处理的增量基准，排除数据加载和 H1 基础网络时延，不应解释为端到端单样本实时延迟。test-v3 未访问、未分析、未使用。

## 产物

- 方案：`docs/plans/route_a_final_candidate_engineering_benchmark_plan_20260801.md`。
- 实现：`experiments/benchmark_route_a_final_h1_b2_engineering.py`。
- 测试：`tests/test_benchmark_route_a_final_h1_b2_engineering.py`，1 passed。
- GPU 输出：`outputs/tapc/route_a_final_h1_b2_engineering_benchmark/route_a_final_h1_b2_engineering_benchmark.json`，SHA256 `563C30B2EFC21A6FDE0ED7F1E9F21F0F145C4C1AA7FFC2483A25ADCAE7986149`。
- CPU 输出：`outputs/tapc/route_a_final_h1_b2_engineering_benchmark_cpu/route_a_final_h1_b2_engineering_benchmark.json`，SHA256 `6DFBE74DC5A3A7C495B598EC9837D7AEB7BD0456C0A5E2516F4200808C0F600B`。
