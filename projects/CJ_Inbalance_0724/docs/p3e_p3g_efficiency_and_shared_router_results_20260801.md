# P3e–P3g 条件计算与共享物理路由结果

## 总结结论

当前 P1n/P2c 的低开销部署目标尚未成立。三种互补方案均在新开发证据上失败：

1. P3e 对同一组昂贵专家做顺序或 CUDA 多流 lazy 调度；
2. P3f 用共享原始物理前端完全替代 U-Net ISRJ 路由；
3. P3g 用共享前端作为一级 cheap gate，只把残余样本送入 U-Net。

这些结果共同排除了“只需重新安排专家调用即可达到 1.5× 主干延迟”的解释。当前成本瓶颈来自昂贵分支对大量样本仍然必要，以及动态小子批次导致 GPU 利用率下降。若继续推进，必须改变训练期表示或蒸馏架构，不能继续在已消费身份上调 lazy 阈值。

全部试验均为 development，`locked_test_used=false`。

## P3e：冻结并行 lazy 调度

### 设计

- 模型、checkpoint、阈值、路由和预测规则完全冻结。
- 比较 main-only、always-on、sequential lazy、three-stream parallel lazy。
- 数据使用未参与 locked test 的 P3d 新开发波形，覆盖 5 个机制区间和 25/35/50 dB。
- 预注册 batch=64 延迟门槛：parallel lazy ≤1.5× main，且至少比 sequential lazy 快 10%。

### 结果

| 变体 | batch=64 median | 相对 main | 峰值显存 |
|---|---:|---:|---:|
| main-only | 约 4.88 ms | 1.00× | 以结果 JSON 为准 |
| always-on | 13.47 ms | 约 2.76× | 619 MB |
| sequential lazy | 14.20 ms | 约 2.91× | 509 MB |
| parallel lazy | 16.13 ms | 3.30× | 553 MB |

候选覆盖率约为 ISRJ 52.8%、SawFM 46.5%、VFT 48.1%。在 batch 内分别压缩三个子集造成多个小批次 FFT/卷积，吞吐损失超过省下的计算。CUDA 多流还增加工作区和调度开销，没有形成有效核重叠。

子批次与整批次执行存在小幅浮点差异；这属于批尺寸相关数值差异，但预注册的逐特征 `1e-5` 等价门槛仍按失败处理。

## P3f：共享原始物理前端完全替代 U-Net

### 设计

- 新训练身份 seed480000000；新开发身份 seed490000000。
- 输入为 raw-IQ complex ACF、envelope ACF 和 spectrum 的共享 `isrj_periodicity` 特征。
- 固定 leaf-7 HistGradientBoosting；只替换 ISRJ 概率，其他分类和参数输出不变。
- selection 身份上选择逐区间/逐功率风险受限阈值。

### 结果

| 模型 seed | 平均 ISRJ Recall 增益 | 最大 FPR | macro-AUPR delta | 最坏区间增益 |
|---:|---:|---:|---:|---:|
| 2041 | +0.05646 | 0.04839 | +0.00255 | +0.05000 |
| 2042 | +0.01396 | 0.04839 | +0.00249 | +0.00729 |
| 2043 | +0.01979 | 0.04839 | +0.00362 | +0.01146 |

三种子均为正方向、FPR 安全且 AUPR 改善，但只有 seed2041 达到预注册的每种子 +0.05 Recall 门槛。共享前端可作为低成本非劣或小增益候选，不能替代 P1n 的强 U-Net 路由。

## P3g：cheap-to-expensive cascade

### 设计

- 新训练身份 seed530000000；新开发身份 seed540000000。
- 冻结重放完整 P1n，监督目标为“完整 P1n 与安全主干的最终 ISRJ 二值决策是否不同”。
- cheap gate 输入共享 raw-physics 特征和六类主干概率。
- selection 规则要求总体 changed-decision recall≥0.98、每个区间-功率单元≥0.90，并最小化 U-Net 调用覆盖率。

### 结果

| seed | U-Net 覆盖率 | 完整 P1n Recall 增益 | cascade 增益 | 增益保留率 | 最坏单元 FPR |
|---:|---:|---:|---:|---:|---:|
| 2041 | 0.91397 | 0.01729 | 0.01729 | 1.000 | 0.12903 |
| 2042 | 0.48910 | 0.01979 | 0.01563 | 0.789 | 0.10645 |
| 2043 | 0.69746 | 0.03083 | 0.02958 | 0.959 | 0.10000 |

覆盖率、增益保留率和最坏 FPR 联合门槛失败。需要 U-Net 的决策改变事件无法由当前廉价特征在低覆盖率下稳定预判。

## 对创新点的影响

### 仍有可靠支撑

- P1n 的异构任务路由、逐功率有限样本风险校准和 predicted-gate 端到端收益仍是 Level-3 结果。
- 共享物理前端具有低成本、FPR 安全和三种子正方向证据，可作为消融或未来蒸馏教师/学生输入。

### 不能再声称

- 当前 P1n 已解决双骨干计算代价；
- 简单 lazy、CUDA 多流或 cheap gate 能把延迟压到 1.5×；
- 共享物理特征可以无损替代 U-Net。

### 下一条允许的机制路线

只允许在全新身份上测试训练期架构变化，例如：

1. 将共享 ACF/包络 ACF/谱形作为 MLCNN 的轻量并行输入，在主干内部联合产生 ISRJ route score；
2. 用冻结 U-Net/P1n 的 route score 或最终改变事件做表示蒸馏，但学生必须只依赖主干中间特征和共享物理特征；
3. 用等 FLOPs 加宽单骨干作强对照，证明收益来自物理归纳偏置而非额外容量；
4. 先在 development 证明 ≥90% 路由收益保留且 batch64≤1.5×，再做两个 untouched identity seeds 确认。

不得继续调 P3e/P3f/P3g 的阈值、覆盖率或树深度。

## 产物

- `configs/p3e_parallel_lazy_scheduler_preregister_20260801.json`
- `outputs/p3e_parallel_lazy_scheduler_benchmark_20260801.json`
- `configs/p3f_shared_isrj_router_preregister_20260801.json`
- `outputs/p3f_shared_isrj_router_development_20260801/summary.json`
- `configs/p3g_cascaded_isrj_prefilter_preregister_20260801.json`
- `outputs/p3g_cascaded_isrj_prefilter_development_20260801/summary.json`

