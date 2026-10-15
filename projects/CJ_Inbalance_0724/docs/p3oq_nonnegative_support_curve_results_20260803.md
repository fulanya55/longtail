# P3o-Q：全延迟窗非负支持曲线 MFT count 实验

## 结论

P3o-Q 未通过预注册的严格整体门槛，结论为 `stop_no_confirmation_data`。全局非负 in-phase 支持比局部相邻峰关系和后验拒识更稳定地改善低计数与 ultra-close 切片，但仍不能提供足够大的整体增量以成为可靠专家。

## 机制与对照

MFT 由正实幅度的复 LFM 参考信号移位叠加而成。C1 不改变 P3o-M 的 16 个稀疏幅度 token；它只在前 8 个 token 的补充通道中编码全 751 样点反卷积窗口的候选计数 1--8 非负 in-phase 支持曲线：剩余复能量、累计正支持、当前支持相对最大支持以及当前支持相对总能量。C0 的补充通道恒为零。

C0/C1 都使用同一 146,983 参数 ordinal CNN。训练/开发 identity 为 1840M/1850M，各 2,079；跨角色重叠为 0、与 413 份历史身份文件最大重叠为 0，三功率严格配对。仅 MFT oracle-positive；未生成/打开/使用 test 或 locked test。

## 三种子结果

| seed | C1 相对 C0 总体 MAE 改善 | 复合 MAE 改善 | 总体 Pearson 变化 | low-count C1 Pearson | ultra-close MAE 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2331 | +3.31% | +2.85% | +0.0185 | 0.3392 | -0.0797 |
| 2332 | +1.72% | +1.46% | +0.0119 | 0.3590 | -0.0584 |
| 2333 | +1.53% | +1.29% | +0.0111 | 0.3657 | -0.0559 |

low-count Pearson、ultra-close MAE 和“没有制度在全部种子恶化超过 2%”均通过；但每个种子的整体/复合 MAE 均小于 5%，Pearson 增量也小于 +0.03，因此不满足主要准入门。

## 归因与停止

P3o-P 的逐峰复相干和 P3o-Q 的全局正支持均产生一致的小幅正向信号，但都不足以克服全复合波形中的非 MFT 结构。结合 P3o-M/N/O，当前证据支持：在不分离 MFT 分量的前提下，增补局部峰几何、复相干或全局非负支持只改变估计误差的边缘项，不能证明可靠 count 感知创新。

不得在 1840M/1850M 上调曲线、support 规则、模型、损失或训练；不得生成 1860M/1870M 确认。MFT count 专家路线在此冻结，后续系统级创新需要先通过成分可分离性或端到端识别安全门槛，而不是继续优化 count 特征。

关键工件：

- `configs/p3oq_nonnegative_support_curve_preregister_20260803.json`
- `configs/p3oq_nonnegative_support_curve_method_freeze_20260803.json`
- `configs/p3oq_nonnegative_support_curve_outcome_freeze_20260803.json`
- `outputs/p3oq_nonnegative_support_curve_preflight_20260803/data_audit.json`
- `outputs/p3oq_nonnegative_support_curve_development_20260803/summary.json`
