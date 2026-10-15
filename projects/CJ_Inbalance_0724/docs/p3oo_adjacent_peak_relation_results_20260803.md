# P3o-O：相邻延迟峰关系 MFT count 实验

## 结论

P3o-O 未通过冻结的三种子开发门槛，结论为 `stop_no_confirmation_data`。相邻可见峰的局部关系在这个数据契约下没有给 MFT count 提供可靠增量信息。

## 对比与边界

- 仅使用 MFT oracle-positive 样本；未使用分类损失、预测标签门控、测试或 locked test。
- C0/C1 均先使用 P3o-M 冻结的 16 个严格局部 contrast 峰和相同 8 通道 146,983 参数 ordinal CNN。
- C0 的后 4 通道为零；C1 的后 4 通道仅为下一有效峰的归一化幅度差、对比度、尖锐度差和峰间延迟。
- 训练/开发身份为 1760000000/1770000000，各 2,079 个；跨角色重叠为 0，且与 393 份历史身份文件的最大重叠为 0。三功率视图按身份配对。

## 三种子结果

| seed | C1 相对 C0 总体 MAE 改善 | 复合 MAE 改善 | 总体 Pearson 变化 | C1 low-count Pearson | ultra-close MAE 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2311 | -0.75% | -0.82% | -0.0018 | 0.3212 | -0.0253 |
| 2312 | -0.85% | -0.95% | +0.0010 | 0.3302 | -0.0200 |
| 2313 | -1.12% | -0.99% | -0.0085 | 0.2718 | +0.0650 |

所有 C0/C1 参数量严格相同，且没有任何 regime 在三种子中均恶化超过 2%；然而，C1 未达到整体与复合 MAE 至少 5%、Pearson 至少 +0.03 的门槛，第三种子的 low-count 相关性与 ultra-close MAE 门槛也失败。

## 失败归因

P3o-M 已经按时间顺序提供局部峰的位置与强度；把相邻差分显式加入模型并没有创造新的可观测分辨信息，反而轻微扩大了低信噪比峰定位误差。结合 P3o-N 的后验间距/强度拒识失败，可否定以下两条在当前复合波形下的简单假设：

1. 相邻峰的显式间距和幅度关系足以修复 count 的低计数不稳定性；
2. 这些关系可在点预测之后作为可靠风险分数使用。

不得在 1760M/1770M 上调整 relation 定义、峰规则、网络、损失、训练或选择，也不得生成 1780M/1790M 确认身份。

关键工件：

- `configs/p3oo_adjacent_peak_relation_preregister_20260803.json`
- `configs/p3oo_adjacent_peak_relation_method_freeze_20260803.json`
- `configs/p3oo_adjacent_peak_relation_outcome_freeze_20260803.json`
- `outputs/p3oo_adjacent_peak_relation_preflight_20260803/data_audit.json`
- `outputs/p3oo_adjacent_peak_relation_development_20260803/summary.json`
