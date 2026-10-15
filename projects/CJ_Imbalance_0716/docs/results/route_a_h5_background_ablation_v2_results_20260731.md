# Route A：development-v2 H5 SawFM 背景消融结果（2026-07-31）

## 1. 结论

H5 在五个种子上都小幅提高了 MFT/VFT/ISRJ 的平均弱分量召回，但效应量不足，
未通过晋级门槛，禁止据此解冻 F2 或进入 test-v2。

最重要的判断是：**SawFM 作为第六个输出造成了轻微的分类头竞争，但它不是
MFT/VFT/ISRJ 弱分量召回不足的根因。** 删除 SawFM 输出后，五标签整体与三个
目标标签均只有千分位到百分位以内的改善，剩余瓶颈更可能位于冻结 F2 的共享
时序表示和缺少标签条件局部证据搜索。

## 2. 数据与证据边界

- development-v2：基础种子 20260731，25200 条、63 种组合、每组合 400 条。
- development-v2 SHA-256：
  `f05a7e30253b0c0fdf10d5040487cb6a6f089abcd48f9cbd6f9e467624498801`。
- test-v2：25200 条，已生成并封存；SHA-256：
  `902e636911e76d6915fe04976f59cc46a588b3d98926bb9a48e2200628617947`。
- test-v2 未进入训练、特征提取、校准、阈值选择或指标计算。
- 历史 locked test 未使用；所有实验 receipt 均记录
  `locked_test_used=false`、`test_v2_used=false`。

## 3. 实验设置

- 冻结表征：历史 development 阶段 F2_DFMS_Full，种子 2039--2043。
- C6：`Linear(128,6)` + 正样本加权 BCE。
- H5：`Linear(128,5)` + 正样本加权 BCE；标签为
  `[FMB, SinFM, MFT, VFT, ISRJ]`。
- H5 保留所有 SawFM 存在及共现样本，仅删除 SawFM 输出和对应损失。
- 每种子重新构造 2400 条相同目标计数的 label-LT 训练 manifest。
- AdamW，lr=1e-3，weight decay=1e-4，batch=256，最多 60 epoch，
  patience=12，gradient clip=5.0。
- calibration role 拟合温度；threshold-selection role 在逐标签负样本
  FPR<=0.20 下最大化弱分量召回；early role 报告 development 指标。

## 4. 五种子聚合结果

| 指标 | C6 | H5 | H5-C6 | 预注册要求 |
|---|---:|---:|---:|---:|
| 五标签 macro-AUPR | 0.87681 | 0.87827 | +0.00146 | >= -0.005 |
| 五标签 macro-F1 | 0.76932 | 0.77176 | +0.00244 | >= +0.010 |
| MFT/VFT/ISRJ 平均弱召回 | 0.33176 | 0.34018 | +0.00842 | >= +0.030 |
| 三目标平均负样本 FPR | 0.19622 | 0.19625 | +0.00002 | H5 <= 0.20 |
| SawFM-present 五标签 macro-F1 | 0.71796 | 0.72185 | +0.00390 | 正增益 |
| SawFM-absent 五标签 macro-F1 | 0.82549 | 0.82655 | +0.00106 | >= -0.005 |

弱召回增益在 5/5 种子为正；三目标标签最差 AUPR 差值为 -0.00471，未发生
明显排序能力退化。但是 macro-F1 与弱召回效应量均低于晋级要求，所以总门
判断为 `promoted=false`。

## 5. 逐目标标签结果

| 标签 | C6 AUPR | H5 AUPR | C6 F1 | H5 F1 | C6 弱召回 | H5 弱召回 | H5 FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFT | 0.90054 | 0.90118 | 0.79732 | 0.79832 | 0.35988 | 0.36245 | 0.18368 |
| VFT | 0.84948 | 0.85082 | 0.71930 | 0.72304 | 0.30820 | 0.32018 | 0.19936 |
| ISRJ | 0.86666 | 0.86838 | 0.73520 | 0.73877 | 0.32721 | 0.33792 | 0.20570 |

VFT 和 ISRJ 的弱召回增益相对更明显，但 ISRJ 在 early role 的平均 FPR
仍为 0.20570。阈值约束是在独立 threshold-selection role 上施加，因此该结果
说明存在小幅角色间泛化漂移，不能通过继续在 early role 调阈值来修复。

## 6. 原因分析

1. **输出竞争仅是次要因素。** H5 在 SawFM-present 和 SawFM-absent 子集都
   有近似的小幅改善，说明收益更像减少一个共享优化任务后的轻微方差下降，
   而不是解除强烈的 SawFM 条件负迁移。
2. **冻结表示决定了收益上限。** C6/H5 均只能使用同一 128 维全局 F2 表征；
   若弱局部证据已在共享池化前被抑制，删除一个输出无法恢复信息。
3. **排序能力已有但弱证据分离间隔小。** AUPR 没有明显变化，阈值附近的
   小幅分数移动只能以接近 0.20 的 FPR 换取有限弱召回。
4. **当前不支持直接解冻主干。** 头部干预没有达到效应量门槛，直接联合微调
   会把分类头的微小收益和表示学习变化混在一起，无法形成清楚证据链。

## 7. 下一步建议

进入冻结时序特征上的“标签条件局部证据”验证，而不是扩大主干：

- 从 F2 池化前 `[B,128,T]` 时序特征读取证据；
- 每个标签独立生成时间权重，并并联无注意力 mean/std 安全旁路；
- 增加 top-k 与局部最大统计，先仅训练六个标签条件 readout；
- C6 线性头作为固定基线，训练角色、校准和阈值规则保持不变；
- 预注册要求至少使 MFT/VFT/ISRJ 平均弱召回提升 0.03，同时维持
  macro-AUPR 非劣和 development FPR 稳定；
- 只有冻结 readout 通过后，才允许以 0.1 倍学习率解冻最后一个多尺度阶段；
- test-v2 继续封存，直到结构与全部超参数在 development-v2 上一次性冻结。

## 8. 产物

- 计划：`docs/plans/route_a_development_v2_h5_plan_20260731.md`
- 数据生成：`experiments/generate_route_a_v2_dataset.m`
- 实验入口：`experiments/run_route_a_h5_background_ablation_v2.py`
- 数据封存：
  `outputs/tapc/route_a_development_v2_generation/test_v2_seal_receipt.json`
- 聚合结果：
  `outputs/tapc/route_a_h5_background_ablation_v2/route_a_h5_background_ablation_v2_summary.json`
- 每种子目录保存 split、manifest、checkpoint、history、calibration、threshold、
  predictions 和 metrics。

