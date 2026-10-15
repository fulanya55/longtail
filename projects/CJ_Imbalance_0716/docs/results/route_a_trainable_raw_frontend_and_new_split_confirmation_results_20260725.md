# 路线 A：可训练 Raw 前端与新划分确认结果（2026-07-25）

## 1. 总结论

本阶段完成了从固定描述符 oracle 到可训练小网络的转换，并用新 split seeds `2039/2040` 完成独立确认。最终结论为：

> 仅使用 RMS 归一化原始 I/Q 的 5521 参数 SawFM 前端，已经稳定恢复弱分量识别；Stage1 分支没有提供足以抵消复杂度的增益。主要判别信息来自局部相位演化，而不是幅度或总功率捷径。

当前决策：`design_bounded_h2_raw_frontend_reconnection`。

这允许下一阶段设计与 H2 的有界接回，但本阶段仍未改变六类正式决策输出，locked test 未使用。

## 2. 三种子可训练结构筛选

### 2.1 网络

每个输入先生成六条物理时序：幅度均值、幅度最大值、幅度标准差、幅度差分、相位增量余弦、相位增量正弦；然后经过 7/31/63 多尺度 depthwise temporal convolution 与 mean/std/max/top-k 双统计旁路。

| 结构 | 参数量 | 输入 |
|---|---:|---|
| N0 Raw-only | 5521 | 原始归一化 I/Q |
| N1 Stage1-only | 5521 | 冻结 F2 Stage1 |
| N2 Raw+Stage1 | 10977 | 两者融合 |

### 2.2 聚合结果

| 结构 | Validation AUROC | Scale AUROC | Development AUROC | Worst-power AUROC | Development AUPR |
|---|---:|---:|---:|---:|---:|
| N0 Raw-only | 0.9989 | 0.9969 | 0.9982 | 0.9977 | 0.9983 |
| N1 Stage1-only | 0.9841 | 0.9831 | 0.9834 | 0.9607 | 0.9842 |
| N2 Raw+Stage1 | **0.9996** | **0.9994** | **0.9995** | **0.9989** | **0.9995** |

N2 的可部署前端门在 `3/3` 种子通过。但 Stage1 互补增量分别为：

- seed 2041：`+0.00279`；
- seed 2042：`+0.00183`；
- seed 2043：`-0.00087`。

均未达到预注册的 `+0.01`，因此不保留 Stage1。选择更简单的 N0 Raw-only。

## 3. 新划分 seeds 2039/2040 确认

新确认完全绕开 F2 checkpoint，直接由 0716 root/power-sweep 根据 seeds 2039/2040 重建 fit、validation、scale、development 四个互斥身份角色。

| 方法 | Validation AUROC | Scale AUROC | Development AUROC | Worst-power AUROC | Development AUPR |
|---|---:|---:|---:|---:|---:|
| R0 Full physics | **0.9983** | **0.9980** | **0.9997** | **0.9993** | **0.9997** |
| R1 Envelope-only | 0.7931 | 0.7780 | 0.7487 | 0.7337 | 0.7098 |
| R2 Phase-only | 0.9973 | 0.9973 | 0.9954 | 0.9950 | 0.9953 |
| R3 Time-shuffled | 0.7402 | 0.7374 | 0.7109 | 0.6652 | 0.6874 |

逐种子 R0：

| Seed | Best epoch | Development AUROC | Worst-power | Scale AUROC |
|---:|---:|---:|---:|---:|
| 2039 | 38 | 0.9996 | 0.9991 | 0.9986 |
| 2040 | 40 | 0.9998 | 0.9994 | 0.9975 |

新划分晋级门 `2/2` 通过。

## 4. 机制结论

### 4.1 不是幅度捷径

输入已经 RMS 归一化，且正负样本按其他五类组合匹配。Envelope-only 仅达到 0.7487 AUROC，明显低于完整输入。因此强结果不是靠总功率或简单包络大小。

### 4.2 主要来自相位演化

Phase-only 达到 0.9954，已接近完整模型 0.9997。这说明 SawFM 的主要可辨识线索是归一化相邻相位增量及其多尺度局部变化。

### 4.3 时间顺序不可删除

对所有样本应用相同固定时间置乱后，AUROC 从 0.9997 降至 0.7109：

- seed 2039 下降 `0.2543`；
- seed 2040 下降 `0.3233`。

因此网络依赖真实局部时间结构，而不是只利用相位增量的边缘分布。

### 4.4 Stage1 不是必要输入

Stage1-only 能识别 SawFM，但低于 Raw-only；Raw+Stage1 的增量很小且不稳定。当前 F2 第一层没有创造关键证据，反而原始物理旁路最直接、稳定、低成本。

## 5. 下一阶段有界接回方案约束

下一阶段只能冻结 Raw frontend 后做接回，不再联合训练主干。候选应包括：

1. H2 原始 SawFM logit；
2. Raw frontend 独立 logit；
3. 有界正残差：`H2 + α*tanh(z_raw)*exp(-|H2|)`，`α∈{0.05,0.10,0.20}`；
4. 只在 H2 低置信且 raw evidence 高时启用的 hard gate。

阈值和 α 只能由 threshold role 选择；scale 与 development 必须保持独立。晋级必须同时约束：

- 40/45/50 dB SawFM AUROC/AUPR；
- 50 dB recall；
- hard-negative FPR；
- 普通六类 macro-AUPR 非劣；
- 其他五类输出完全不变；
- locked test 仍冻结。

## 6. 证据边界

已验证：

- 可训练 Raw-only 前端在五个 seeds 的身份隔离 development 上稳定高分；
- seeds 2039/2040 使用新数据划分确认；
- 主要信号来自局部相位演化；
- 参数量为 5521；
- 没有使用 locked test。

尚未验证：

- 与 H2 接回后的 FPR/召回折中；
- 在完整普通六类样本上的决策非劣性；
- 最终 locked-test 性能；
- 是否能以同一结构迁移到其他弱类。

## 7. 产物

- 训练方案：`docs/plans/route_a_trainable_raw_stage1_sawfm_frontend_plan_20260725.md`
- 新划分确认方案：`docs/plans/route_a_raw_only_new_split_confirmation_plan_20260725.md`
- 模型：`cjlt/models/evidence_readout.py::RawStage1SawFMFrontend`
- 三种子程序：`experiments/run_route_a_trainable_raw_stage1_sawfm_frontend.py`
- 新划分确认程序：`experiments/run_route_a_raw_only_new_split_confirmation.py`
- 三种子汇总：`outputs/tapc/route_a_trainable_raw_stage1_sawfm_frontend/route_a_trainable_raw_stage1_sawfm_frontend_summary.json`
- 新划分汇总：`outputs/tapc/route_a_raw_only_new_split_confirmation/route_a_raw_only_new_split_confirmation_summary.json`
- 三种子汇总 SHA256：`32718F851B2BE66B60B3BBFEDBC6C4B1C82BD76CB2DD9DE8244424E6F707E8F7`
- 新划分汇总 SHA256：`7E71E7395E37B64A1657491E9764B38DE819AA58C2453FB69666229DCE7D93FD`
