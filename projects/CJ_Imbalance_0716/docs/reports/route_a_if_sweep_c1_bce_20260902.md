# Route-A 新六类 C1：IF>5 标签长尾影响实验

## 1. 协议状态

本实验只从现有六类训练集的 C1 fit 部分无放回选取样本，不重新生成信号，不修改 validation/test。六类为 `FMB、AMA、SawFM、SinFM、MFT、ISRJ`。C1 配置固定为 `F2_DFMS_Full`、真实 I/Q `[B,2,3000]`、BCEWithLogitsLoss、AdamW、lr=`1e-4`、weight decay=`1e-4`、batch=`32`、8 epochs；所有条件使用相同初始化和训练/验证划分。

GPU 审计：RTX 5080。C1 fit=20,160、validation=5,040；测试身份与 fit 无重叠。E5 未修改，`locked_test_used=false`。

固定训练量为 `N=8,000`。Anchor、IF6、IF10 均满足尾类至少 500 个正样本；IF20 只能在尾类最低约 200 个正样本时实现，因此标记为极端探索条件。

## 2. 实际 manifest

| 条件 | 实际 IF | 每类正样本数（FMB/AMA/SawFM/SinFM/MFT/ISRJ） | 实际 CF | 唯一组合数 | 组合漂移 JS |
|---|---:|---|---:|---:|---:|
| Anchor | 1.01–1.05 | 随种子约 4,000/类 | 1.32–1.44 | 63 | 0.00056 |
| IF6 | 6.00 | 5082/2851/4397/4380/4188/847 | 4.43 | 26 | 0.26882 |
| IF10 | 10.00 | 5000/4679/4314/3545/4344/500 | 47.00 | 27 | 0.26673 |
| IF20 | 20.02 | 4205/3632/4205/4205/4205/210 | 337.00 | 28 | 0.26033 |

附加控制 `IF6_combo_matched`：实际 IF=6.00、唯一组合数=63，但由于固定 N 和标签比值约束，CF=329.00；该控制用于判断“标签长尾”和“组合覆盖坍缩”谁是主要原因。

关键事实：IF 增大后，组合覆盖从 63 个降至约 26–28 个，CF 同时急剧增大。因而 IF6 以后不是纯标签长尾，而是“标签长尾 + 组合覆盖漂移”的耦合条件。

## 3. BCE validation 筛查（三种子）

| 条件 | Macro-AUPR | Macro-F1 | 组合准确率 | Hamming loss |
|---|---:|---:|---:|---:|
| Anchor | 0.999993 | 0.999388 | 0.996296 | 0.000628 |
| IF6 | 0.995991 | 0.921465 | 0.669127 | 0.065847 |
| IF10 | 0.994324 | 0.902982 | 0.615608 | 0.081107 |
| IF20 | 0.993664 | 0.898356 | 0.606217 | 0.081349 |

相对 Anchor，IF6 已达到预定义 BCE 瓶颈：

- Macro-F1 下降约 7.46 个百分点；
- 组合准确率下降约 32.15 个百分点；
- Hamming loss 增加约 6.36 个百分点；
- 配对 Wilcoxon（Anchor vs IF6，3 个共同种子）组合准确率、Macro-F1、Hamming 的 `p=0.25`。样本量很小，p 值不显著，但三个种子方向完全一致，且效应量很大。

IF6 validation 的主要错误来自尾类和误报：ISRJ recall 约 0.519，MFT negative-FPR 约 0.117；因此 IF6 是首个稳定失效条件。

## 4. 固定测试集结果

所有条件都使用相同 development-test 来源。全功率筛查采用每功率确定性 512 个样本；IF6 五种子另外对完整 30 dB 测试集（25,200 样本）进行了确认。

### 30 dB

| 条件 | Macro-AUPR | Macro-F1 | 组合准确率 | Hamming loss |
|---|---:|---:|---:|---:|
| Anchor（3 seeds） | 0.999998 | 0.999458 | 0.996745 | 0.000543 |
| IF6（5 seeds，完整 30 dB） | 0.995980 | 0.921628 | 0.673690 | 0.065757 |
| IF10（3 seeds，512 样本） | 0.994411 | 0.897511 | 0.598958 | 0.084310 |
| IF20（3 seeds，512 样本） | 0.993775 | 0.888914 | 0.587240 | 0.084527 |

IF6 完整 30 dB 五种子逐类平均：

| 标签 | AUPR | Recall | Negative-FPR |
|---|---:|---:|---:|
| FMB | 0.9997 | 1.0000 | 0.0477 |
| AMA | 1.0000 | 0.9991 | 0.0000 |
| SawFM | 0.9968 | 0.9740 | 0.0379 |
| SinFM | 0.9998 | 0.9993 | 0.0379 |
| MFT | 0.9899 | 0.9911 | 0.1226 |
| ISRJ | 0.9896 | 0.4982 | 0.0000 |

组合匹配控制的三种子 validation 为 Macro-AUPR=0.999906、Macro-F1=0.984777、组合准确率=0.915873、Hamming=0.015046；同样 IF=6 的普通子集分别为 0.995991、0.921465、0.669127、0.065847。控制的 30 dB、512 样本/种子测试组合准确率为 0.8750、0.8965、0.8809（均值约 0.8841），显著高于普通 IF6 的 0.6737。

在三个共同种子上，组合匹配控制相对普通 IF6 的组合准确率提升 0.2411、Macro-F1 提升 0.0600、Hamming 降低 0.0492；配对 Wilcoxon 因 n=3 均为 `p=0.25`，因此这里报告为强效应的机制证据，不宣称统计显著。

### 功率分层筛查

| 条件 | 10 dB Macro-F1 / 组合准确率 | 30 dB Macro-F1 / 组合准确率 | 50 dB Macro-F1 / 组合准确率 |
|---|---:|---:|---:|
| Anchor | 0.6943 / 0.2422 | 0.9995 / 0.9967 | 0.5609 / 0.2578 |
| IF6 | 0.6841 / 0.2195 | 0.9110 / 0.6391 | 0.4382 / 0.2285 |
| IF10 | 0.6832 / 0.1960 | 0.8975 / 0.5990 | 0.5759 / 0.2311 |
| IF20 | 0.6887 / 0.2201 | 0.8889 / 0.5872 | 0.4510 / 0.2331 |

标签长尾对 30 dB 的影响最清晰；在 10 dB 和 50 dB，原始可观测性/功率因素已经很困难，IF 效应会被低功率和高功率噪声掩盖，不能只用这些功率下的变化定义长尾结论。

## 5. 证据边界与判定

1. 普通 BCE 的首个可靠失效边界为 `IF≈6`，但该条件伴随 CF 从约 1.4 增至约 4.4、组合覆盖从 63 降至 26，严格说是标签—组合耦合失效，而不是纯标签边际效应。
2. IF10 和 IF20 没有继续显著恶化 30 dB 组合准确率，主要原因是 IF6 后组合覆盖已经严重坍缩；不能据此认为 IF10/IF20 比 IF6 更容易。
3. ISRJ 是最敏感的尾类：IF6 完整 30 dB recall 约 0.498，而 Anchor 约 0.997；MFT 的误报也明显增加。
4. IF20 只有 210 个尾类正样本，适合作为极端探索，不适合作为高置信度主结论。
5. 当前没有进行任何长尾损失函数比较；本阶段只用于建立普通 BCE 的数据边界，后续方法改进必须在 IF6 瓶颈上进行，并使用相同 manifest 和固定测试角色。

## 6. 成本与完整性审计

所有模型均为 257,511 参数的同一 F2 backbone，manifest 选择均无重复样本。单次 validation 训练耗时约 40–77 秒（RTX 5080）；测试评估单独分块执行，避免一次性加载 30 dB 大文件造成内存耗尽。原始 MAT 源文件哈希、manifest 哈希、checkpoint 哈希、运行耗时和 `locked_test_used/test_files_opened` 已写入 JSON 收据。

## 7. 组合匹配控制与下一步

组合匹配控制已经完成。它在同样 IF=6 下恢复了大部分组合准确率，说明普通 IF6 的 32 个百分点组合性能损失主要来自组合覆盖坍缩；标签边际长尾本身的影响相对较小，但仍表现为部分尾类 recall/FPR 变化。由于控制的 CF=329，尚不能称为纯标签因果实验，只能作为“组合覆盖因素被部分隔离”的证据。

下一步应在保持更多组合、降低 CF 的约束下重新做多目标可行性审计；之后再比较 Class-Balanced BCE、Positive-Focal、Logit Adjustment 或组合感知损失。当前不应把 IF6 普通子集结果直接用于证明纯标签长尾创新。

收据：

- [IF 可行性](<E:/LJQ_code/CJ_Imbalance_0716/outputs/route_a_if_sweep_20260902/feasibility.json>)
- [Manifest 汇总](<E:/LJQ_code/CJ_Imbalance_0716/outputs/route_a_if_sweep_20260902/manifest_summary.json>)
- [运行汇总](<E:/LJQ_code/CJ_Imbalance_0716/outputs/route_a_if_sweep_20260902/runs/summary.json>)
- [测试汇总](<E:/LJQ_code/CJ_Imbalance_0716/outputs/route_a_if_sweep_20260902/test_eval/summary.json>)
- [最终聚合与漂移分析](<E:/LJQ_code/CJ_Imbalance_0716/outputs/route_a_if_sweep_20260902/final_summary/summary.json>)
- [IF 影响曲线](<E:/LJQ_code/CJ_Imbalance_0716/outputs/route_a_if_sweep_20260902/final_summary/if_effect_curves.png>)
- [组合漂移图](<E:/LJQ_code/CJ_Imbalance_0716/outputs/route_a_if_sweep_20260902/final_summary/combination_drift.png>)
- [最终聚合汇总](<E:/LJQ_code/CJ_Imbalance_0716/outputs/route_a_if_sweep_20260902/final_summary/summary.json>)
