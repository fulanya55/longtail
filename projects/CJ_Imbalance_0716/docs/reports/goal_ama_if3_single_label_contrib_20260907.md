# AMA IF3 六类冻结 logits：单标签 exact 损失贡献分解

## 1. 实验边界

本试验复用上一次实验已经冻结的五个 seed（3050–3054）、epoch-8 checkpoint 对应的 test logits、标签和阈值参数，不重新训练、不重新选择 checkpoint、不用 test 标签调参，也不修改原始 MAT 或历史产物。

主分析为 `combo_exact_coordinate_threshold`，`fixed_0.5` 作为敏感性对照。标签顺序为 `[FMB, AMA, SawFM, SinFM, MFT, ISRJ]`。

这里的“贡献”是冻结预测上的 oracle 误差归因，不是因果效应：将指定错误位置直接改成真实标签，观察 exact 能恢复多少。四个因素定义为：

- `FMB_FP`：真实 FMB=0、预测 FMB=1；
- `SawFM_FN`：真实 SawFM=1、预测 SawFM=0；
- `MFT_FN`：真实 MFT=1、预测 MFT=0；
- `ISRJ_FN`：真实 ISRJ=1、预测 ISRJ=0。

四因素枚举全部 16 个子集，并使用 Shapley 分解避免多个因素在同一样本上同时出错时重复归因。

## 2. 主结果：组合坐标阈值

五 seed 的基础 test sample exact 为 `0.575450 ± 0.005551`。四因素的结果如下：

| 因素 | 平均错误数 | 单因素 oracle exact gain | Shapley exact gain | 正向 seed 数 |
|---|---:|---:|---:|---:|
| FMB_FP | 267.4 | 0.020100 ± 0.005393 | 0.024127 ± 0.005862 | 5/5 |
| SawFM_FN | 752.8 | 0.055100 ± 0.010760 | 0.065523 ± 0.011464 | 5/5 |
| MFT_FN | 553.0 | 0.038675 ± 0.002883 | 0.044590 ± 0.003323 | 5/5 |
| ISRJ_FN | 850.0 | 0.060075 ± 0.005989 | 0.068835 ± 0.007387 | 5/5 |

四因素同时 oracle 修正后的 exact 提升为 `0.203075 ± 0.006655`，即从约 `0.57545` 提升到约 `0.778525`。四因素可以解释的 exact 错误损失比例为 `0.478367 ± 0.015665`；四因素修正后仍有 `0.221475 ± 0.007873` 的 exact 损失残余。

Shapley 贡献排序为：

1. `ISRJ_FN`：`0.068835`；
2. `SawFM_FN`：`0.065523`；
3. `MFT_FN`：`0.044590`；
4. `FMB_FP`：`0.024127`。

四因素存在明显重叠，不能直接将单因素 gain 相加。组合坐标阈值下，平均两两错误掩码交集/exact 交互项为：

| 因素对 | 平均重叠样本数 | exact 交互项 |
|---|---:|---:|
| FMB_FP + SawFM_FN | 38.0 | 0.003375 |
| FMB_FP + MFT_FN | 14.2 | 0.000900 |
| FMB_FP + ISRJ_FN | 34.6 | 0.003000 |
| SawFM_FN + MFT_FN | 82.8 | 0.006375 |
| SawFM_FN + ISRJ_FN | 108.8 | 0.009800 |
| MFT_FN + ISRJ_FN | 50.6 | 0.003525 |

其中 SawFM_FN 与 ISRJ_FN 的重叠和交互最大，说明这两个因素经常共同出现在同一条组合错误样本中；Shapley 分配正是为避免把这种共同错误重复算给两个标签。

因此，在当前组合坐标阈值下，SawFM/MFT/ISRJ 漏检的联合贡献明显高于 FMB 误报；ISRJ 和 SawFM 是最主要的两个指定误差因素。

## 3. fixed 0.5 敏感性对照

fixed 0.5 的基础 exact 为 `0.564500 ± 0.004525`：

| 因素 | 平均错误数 | 单因素 oracle exact gain | Shapley exact gain | 正向 seed 数 |
|---|---:|---:|---:|---:|
| FMB_FP | 481.2 | 0.035100 ± 0.008940 | 0.042546 ± 0.009627 | 5/5 |
| SawFM_FN | 955.4 | 0.068025 ± 0.012902 | 0.083083 ± 0.015746 | 5/5 |
| MFT_FN | 677.8 | 0.047825 ± 0.009433 | 0.056433 ± 0.010362 | 5/5 |
| ISRJ_FN | 831.2 | 0.059175 ± 0.010357 | 0.070038 ± 0.012688 | 5/5 |

四因素同时 oracle 的 exact 提升为 `0.252100 ± 0.017468`，四因素可解释 exact 损失比例为 `0.579156 ± 0.044627`，修正后的残余 exact 损失为 `0.183400 ± 0.020849`。

阈值校准减少了 FMB 误报和部分指定漏检，因此组合坐标阈值下四因素的绝对可恢复空间变小；这不是矛盾，而是说明阈值校准已经提前消除了一部分这些错误。

## 4. 六个完整标签 oracle

“完整标签 oracle”同时修正该标签的 FP 和 FN，因此不能与四因素中的单一错误类型直接等同。主分析组合坐标阈值下的平均 exact gain 为：

| 标签 | 完整标签 oracle gain |
|---|---:|
| FMB | 0.042700 |
| AMA | 0.011325 |
| SawFM | 0.096875 |
| SinFM | 0.035525 |
| MFT | 0.057225 |
| ISRJ | 0.074350 |

这进一步表明 SawFM 和 ISRJ 是当前单标签识别对 exact 影响最大的类别，MFT 次之；AMA 的完整标签 oracle 增益很小。

## 5. 误差重叠与解释边界

单因素 oracle gain 不能直接相加，因为同一条样本可能同时存在 FMB 误报、SawFM 漏检、MFT 漏检和 ISRJ 漏检。四因素 Shapley 值将所有加入顺序平均，因此满足：

```text
四因素 Shapley gain 之和 = 四因素同时 oracle gain
```

五个 seed、两种阈值方法均满足该加和关系，数值误差在浮点精度范围内。

这些结果只能说明冻结预测中的错误归因，不能证明“修复某个标签一定会因果地提高模型性能”。四因素之后的残余 exact 损失来自：AMA/SinFM 错误、四因素之外的 FMB FN 或其他 FP、多个错误未被定义因素覆盖，以及特征不可分性。

## 6. 结论

在当前新六类 AMA IF3 数据上，组合全正确率的主要指定误差来源是：

1. ISRJ 漏检；
2. SawFM 漏检；
3. MFT 漏检；
4. FMB 误报。

这支持“当前 exact 瓶颈首先来自单标签识别边界”的判断。四因素总共只能解释组合坐标阈值下约 47.8% 的 exact 损失，仍有约 52.2% 无法由这四类错误单独解释。因此下一步应优先针对 ISRJ、SawFM、MFT 的单标签召回和 FMB 的误报边界做局部诊断，而不是直接把全部剩余问题归因于联合组合建模。

本结果使用的是已冻结 development final confirmation 的 test logits，不能升级为 locked-test 结论。

## 7. 收据

- [Goal prompt](E:/LJQ_code/CJ_Imbalance_0716/outputs/goal_ama_if3_single_label_contrib_20260907/goal_prompt.md)
- [配置](E:/LJQ_code/CJ_Imbalance_0716/outputs/goal_ama_if3_single_label_contrib_20260907/config.json)
- [完整汇总](E:/LJQ_code/CJ_Imbalance_0716/outputs/goal_ama_if3_single_label_contrib_20260907/summary.json)
- [Shapley 子集矩阵：组合阈值](E:/LJQ_code/CJ_Imbalance_0716/outputs/goal_ama_if3_single_label_contrib_20260907/subset_matrix_combo_exact_coordinate_threshold.json)
- [分解图](E:/LJQ_code/CJ_Imbalance_0716/outputs/goal_ama_if3_single_label_contrib_20260907/single_label_contribution.png)
- [执行脚本](E:/LJQ_code/CJ_Imbalance_0716/experiments/goal_ama_if3_single_label_contrib_20260907.py)
