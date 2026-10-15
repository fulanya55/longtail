# 新六类 AMA IF3 数据：BCE 与组合全正确指标一致性验证

## 1. Goal 与数据边界

本次验证使用新六类数据：`[FMB, AMA, SawFM, SinFM, MFT, ISRJ]`，数据目录为
`E:\论文相关\干扰数据\复合干扰数据0716\six_class_pure_label_longtail_v2_IF3_CF8_N8000_AMA`。

数据审计已确认：训练集和测试集各 8,000 条，均覆盖 63 个非空组合；完整训练集实际 IF=3.0、CF=7.853658536585366、JS=0.09552437893063263；训练/测试 `receiver_noise_seed` 交集为 0。训练集 SHA-256 为 `dd7ee3f823e88174e347dca6e57d2d49c48d9d30f8c86c7ea707eefbe3411da6`，测试集 SHA-256 为 `360e5e682df279902df550336bc6f008c46b459e6c5efc51aaf225a8f1066e52`。

本次实验输出目录为 [goal_ama_if3_bce_exact_20260905](E:/LJQ_code/CJ_Imbalance_0716/outputs/goal_ama_if3_bce_exact_20260905)。测试集在本次脚本中只在所有 checkpoint、阈值和校准参数冻结后载入；但该 MAT 在历史 AMA 开发脚本中曾被读取，因此本报告属于 development final confirmation，不是未消费的 locked-test 证据。

## 2. 角色拆分与可复现配置

训练集按组合分层拆分为：

| 角色 | 样本数 | 唯一组合 | 实际 IF | 实际 CF |
|---|---:|---:|---:|---:|
| fit | 3,979 | 63 | 3.0180 | 8.0500 |
| early-stop | 1,632 | 63 | 2.9222 | 7.2222 |
| threshold-selection | 1,197 | 63 | 3.0360 | 8.1667 |
| check | 1,192 | 63 | 3.0120 | 8.0000 |

四个角色的 `receiver_noise_seed` 交集全部为 0。使用普通 `BCEWithLogitsLoss`、F2_DFMS_Full、AdamW、8 epoch、5 个新种子 `3050–3054`。完整配置、角色索引、索引哈希、逐 epoch logits、逐 epoch checkpoint 和阈值收据均保存在输出目录中。

## 3. 评估器一致性审计

在 threshold-selection、check 和最终 test 的每一个 seed/策略/阈值方法上，均逐批断言：

```text
subset_accuracy
combination_exact_accuracy
np.mean(np.all(preds == targets, axis=1))
```

三者最大绝对差为 `0.0`。因此当前实现中的 sample-level exact 定义没有发现评估器不一致或编码错误。

同时明确区分：

- `sample_exact_accuracy`：按样本计权，六个标签全部正确才算正确；
- `frequency_weighted_combination_accuracy`：按组合样本数加权，数值上与 sample exact 相同；
- `macro_combination_accuracy`：先按真实组合分别计算 exact，再对组合等权平均；
- `cardinality_stratified_exact`：按真实组合基数分别统计 exact。

因此，sample exact 与 macro-combination exact 不应再使用同一个“组合准确率”名称混写。

## 4. Checkpoint 选择结果

预注册规则为：

1. `macro_aupr`：early-stop 上 Macro-AUPR 最大；
2. `exact_first_constrained`：在 Macro-AUPR 不低于最大值 `0.002` 的候选中，选择 sample exact 最大者。

5 个 seed 的两种规则全部选择第 8 epoch：

| seed | Macro-AUPR checkpoint | exact-first constrained |
|---:|---:|---:|
| 3050 | 8 | 8 |
| 3051 | 8 | 8 |
| 3052 | 8 | 8 |
| 3053 | 8 | 8 |
| 3054 | 8 | 8 |

因此，在当前 8 epoch 训练预算内，尚未观察到 checkpoint 选择规则分叉。这个结果不能证明两种目标长期一致，只能说明本实验预算下两者都仍在继续改善。

## 5. 阈值验证结果

下表是冻结 checkpoint 后、在测试集一次性确认的 5 seed 均值 ± 样本标准差。Macro-AUPR 对所有阈值方法保持 `0.969181`，因为阈值、温度和 logit bias 不改变样本排序。

| 方法 | sample exact | Macro-F1 | Hamming | macro-combination exact |
|---|---:|---:|---:|---:|
| fixed 0.5 | 0.564500 ± 0.004525 | 0.900617 | 0.094221 | 0.470801 |
| global exact threshold | 0.563850 ± 0.005558 | 0.899594 | 0.094592 | 0.472494 |
| per-label F1 threshold | 0.567925 ± 0.005341 | 0.907837 | 0.092546 | 0.488149 |
| combo exact coordinate threshold | **0.575450 ± 0.005551** | 0.906911 | 0.091167 | **0.511684** |
| temperature scaling | 0.564500 ± 0.004525 | 0.900617 | 0.094221 | 0.470801 |
| logit bias BCE | 0.566600 ± 0.006835 | 0.902768 | 0.093412 | 0.456212 |
| temperature + logit bias | **0.576150 ± 0.004157** | **0.907744** | **0.090650** | 0.490742 |

相对 fixed 0.5：

- `combo exact coordinate threshold` 的 exact 提升 `+1.095` 个百分点，5/5 个 seed 为正；
- `temperature + logit bias` 的 exact 提升 `+1.165` 个百分点，5/5 个 seed 为正；
- 单独温度缩放没有改变 exact，这是预期结果：正温度缩放不改变 logit 的正负号，而 0.5 概率阈值等价于 0 logit 阈值；
- 全局阈值略有退化，说明六类输出的最优决策边界并不共享一个阈值。

threshold-selection 上 `combo exact coordinate threshold` 的 exact 为 `0.604010`，check 上为 `0.586913`，最终 test 为 `0.575450`。存在一定的选择集到最终 test 回落，但方向在 5 个 seed 上一致，说明阈值校准确实恢复了一部分决策边界性能，而不是只改变了排序指标。

## 6. 机制诊断

### 6.1 主要是决策边界问题，但不是全部问题

Macro-AUPR 在 fixed 0.5、组合阈值和温度/偏置方法之间不变，而 exact、Macro-F1 和 Hamming 发生变化，说明当前有一部分误差来自 logit 位置和标签阈值，而不是排序能力完全缺失。

固定 0.5 测试结果中，FMB 的 recall 为 `0.9746`，但 negative-FPR 为 `0.3786`。组合阈值后 FMB FPR 降至 `0.2104`，说明大量组合错误是由某些标签的误报造成的。与此同时，SawFM recall 约 `0.7595`，MFT 约 `0.8334`，ISRJ 约 `0.7915`，尾部/弱标签漏检仍然存在。

### 6.2 组合基数导致 exact 快速下降

组合阈值方法在测试集上的 cardinality-stratified exact 为：

| 真实基数 | exact |
|---:|---:|
| 1 | 0.9338 |
| 2 | 0.7779 |
| 3 | 0.5580 |
| 4 | 0.4234 |
| 5 | 0.3364 |
| 6 | 0.2778 |

随着一个样本中同时激活的标签数量增加，任意一个标签漏检或误报都会使整个组合判错。因此，当前 exact 的主要结构性瓶颈仍然是多标签错误累积和联合决策，而不是单一 AUPR 不足。

### 6.3 sample exact 与 macro-combination exact 的差异是真实的加权差异

fixed 0.5 的 sample exact 为 `0.5645`，但 macro-combination exact 只有 `0.4708`；组合阈值后分别为 `0.5755` 和 `0.5117`。这不是评估器矛盾：前者按样本频率加权，后者让每个组合等权。长尾组合上的错误会在 macro-combination 指标中被放大。

## 7. 结论与下一道门

本次验证支持以下结论：

1. 三个 sample-level exact 实现完全一致，当前没有发现组合 exact 计算 bug；
2. 温度/阈值不改变 AUPR，但能够改变 exact，说明 BCE 的排序目标与固定 0.5 决策存在实际错位；
3. 独立阈值校准在 5 个 seed 上稳定改善 exact，但只能恢复约 1.1 个百分点，不能解释全部组合错误；
4. 当前 8 epoch 下 Macro-AUPR checkpoint 和 exact-first checkpoint 尚未分叉，不能据此宣称二者长期一致；
5. 高基数组合、FMB 误报以及 SawFM/MFT/ISRJ 漏检是剩余主要误差来源；
6. 这份结果仍是新六类数据上的 development evidence，不能把历史 IF6 普通子集结果或本次 development-test 结果升级为 locked-test 结论。

因此，下一阶段应优先固定 threshold-selection 协议、扩大训练预算观察 checkpoint 分叉，并在低 CF 且保持 63 组合的 Pareto 数据上重复该审计。只有在阈值校准和数据覆盖控制后 exact 仍显著偏低，才进入 BCE+组合 margin、基数约束或合法组合 MAP 解码；当前没有足够证据直接宣称普通 BCE 失效。

## 8. 运行收据

- Goal prompt：[goal_prompt.md](E:/LJQ_code/CJ_Imbalance_0716/outputs/goal_ama_if3_bce_exact_20260905/goal_prompt.md)
- 配置：[config.json](E:/LJQ_code/CJ_Imbalance_0716/outputs/goal_ama_if3_bce_exact_20260905/config.json)
- 角色清单：[role_manifest.json](E:/LJQ_code/CJ_Imbalance_0716/outputs/goal_ama_if3_bce_exact_20260905/role_manifest.json)
- 原始汇总：[summary.json](E:/LJQ_code/CJ_Imbalance_0716/outputs/goal_ama_if3_bce_exact_20260905/summary.json)
- 终端收据：[terminal_receipt.json](E:/LJQ_code/CJ_Imbalance_0716/outputs/goal_ama_if3_bce_exact_20260905/terminal_receipt.json)
- 验证脚本：[goal_verify_ama_if3_bce_exact_20260905.py](E:/LJQ_code/CJ_Imbalance_0716/experiments/goal_verify_ama_if3_bce_exact_20260905.py)
