# 两条路线下一阶段实验结果、问题诊断与改进安排

日期：2026-07-22  
环境：`D:\anaconda3\envs\PY10\python.exe`  
数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`  
协议：仅使用 development roles；`locked_test_used=false`

## 1. 执行结论

本轮完成了路线 A 的尾类轮换与 ordering 梯度解耦筛选，以及路线 B 的无标签功率自监督三种子筛选。

1. 路线 A 的 DB Loss 值得保留。8 个“尾类 × seed”配对中，DB 的 macro-AUPR 全部提高，平均提高 `0.03970`；尾类 AUPR 在 7/8 个配对中提高，平均提高 `0.00904`。
2. DB 的固定 0.5 阈值尾类召回在 8/8 个配对中下降，平均下降 `0.19902`；但 Hamming 平均改善 `0.12281`。因此主要问题是类别相关的分数平移和阈值失配，而不是排序能力整体失效。
3. 路线 A 的 ordering 暂停进入主模型。共享梯度虽然提高方向满足率，但 Hamming 超过门槛；StopGrad 和 Delayed-StopGrad 没有保留物理收益，也没有恢复固定阈值分类质量。
4. 路线 B 证明了无标签对比预训练能显著改善冻结表示，但没有证明同源跨功率配对优于普通同视图增强。SameViewAug 的线性探针为 `0.87068 ± 0.00905`，高于 SameSourcePower 的 `0.84720 ± 0.00738`。
5. 20% 标签全量微调后，RandomInit、SameViewAug、RandomSource、SameSourcePower 的 macro-AUPR 均落在 `0.9042–0.9075`，预训练优势基本被监督训练抹平。当前路线 B 只能作为“表征初始化”证据，尚不能作为最终识别增益证据。

## 2. 路线 A1：DB Loss 尾类轮换

协议：四分类 T0、真实 label-LT manifest、IF=3、标签计数容差 20%、optimization/development=1024/300、20 epochs、seeds 2026/2027。

| 尾类 | ASL macro-AUPR | DB macro-AUPR | 差值 | ASL tail-AUPR | DB tail-AUPR | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| FMB | 0.83233 | 0.90391 | +0.07158 | 0.97198 | 0.98250 | +0.01052 |
| SinFM | 0.81631 | 0.85642 | +0.04012 | 0.95149 | 0.95534 | +0.00386 |
| MFT | 0.87543 | 0.91667 | +0.04124 | 0.94095 | 0.95543 | +0.01448 |
| ISRJ | 0.93923 | 0.94511 | +0.00588 | 0.92630 | 0.93359 | +0.00729 |

逐 run 统计：

- macro-AUPR：DB 8/8 胜，配对平均差 `+0.03970 ± 0.02969`；
- tail-AUPR：DB 7/8 胜，配对平均差 `+0.00904 ± 0.01935`；
- tail recall@0.5：DB 0/8 胜，配对平均差 `-0.19902 ± 0.11790`；
- Hamming@0.5：DB 8/8 改善，配对平均差 `-0.12281 ± 0.05978`。

### 2.1 原因判断

DB 对正负样本梯度和类别先验的调整明显改善了排序质量，同时减少了整体二值错误；但不同类别的最优阈值不再自然落在 0.5。tail recall 下降与 Hamming 改善同时出现，说明 DB 更保守地输出尾类正例，而不是简单地学坏了尾类。

SinFM/seed 2026 是唯一 tail-AUPR 下降的配对，下降 `0.03071`；seed 2027 则提高 `0.03842`。这说明尾类排序收益仍有方差，当前两种子只够通过机制筛选，不够形成论文级稳定性结论。

### 2.2 下一步改进

路线 A 主线改为“DB 表征/排序 + 仅由 calibration role 学习逐类阈值”：

1. 保留 DB，不再继续扩大 ASL 参数搜索。
2. 在 calibration role 上分别比较固定 0.5、逐类 F1 阈值、逐类 Youden/成本敏感阈值；阈值不得读取 early-stop 或 locked test。
3. 扩展到 seeds 2026–2030，并报告配对 bootstrap 置信区间、每类 AUPR、每类 recall、Hamming 和 cardinality error。
4. 在 IF=2、3、5 上验证失衡强度趋势，并保留四个尾类轮换，证明收益不是单一 manifest 特例。
5. 只有校准后 tail recall 不低于 ASL、macro-AUPR 和 Hamming 仍保持优势，DB 才进入六分类与联合长尾实验。

## 3. 路线 A2：ordering 梯度解耦

| 变体 | macro-AUPR | F1@0.5 | Hamming@0.5 | weak recall | satisfaction | 梯度余弦 |
|---|---:|---:|---:|---:|---:|---:|
| T0P | 0.97917 | 0.89097 | 0.12917 | 0.31 | 0.63550 | — |
| Shared | 0.97681 | 0.88031 | 0.14250 | 0.32 | **0.72662** | +0.26620 |
| StopGrad | 0.97897 | 0.86623 | 0.16417 | 0.30 | 0.63497 | +0.28910 |
| Delayed-StopGrad | 0.97906 | 0.87048 | 0.15833 | 0.29 | 0.63286 | +0.11186 |

Shared 的 AUPR 只下降 `0.00236`，但 Hamming 恶化 `0.01333`，超过冻结门槛；另外两个变体的 Hamming 分别恶化 `0.03500` 和 `0.02917`，且物理方向指标没有提高，均不通过。

### 3.1 原因判断

- 分类梯度与 ordering 梯度的平均余弦为正，不支持“简单的反向梯度冲突”解释。
- StopGrad 阻断特征提取器后，ordering 只能重塑分类头；物理收益消失但固定阈值仍退化，说明问题至少部分位于 logit 标度/偏置和分类校准层面。
- 强制跨功率单调并不一定对所有样本成立。功率变化可能改变弱分量可辨识度和复合成分竞争关系，把统一 margin 直接施加到类别 logit 上会过度约束分类空间。

### 3.2 下一步安排

ordering 不再作为主分类损失。若继续研究，只保留一个独立机制实验：从 stop-gradient 特征上训练独立 order head，order head 不回写分类器；同时加入功率差条件和可学习 margin。该分支仅用于证明表征是否包含功率方向信息，不作为提升分类指标的核心创新点。未得到新的机制证据前，不再与 DB 或 T2D 联合。

## 4. 路线 B：无标签功率自监督

协议：预训练仅使用 `semi_plan["unlabeled_indices"]` 中的 1024 个源身份、25/30/35 dB 成对视图、12 epochs；冻结骨干后用相同 20% labeled subset 训练 30 epochs 线性头；随后用同一 labeled subset 全量微调 20 epochs；seeds 2026/2027/2028。

| 变体 | linear-probe macro-AUPR | finetune macro-AUPR | finetune Hamming@0.5 |
|---|---:|---:|---:|
| RandomInit | 0.61548 ± 0.02294 | 0.90504 ± 0.00849 | 0.44972 |
| SameViewAug | **0.87068 ± 0.00905** | 0.90734 ± 0.01488 | 0.46139 |
| RandomSource | 0.80247 ± 0.02870 | 0.90421 ± 0.00966 | 0.45889 |
| SameSourcePower | 0.84720 ± 0.00738 | **0.90749 ± 0.01047** | **0.44806** |

SameSourcePower 的逐种子配对差：

- 相对 RandomInit，线性探针平均 `+0.23172 ± 0.01889`，3/3 为正；
- 相对 RandomSource，线性探针平均 `+0.04473 ± 0.02362`，3/3 为正；
- 相对 SameViewAug，线性探针平均 `-0.02348 ± 0.00598`，3/3 为负；
- 相对 RandomInit，全量微调平均仅 `+0.00245 ± 0.00286`。

### 4.1 原因判断

本轮通过了“优于随机初始化、优于错误跨源配对”的原冻结门槛，但没有通过更关键的“优于普通自监督增强”门槛。原因包括：

1. 同源跨功率正对同时改变了接收信噪条件，直接要求完全不变可能抹去弱分量的可辨识线索；普通轻增强更接近真正的语义保持变换。
2. 当前每个源身份可形成多个功率记录；同一身份的另一条记录可能在 batch 中被当作负样本，形成 false negative，削弱同源目标。
3. RandomSource 仍显著高于 RandomInit，说明当前对比任务的大部分收益可能来自低层滤波器预训练和通用正则化，而不是正确身份配对本身。
4. 20% 标签、20 轮微调对当前四分类开发集已经足以覆盖初始化差异，因而线性可分性优势没有转化为明显的最终指标优势。
5. 所有固定阈值 F1/Hamming 结果仍受校准影响；当前 early-stop role 同时承担 checkpoint 选择和开发报告，只能作为筛选证据，不是最终独立评估。

### 4.2 下一步改进

路线 B 不再使用“纯跨功率不变性”作为唯一目标，改为以下分解结构：

- 主目标：SameViewAug 的实例对比学习；
- 跨功率目标：同身份多正样本、identity-aware false-negative masking，并降低跨功率项权重；
- 保真目标：增加相对功率差/功率等级预测辅助头，避免把所有功率信息从表征中抹掉；
- 下游验证：分别使用 5%、10%、20% 标签，比较 frozen linear probe、全量微调和 held-out power 泛化；
- 统计：seeds 2026–2030，报告配对差及置信区间；
- 对照：RandomInit、SameViewAug、SameViewAug+naive power pair、SameViewAug+multi-positive/masked power pair、错误身份配对。

只有改进后的跨功率分支在至少两个标签预算上显著优于 SameViewAug，并在 held-out power 上保持优势，才可以把“功率结构自监督”作为 SCI 主创新点。否则 SameViewAug 应作为路线 B 的实际预训练基线，跨功率配对只保留为负结果和机制分析。

## 5. 下一轮优先级

1. **最高优先级：路线 A 的 DB + 独立 calibration。** 当前证据最一致，预计能直接解决 tail recall@0.5 的主要矛盾。
2. **第二优先级：路线 B 的 multi-positive/false-negative-masked 功率目标。** 先在 5% 标签和 held-out power 条件做小规模筛选，验证是否真正提供普通增强没有的收益。
3. **暂停：ordering 并入分类主干。** 只允许独立 order head 做机制诊断。
4. 以上开发结论稳定后，再扩大到六分类、五种子完整训练；locked test 继续冻结，直到最终模型与阈值全部确定。

## 6. 可复现实验产物

- 路线 A 汇总：`outputs/tapc/route_a_next_stage/route_a_next_stage_summary.json`
- 路线 B 三种子汇总：`outputs/student_b/route_b_power_ssl_multiseed_summary.json`
- 路线 B 单种子目录：`outputs/student_b/route_b_power_ssl_seed_2026`、`seed_2027`、`seed_2028`
- 路线 A runner：`experiments/run_route_a_next_stage.py`
- 路线 B runner：`experiments/run_route_b_power_ssl.py`
- 路线 B 聚合器：`experiments/aggregate_route_b_power_ssl.py`

所有 run receipt 均保存 seed、训练规模、split/semi-split/pair hash、配置 hash、checkpoint、history 和 `locked_test_used=false`。
