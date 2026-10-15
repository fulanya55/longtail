# Route A：ISRJ 联合周期特征专家结果（development-v3）

## 1. 结论

310维联合周期特征与固定小叶节点HGB通过全部预注册局部晋级门槛，首次为ISRJ建立了严格组合匹配下的可用独立专家证据：early AUROC `0.62269`，相对冻结final-temporal提高 `+0.03008`，相对25点块置换HGB提高 `+0.07359`，相对aligned logistic提高 `+0.04850`。

4/5种子优于final-temporal、5/5优于块置换，validation→early gap为 `0.10386`。这同时证明：ISRJ弱分量中存在100–1200点的长延迟周期信息；该信息需要复自相关、包络自相关、频谱和周期峰的联合非线性交互，不能由普通线性头或局部功率STFT充分利用。

## 2. 五种子结果

| 指标 | 均值 | 95% bootstrap CI |
|---|---:|---:|
| aligned HGB early AUROC | **0.62269** | [0.60603,0.64338] |
| aligned HGB early AUPR | 0.25216 | [0.23755,0.26651] |
| aligned logistic AUROC | 0.57420 | [0.55798,0.58999] |
| block-shuffled HGB AUROC | 0.54910 | [0.54296,0.55568] |
| HGB − final-temporal | **+0.03008** | [0.00283,0.05753] |
| HGB − block-shuffled | **+0.07359** | [0.05956,0.08855] |
| HGB − aligned logistic | **+0.04850** | [0.01992,0.08334] |
| validation − early gap | 0.10386 | [0.03900,0.16619] |

## 3. 逐种子

| seed | final temporal | aligned HGB | logistic | block-shuffled HGB | HGB−final | HGB−shuffle | gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2039 | 0.5904 | 0.6150 | 0.5541 | 0.5482 | +0.0245 | +0.0667 | 0.0622 |
| 2040 | 0.6216 | 0.6011 | 0.5863 | 0.5391 | -0.0205 | +0.0620 | 0.2067 |
| 2041 | 0.5578 | 0.6285 | 0.5922 | 0.5446 | +0.0707 | +0.0839 | 0.1510 |
| 2042 | 0.6104 | 0.6624 | 0.5488 | 0.5606 | +0.0520 | +0.1018 | -0.0061 |
| 2043 | 0.5828 | 0.6065 | 0.5896 | 0.5530 | +0.0237 | +0.0535 | 0.1054 |

seed2040单独低于final-temporal且gap较大，但五种子均值、正向种子数和总体gap均通过预注册门槛。该种子风险必须在统一接回中继续单独报告。

## 4. 机制证据

1. **长周期时间顺序必要。** 25点块置换保留局部I/Q波形和幅度分布，但破坏50点以上间隔；aligned HGB在5/5种子上均明显优于它，排除了仅靠局部幅度纹理或模型容量解释。
2. **非线性交互必要。** 相同aligned 310维特征上，HGB比Logistic高 `+0.04850`，说明单个周期峰或线性加权不足，需要组合复相关、包络相关和频谱条件。
3. **普通功率STFT失败不代表原始I/Q无信息。** 上一轮STFT-CNN接近随机，而联合周期HGB达到0.62269；问题是归纳偏置与特征统计尺度，而不是单纯网络深度。
4. **证据仍是局部组件级。** 本轮只评估ISRJ弱正与组合匹配负样本排序，尚未证明接回六标签系统后能提高阈值后弱召回、保持强正和FPR。

## 5. 下一步

冻结HGB、310维特征、lag区间和全部超参数。下一轮只验证统一接回：将fit严格匹配分数标准化后，通过C4不确定性门控和0.5硬上限只修改ISRJ logit；与C4和block-shuffled HGB残差比较，重新执行独立calibration、threshold和early评估。

不得在接回失败后增加叶节点、搜索lag或使用sample_width真值。

## 6. 产物

- 预注册：`docs/plans/route_a_isrj_joint_periodicity_expert_v3_plan_20260801.md`
- 代码：`experiments/run_route_a_isrj_joint_periodicity_expert_v3.py`
- 汇总：`outputs/tapc/route_a_isrj_joint_periodicity_expert_v3/route_a_isrj_joint_periodicity_expert_v3_summary.json`
- 汇总 SHA-256：`BFD4BFF5555C24F7C0C25B2049C5363BD49E3461E3134BDD543CE4019ECDFC6F`
- 每种子保存模型、归一化、预测、角色和模型SHA、匹配计数及置换SHA
- `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`

