# 路线 A：功率域决策校准与 S2 稳定性结果（2026-07-25）

## 总结

本轮两条验证得到相反但互补的结论：

- **路线 S 通过全部门槛**：种子 2043 的三个新初始化把 S2 四档平均 AUROC 从 0.6080 提高到 0.9082～0.9249，说明此前的 2043 失败主要是优化/初始化偶然性，不是 stage2 表征上限。
- **路线 D 未通过**：冻结旧 C2 后，一维双域 Platt 和单一统一阈值无法同时保持普通域 FPR并恢复 40/45/50 dB 召回。问题不是简单阈值数值错误，而是跨可见度分数分布发生系统平移，且旧 2043 专家进一步放大了这一问题。

因此当前不能解冻最后 stage。下一步应先选定稳定 S2 checkpoint，重新建立 C2 分数尺度，再验证不读取真实功率标签的“可见度条件校准”。

## 1. 路线 D：冻结 C2 的功率稳健决策

### 协议

- 每种子从 train role 排除原 S2 专家训练身份与上一轮路由训练身份。
- 新选择 28 组合 × 20 对正样本/hard negatives，按组合各半拆成 power-calibration 和 power-threshold。
- 两角色身份互斥；40/45/50 dB 视图以 identity 绑定。
- early-development 普通域及三档功率仅作最终评价。
- C2、F2、H2、S2 和 alpha=.80 全冻结，只拟合 2 参数 Platt 与一个统一阈值。

### 聚合结果

| 方案 | 普通 Saw recall | 普通 Saw FPR | 三档平均 recall | 最差档 recall | 最大功率 FPR |
|---|---:|---:|---:|---:|---:|
| D0 普通域阈值 | 0.5813 | 0.1783 | 0.0327 | 0.0000 | 0.0000 |
| D1 双域 Platt + 0.5 | 0.8704 | 0.5287 | 0.2791 | 0.0071 | 0.1543 |
| D2 双域 Platt + 稳健阈值 | 0.5751 | 0.1709 | 0.0293 | 0.0000 | 0.0000 |

D1 证明分数可以整体抬高，但普通域 FPR 随即升到 0.529；D2 为满足普通域及各功率 FPR≤0.20，只能把阈值提高到 0.62～0.74，最终回到几乎不报弱正的状态。

### D2 逐种子

| 种子 | 阈值 | 普通 recall/FPR | 40 dB recall | 45 dB recall | 50 dB recall |
|---:|---:|---:|---:|---:|---:|
| 2041 | 0.67 | 0.691/0.159 | 0.204 | 0.000 | 0.000 |
| 2042 | 0.74 | 0.506/0.162 | 0.060 | 0.000 | 0.000 |
| 2043 | 0.62 | 0.528/0.192 | 0.000 | 0.000 | 0.000 |

三种子均未通过单种子门槛，路线 D 总门槛失败。

### Oracle 分档阈值审计

若允许在 early-development 上为每个功率档单独选择 oracle 阈值，并保持 FPR≤0.20：

- 2041：40/45/50 dB recall 为 0.837/0.714/0.673；
- 2042：0.980/0.850/0.630；
- 2043：0.128/0.096/0.074。

这说明：

1. 2041/2042 的表征排序足以支持高召回，但不同功率需要明显不同阈值，例如 2041 的 40/45/50 dB 最优阈值约 0.51/0.37/0.26。
2. 一个统一标量阈值无法吸收这种随可见度变化的分数平移。
3. 旧 2043 不只是校准失败，其分档 oracle 召回也很低，属于专家 checkpoint 失稳。

## 2. 路线 S：种子 2043 的三个初始化

### 固定协议

- 数据 seed、split、1120 对训练身份、896/224 fit-validation、hard negatives、10 dB teacher、40/50 dB student、45 dB checkpoint 选择全部固定。
- 网络与损失不变：3651 参数、rank=8、AdamW `3e-4`、最多 30 epoch。
- 只改变初始化：274301、274302、274303。
- 35/40/45/50 dB early-development 独立身份评价；locked test 未使用。

### 结果

| 初始化 | 四档平均 AUROC | 最差档 | 跨档 std | 六分量均值 | 相对原 S2 |
|---:|---:|---:|---:|---:|---:|
| 274301 | 0.9099 | 0.8472 | 0.0492 | 0.91 | +0.3019 |
| 274302 | 0.9082 | 0.8646 | 0.0323 | 0.62 | +0.3002 |
| 274303 | **0.9249** | 0.8585 | 0.0496 | 0.74 | **+0.3169** |

原 2043 S2 的四档 AUROC 为：

- 35 dB：0.6838；
- 40 dB：0.6257；
- 45 dB：0.5784；
- 50 dB：0.5440；
- 平均：0.6080。

三个新初始化的平均 AUROC 标准差仅 0.0092，且所有初始化都通过平均 AUROC、最差功率、六分量和相对增益门槛。路线 S 全部通过。

### 机制判断

原 2043 与新初始化使用完全相同的数据和训练规则，best epoch 都为 30；唯一改变是参数初始化。因此 2043 的低性能不能再解释为“数据 seed 2043 天生困难”或“stage2 没有足够特征”，更直接的原因是非凸优化落入了低质量解。

## 3. 双路线最终决策

按照预注册决策表：

- D 失败；
- S 通过。

对应决策为：**保持主干冻结，先替换/稳定专家，再做可见度条件决策校准；暂不执行最后 stage 解冻。**

## 4. 下一步技术方案

### 4.1 稳定专家选择

不能根据 early-development 直接选择 274303。下一轮应使用原 45 dB validation role 冻结选择规则：

1. 在三个新 checkpoint 中按 45 dB validation AUROC、AUPR和 40/50 dB validation 一致性选择；
2. 选择完成后冻结 checkpoint hash；
3. early-development 只用于确认，不再参与选择。

也可用三个专家 logit 的均值作为 3-checkpoint ensemble 对照，验证是否比单 checkpoint 更稳定；推理成本需单独报告。

### 4.2 可见度条件校准

不输入真实 JNR 或功率档，使用可部署的分数统计估计可见度：

- C2 SawFM logit；
- S2 evidence map 的 top-k 均值、最大值、标准差；
- H2 与 S2 分数差；
- H2 扫频三标签最大 logit；
- 同一样本无注意力 stage2 mean/std。

只训练一个低容量单调校准器，输出样本条件偏置或温度，但保持 SawFM 排序主体冻结。对照：统一 Platt、三段分数分位校准、单调 spline/isotonic、低容量 visibility-conditioned affine。

### 4.3 下一轮门槛

- 40/45/50 dB 平均 recall≥0.50、每档≥0.35；
- 每档 hard-negative FPR≤0.20；
- 普通域 FPR 增量≤0.02；
- 普通域 recall 降幅≤0.05；
- 3/3 数据种子中至少 2 个通过；
- 与分档 oracle recall 的差距不超过 0.15；
- checkpoint 选择不使用 early-development 或 locked test。

只有这些门槛通过，才解冻最后 stage。

## 5. 产物

- 方案：`docs/plans/route_a_power_decision_and_s2_stability_plan_20260725.md`
- D 程序：`experiments/run_route_a_power_robust_decision.py`
- D 汇总：`outputs/tapc/route_a_power_robust_decision/route_a_power_robust_decision_summary.json`
- S 程序：`experiments/run_route_a_sawfm_s2_initialization_stability.py`
- S 汇总：`outputs/tapc/route_a_sawfm_s2_initialization_stability/route_a_sawfm_s2_initialization_stability_summary.json`
- 三个新 S2 checkpoint、history、四档预测、功率决策预测、阈值搜索、身份审计及 receipts 均已保存。
