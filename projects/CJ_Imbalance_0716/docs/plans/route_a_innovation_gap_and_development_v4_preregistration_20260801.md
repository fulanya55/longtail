# Route A 创新证据缺口矩阵与 development-v4 预注册计划

日期：2026-08-01

## 1. 本轮结论

当前 `GradientContrastFixed1p5` 已在一次性 test-v2 上达到
`C_independent_test`，但该证据只证明方法在当前 0716 数据协议内有效。
由于候选由

`theta_C6 + 1.5 * (theta_M5 - theta_C6)`

构成，其数学形式同时属于参数空间任务方向和模型融合邻域。现有结果尚未排除
普通插值、随机等范数扰动、参数位置重排、其他标签屏蔽方向以及非配对任务方向
等替代解释。因此，本轮只回答一个问题：

> 当前增益是否具有 SawFM 特异性，并且依赖 C6/M5 的严格配对？

本轮证据角色为 `development`。历史 locked-test、已经消耗的 test-v2 和密封的
test-v3 均不得访问、运行、校准、选择或用于修复结果。

## 2. 最近三轮正式证据

| 轮次 | 证据 | 结果 | 决策 |
|---|---|---|---|
| development-v2 | 反复开发 | fixed alpha 方向 macro-F1 `+0.01233`，弱分量 Recall `+0.02597` | 未过原弱 Recall 门槛 |
| development-v3 | 新身份 development | fixed alpha 方向 macro-F1 `+0.01337`，弱分量 Recall `+0.03283`；LSCP 仅 `+0.00360` | fixed alpha 通过，LSCP 停止 |
| test-v2 once | 独立一次性测试 | macro-AUPR `+0.00648`，macro-F1 `+0.01390`，弱分量 Recall `+0.03048`，5/5 种子为正 | C 级；test-v2 已消耗 |

## 3. 冻结边界审计

- test-v2 receipt：`test_v2_used=true`；
- test-v2 receipt：`fit_or_selection_on_test_v2=false`；
- test-v2 receipt：`rerun_or_retuning_allowed=false`；
- 历史 locked-test：禁止访问；
- test-v3：保持密封，不是 development-v4 的测试目标；
- development-v4 只生成 train split，不生成新的 test split；
- 全部 SawFM-containing 样本继续保留，只改变监督标签或参数方向；
- smoke 只验证实现，不计算晋级结论。

## 4. 文献新颖性审计

### 4.1 最接近的方法族

| 方法族 | 原始工作 | 与本方法的关系 | development-v4 必须排除的解释 |
|---|---|---|---|
| 梯度投影 | [PCGrad](https://arxiv.org/abs/2001.06782) | 对冲突任务梯度做局部投影 | 当前增益是否只是通用梯度手术可获得的效果 |
| 冲突规避优化 | [CAGrad](https://arxiv.org/abs/2110.14048) | 在平均目标与最差局部任务改进间求折中 | SawFM 条件方向是否优于通用多任务优化 |
| 多目标优化 | [MGDA](https://arxiv.org/abs/1810.04650) | 将多任务训练作为 Pareto 优化 | 当前方向是否只是不同损失权重的一个近似 |
| 梯度平衡 | [GradNorm](https://arxiv.org/abs/1711.02257) | 动态调节各任务梯度尺度 | 增益是否仅来自 SawFM 权重缩小 |
| 博弈式多任务优化 | [Nash-MTL](https://arxiv.org/abs/2202.01017) | 通过协商得到联合更新方向 | 标签特异方向是否比一般联合方向更必要 |
| 低开销动态加权 | [FAMO](https://arxiv.org/abs/2306.03792) | 以低开销平衡各任务损失下降 | 当前方法的收益/成本是否有优势 |
| 任务向量 | [Task Arithmetic](https://arxiv.org/abs/2212.04089) | 用同一预训练起点与微调模型之差定义参数方向 | 当前公式是否只是已有任务向量外推 |
| 模型平均 | [Model Soups](https://proceedings.mlr.press/v162/wortsman22a.html) | 在权重空间平均多个模型 | 普通 midpoint 是否已能解释增益 |
| 冲突感知模型融合 | [TIES-Merging](https://arxiv.org/abs/2306.01708) | 处理任务向量的稀疏与符号冲突 | 参数符号/支持冲突是否比 SawFM 语义更关键 |
| 自适应融合 | [AdaMerging](https://arxiv.org/abs/2310.02575) | 以任务或层为单位学习融合系数 | 固定 alpha 是否仅是未优化的融合系数 |
| 近期任务向量几何 | [Task-Vector Geometry Analysis](https://arxiv.org/abs/2607.16062) | 用随机初始化下界、同运行上界和方向几何解释融合 | 必须给方向相似度提供随机/配对参照 |

### 4.2 当前允许和禁止的创新表述

允许：

> 当前数据协议下得到独立验证的 SawFM 条件配对参数外推方法。

尚不允许：

> 通用梯度冲突解决方法；或一种已经证明优于任务向量、模型融合及通用多任务优化的新方法。

只有在 SawFM 特异性、配对依赖性、通用基线、跨分布和外部测试全部完成后，
才能升级方法学创新声明。

## 5. 创新证据缺口矩阵

| 科学问题 | 当前证据 | 缺口 | 本轮处理 |
|---|---|---|---|
| 效果是否真实 | development-v3 和 test-v2 均为正 | 新身份再次复现 | 是 |
| 是否 SawFM 特异 | 只有 C6/M5 方向 | 缺少其他标签屏蔽方向 | 是 |
| 是否依赖配对 | C6/M5 共享初始化与数据顺序 | 缺少非配对方向 | 是 |
| 是否只是普通融合 | alpha=1.5 优于端点，但未正式加入 midpoint | 缺少 alpha=0.5 模型平均 | 是 |
| 是否只是参数噪声 | 无 | 缺少等层范数随机方向 | 是 |
| 是否依赖参数位置 | 无 | 缺少层/张量位置置乱方向 | 是 |
| 是否优于通用梯度方法 | LSCP 失败不是完整基线族 | 缺 PCGrad/CAGrad/MGDA/GradNorm | 下一阶段 |
| 是否跨分布/骨干 | 仅当前 F2 和当前数据生成协议 | 缺分布与架构复现 | 后续阶段 |
| 是否外部泛化 | test-v2 仍属于同一生成体系 | 缺新生成体系或实测数据 | 最终阶段 |

## 6. development-v4 数据协议

- 数据角色：`development_v4`；
- 只生成 `closedset_6class_train.mat`；
- base sample seed：`34000000`；
- 样本数：25,200；
- development-v4 sample seed 区间：`[34000001, 34025200]`；
- 声明的预留 test-v4 区间：`[37000001, 37025200]`，本轮不生成；
- 与 v2 train、test-v2、v3 train、密封 test-v3 的声明 seed 区间均不重叠；
- 在运行模型前记录文件 SHA256、receiver_noise_seed 范围、唯一数和交集审计；
- 禁止通过读取任何已消耗/密封测试文件完成身份审计，只能使用冻结声明和开发文件本身。

## 7. 本轮候选和对照

### 7.1 训练端点

所有端点共享相同样本、split、label-longtail manifest、batch 顺序、训练预算、
early-stop、校准和阈值规则。

1. `C6_Joint`；
2. `M5_GradientIsolated`，屏蔽 SawFM；
3. `MaskFMB`；
4. `MaskSinFM`；
5. `MaskMFT`；
6. `MaskVFT`；
7. `MaskISRJ`。

屏蔽只作用于标签损失，所有 SawFM 共现样本均保留。

### 7.2 参数空间变体

1. `GradientContrastFixed1p5`：`C6 + 1.5*(M5-C6)`；
2. `ModelAverage0p5`：`C6 + 0.5*(M5-C6)`；
3. `RandomNormMatched1p5`：在与 SawFM 方向相同的张量支持上生成随机方向，逐张量匹配范数，再乘 1.5；
4. `LayerShuffled1p5`：同形张量之间循环置换；单例张量内部确定性置乱；
5. `MaskFMBContrast1p5`；
6. `MaskSinFMContrast1p5`；
7. `MaskMFTContrast1p5`；
8. `MaskVFTContrast1p5`；
9. `MaskISRJContrast1p5`；
10. `NonPairedSeedContrast1p5`：把相邻种子的 M5-C6 方向应用到当前种子的 C6。

非配对方向明确属于反事实对照，不能被解释为一个可部署候选。

## 8. 评价指标与统计

主要指标：

- common-five macro-AUPR；
- common-five macro-F1；
- MFT/VFT/ISRJ mean weak recall；
- target mean negative FPR；
- SawFM-present macro-F1；
- SawFM-absent macro-F1；
- 每个 common label 的 AUPR。

统计：

- seeds：2039--2043；
- 报告逐种子配对差值；
- 以固定随机种子的 seed bootstrap 给出 95% CI；
- 报告方向全局范数、逐张量范数误差、候选与控制方向余弦；
- smoke 不运行任何晋级门槛。

## 9. 预注册门槛

### 9.1 效果复现门槛

`GradientContrastFixed1p5` 相对 `C6_Joint` 必须同时满足：

1. macro-AUPR delta `>= -0.005`；
2. macro-F1 delta `>= +0.010`；
3. target weak recall delta `>= +0.030`；
4. candidate target FPR `<= 0.20`；
5. 至少 4/5 种子的 target weak recall delta 为正；
6. SawFM-present macro-F1 delta `> 0`；
7. SawFM-absent macro-F1 delta `>= -0.005`；
8. 三个目标标签 AUPR delta 均 `>= -0.010`。

### 9.2 SawFM 特异性门槛

在效果复现通过的前提下：

1. 候选 target weak recall delta 必须高于 `ModelAverage0p5`、
   `RandomNormMatched1p5`、`LayerShuffled1p5` 和
   `NonPairedSeedContrast1p5`；
2. 候选必须高于五个 other-label contrast 中的最强者；
3. 上述两组最小优势至少有一项达到 `+0.005` weak recall，另一项不得为负；
4. 候选在 SawFM-present 子集的 macro-F1 delta 必须高于控制组中位数；
5. 所有输出有限，身份审计通过，locked-test/test-v2/test-v3 均未使用。

该门槛用于排除明显替代解释，不用于声称已经优于 PCGrad/CAGrad 等通用方法。

## 10. 决策规则

- `promote`：效果复现和 SawFM 特异性门槛均通过，进入通用多任务基线阶段；
- `revise`：效果复现通过但特异性失败，保留“任务方向有效”结论，停止 SawFM 特异机制强声明，并设计方向归一化/层级机制实验；
- `stop`：效果复现失败，冻结负结果，不访问任何测试集，重新提出可证伪机制。

## 11. 证据等级

本轮无论结果如何均为 `development`。即使全部门槛通过，也不得访问 test-v2、
test-v3 或历史 locked-test，不得将本轮称为 external evidence。

## 12. 必须保存的产物

- 数据生成脚本与 receipt；
- development-v4 配置；
- identity audit 与 SHA256；
- runner 与自动测试；
- 每种子 split、manifest、role indices；
- 所有端点和派生模型 checkpoint；
- calibration、threshold、预测指标和方向诊断；
- 汇总 JSON、run receipt 和正式 15 节结果报告。

