# FC-PTV 新机制文献差异矩阵与 fresh-development 预注册计划

日期：2026-08-01  
状态：**预注册冻结稿；尚未生成 development-v12 数据，尚未实现或运行正式 FC-PTV**  
内部代号：Functionally Constrained Paired Task Vector（FC-PTV）

## 1. 本轮结论

本轮只完成方法身份审计、数学定义和 fresh-development 预注册，不形成性能结论。

现有 `FullAlpha1p5` 已证明在当前协议内具有稳定的 macro-AUPR/macro-F1 正增益，并在
development-v6 中优于最强基线 M5；但是 target weak recall 的跨身份效应量为
`+0.03283、+0.01691、+0.02813、+0.02454`，没有稳定超过既定下限。因此，继续搜索
普通 alpha 不再具有可接受的科学解释。

文献审计同时表明：

1. “task vector + trust region” 已与 TATR 高度相邻；
2. “以预测分布或功能距离代替欧氏参数距离”已与 Fisher–Rao/Karcher merging、RegMean
   等工作相邻；
3. “逐层或分块学习融合系数”已与 AdaMerging 相邻；
4. 因而 FC-PTV 不能把上述宽泛概念本身作为新颖性主张。

FC-PTV 仍可检验的窄创新身份是：

> 对同初始化、同数据、同 batch 顺序下，仅删除 SawFM 标签损失而产生的严格配对
> `C6 -> M5` 任务向量，使用 **C6 的 fit-only common-five logit 稳健尺度**定义逐标签、
> 逐 SawFM 共现层的功能半径，再将 `shared` 与 `classifier` 两个非零参数块联合投影到该
> 固定 trust region；整个投影不读取 validation、early、calibration、threshold 或任何
> test 角色，也不以正式评价指标选择系数。

这是一个待证伪的项目级机制，不是已确认的文献新颖性结论。在外部检索和实验门槛全部通过前，
“FC-PTV”仅作为内部代号。

## 2. 最近三轮冻结结果

| 轮次 | 数据角色 | macro-AUPR Δ | macro-F1 Δ | target weak recall Δ | 关键结论 |
|---|---|---:|---:|---:|---|
| development-v4 | fresh train-only development | +0.00646 | +0.01163 | +0.01691 | 5/5 正；ISRJ `-0.00116`；效应量门槛失败 |
| development-v5 | fresh train-only development | +0.00632 | +0.01464 | +0.02813 | Full1.5 最优；shared-only 有害，classifier-only 贡献显著 |
| development-v6 | fresh train-only development | +0.00586 | +0.01266 | +0.02454 | 优于 M5 `+0.00853`，CI `[+0.00384,+0.01408]`；自身门槛差 `0.00046` |

冻结解释：当前成功更符合“严格配对的 SawFM label-deletion shared+classifier task-vector
协同”，不支持“纯共享梯度冲突抑制”。LSCP 保持正式负结果。

## 3. 本轮唯一科学问题

仅使用 fit 角色测得的功能位移约束，能否在不牺牲 macro-AUPR、macro-F1 和 FPR 的前提下，
降低 `FullAlpha1p5` 的跨种子弱分量效应波动，并使 MFT、VFT、ISRJ 三个目标标签同时获得
稳定正增益？

## 4. 可证伪假设

固定 `alpha=1.5` 只控制欧氏参数步长，无法控制相同参数位移在不同数据身份上的功能幅度。
如果弱分量不稳定主要来自某些标签/共现层上的功能位移过大，那么把 Full1.5 最小幅度地投影到
由 C6 fit-only logits 定义的功能 trust region，应当：

1. 保留 classifier 方向的主要收益；
2. 对功能敏感度更高的 shared 位移施加更强约束；
3. 提高 Full1.5 的最差目标标签；
4. 降低 `FC-PTV - C6` 的跨种子 target weak recall 标准差；
5. 相比等参数范数的全局缩放，产生额外收益。

反证条件包括：FC-PTV 与 Full1.5 等价、退化为 classifier-only、不能满足功能半径、不能超过
等参数范数控制，或不能通过第 13--15 节任一强制门槛。

## 5. 原始文献检索范围

检索日期为 2026-08-01。只把论文原始页面、正式 proceedings、OpenReview 或 arXiv 原始条目
作为方法定义来源；博客和二手综述不作为差异矩阵依据。检索主题包括 task arithmetic、model
soup、TIES/AdaMerging、Fisher/RegMean、function-space/Fisher–Rao merging、trust-region、
PCGrad/CAGrad/MGDA/GradNorm、negative transfer、missing/partial labels 和 multi-label
interference。

本轮检索没有发现与“保留全部 SawFM 共现样本、只删除 SawFM loss，再用该严格配对差分改善
其他标签弱分量”完全相同的原始工作；该检索结果不能证明不存在先例，只能说明当前尚未找到完全
同构方法。

## 6. 文献差异矩阵

| 方法 | 任务向量/模型来源 | 约束或归一化空间 | 数据需求及是否需 test | 是否逐层/逐块 | 是否显式保持目标标签 | 训练与推理成本 | 与 FC-PTV 的关键差异 |
|---|---|---|---|---|---|---|---|
| Model Soups | 同一预训练模型经不同超参数微调的多个模型 | 欧氏参数平均；greedy soup 由验证性能逐个纳入 | uniform 不需额外数据；greedy 需要 validation，不需 test | 否 | 否；面向同任务总体准确率 | 需多个微调模型；单模型推理 | 不构造 label-deletion 配对向量，不定义功能半径 |
| Task Arithmetic | `theta_task - theta_pre` | 欧氏参数加减与全局缩放 | 合并本身不需 test | 原始方法否 | 否 | 需任务端点；单模型推理 | FC-PTV 只使用 C6/M5 严格配对差分，并在 fit-only 功能空间投影 |
| TIES-Merging | 多个 task vectors | 参数坐标中 Trim、Elect Sign、Disjoint Merge | 合并本身可无数据、无 test | 元素级，但不是输出功能约束 | 否 | 无再训练；单模型推理 | 解决多向量符号/冗余干扰，不控制 C6 输出位移 |
| AdaMerging | 多个 task vectors | 以熵最小化学习 task-wise 或 layer-wise 系数 | 原论文使用无标签 test samples 调系数 | 是 | 否 | 需系数优化；单模型推理 | FC-PTV 禁止 test-time 系数学习，半径和投影只由 fit/C6 确定 |
| Fisher Merging | 同初始化的多个模型 | Fisher 精度加权的参数后验近似 | 需数据估计 Fisher；不必为 test | 参数元素级 | 否 | Fisher 统计成本；单模型推理 | 是参数重要性加权，不是逐标签 logit trust region |
| RegMean / Dataless Knowledge Fusion | 多个领域模型 | 各线性层的回归/输出差异匹配 | 依赖层输入统计或相应代理；不以 test 指标选系数 | 是 | 否 | 闭式逐层融合；单模型推理 | 目标是匹配多个专家，不以 C6 为唯一锚点，也不针对 label-deletion 弱分量 |
| TATR (2025 preprint) | 多任务 task vectors | 以 `|grad L_j| * |Delta_i|` 参数敏感度建立元素级 trust region 并掩膜 | 使用每任务 exemplar 估计梯度；不必用 test | 元素级；报告层敏感度 | 否 | 梯度/掩膜计算；单模型推理 | 与 FC-PTV 最接近；TATR 约束参数敏感维度，FC-PTV 约束 C6 common-five 输出位移，并只处理一个严格配对删除向量的 shared/classifier 联合投影 |
| Fisher–Rao Karcher Merging (2026 preprint) | N 个异质 LLM experts | Fisher–Rao/KL 功能距离的 Karcher mean；实践用球面代理 | 轻量代理可不依赖 test；可选 Fisher 预条件需统计 | 可选 blockwise | 否 | 固定点融合；单模型推理 | 面向 N 专家重心和表示坍塌，不是相对 C6 的有界单向量投影，也无标签级保护 |
| TRPO / 经典 trust-region 优化 | 连续策略迭代更新 | 由旧策略分布上的 KL 半径约束策略更新 | 需要 on-policy 轨迹，不是 test | 否 | 不适用 | 迭代训练成本高 | 提供“固定代理目标 + 分布距离约束”的思想来源，不是模型合并方法 |
| MGDA | 同步多任务梯度 | 求共同下降方向/最小范数组合 | 只需训练数据，不需 test | 通常作用于共享参数 | 否 | 每步多梯度；推理不变 | 在线训练优化，不是训练后配对向量功能投影 |
| PCGrad | 同步多任务梯度 | 对负余弦梯度做投影 | 只需训练数据，不需 test | 通常作用于共享参数 | 否 | 每步多梯度和投影；推理不变 | 处理瞬时梯度冲突；已有 v6 证据表明不能解释主要收益 |
| CAGrad | 同步多任务梯度 | 在平均梯度邻域内求 conflict-averse 更新 | 只需训练数据，不需 test | 通常作用于共享参数 | 否 | 每步优化组合；推理不变 | 在线梯度方向控制；不是端点差分或功能距离 |
| GradNorm | 多任务损失 | 依据训练速率动态平衡梯度范数/损失权重 | 只需训练数据，不需 test | 任务权重，不是逐层融合 | 否 | 额外梯度范数；推理不变 | 平衡训练速度，不约束最终模型功能位移 |
| Missing/Partial-label MLC | 部分标签未知或缺失的训练样本 | 风险重加权、部分标签损失、补全或伪标签 | 需要训练标签/部分标签；不需 test | 通常否 | 目标是恢复缺失标签本身 | 重新训练；推理不变 | 处理自然缺失标注；FC-PTV 的 SawFM 删除是有意构造的对照端点，SawFM 仍保留为输入共现背景 |
| Multi-label MTL / negative-transfer 方法 | 把标签或任务分别优化 | 梯度、任务关系或共享表示约束 | 训练数据；不需 test | 方法相关 | 通常优化平均任务，不保证指定三个标签 | 重新训练；推理不变 | FC-PTV 关注训练后配对差分以及弱分量最差标签的强制门槛 |

## 7. 新颖性审计结论与允许表述

### 7.1 不允许的表述

- “首次提出 task vector trust region”；
- “首次在功能空间做模型融合”；
- “首次逐层/分块调整 task vector”；
- “提出通用梯度冲突解决方法”；
- “已证明 FC-PTV 具有文献新颖性”；
- “已证明跨分布、跨骨干或外部泛化”。

### 7.2 当前允许的研究表述

> 本研究检验一种项目特定的功能约束：以 SawFM loss deletion 形成严格配对任务向量，以
> C6 fit-only common-five logits 的稳健尺度定义逐标签/共现层半径，并对最后 shared block
> 与 classifier block 做联合投影。该组合与 TATR、Fisher–Rao merging、AdaMerging 相邻，
> 其差异和有效性仍需 fresh development、正式基线和外部一次性验证。

## 8. 三个候选机制与唯一选择

### 8.1 候选 A：Global Functional Radius Scaling

使用一个 fit-only 功能半径把 `alpha=1.5` 全局缩小为单一 scale。

拒绝原因：它本质上仍是数据依赖的全局 alpha，无法利用 v5 已确认的 shared/classifier 贡献差异，
也难以在不降低 target mean weak recall 的同时改善最差标签。

### 8.2 候选 B：Two-block Function-space Projection（选中）

把 paired vector 严格分成 `d_shared` 和 `d_classifier`，使用逐标签、逐 SawFM 共现层的 C6
logit 稳健半径建立二维凸约束，并把 `(1.5,1.5)` 最小距离投影到可行域。

选择原因：

- 直接回答 v5 暴露的 shared-only 有害和 shared+classifier 协同；
- 只有两个系数，可完整保存、审计和单元测试；
- 系数由固定功能约束确定，不由 validation/test 指标选择；
- 可用 parameter-norm-only、classifier-only 和 orthogonal subspace 做明确反事实控制。

### 8.3 候选 C：Fisher/KL Block Projection

以 predictive Bernoulli KL 或 Fisher 二次型约束两个参数块。

暂不选择原因：与 TATR/Fisher–Rao merging 文献邻近度更高；多标签 Bernoulli 概率在饱和区可能
弱化 logit 位移；实现、数值和解释风险高于本轮需要。

本轮正式候选唯一冻结为候选 B。不得在 development-v12 结果后切换候选或把 A/C 当作补救。

## 9. FC-PTV 数学定义

### 9.1 端点与参数块

对每个训练种子 `s`：

```text
theta0_s = theta_C6_s
d_s      = theta_M5_s - theta_C6_s
d_s      = d_shared_s + d_classifier_s
```

两个端点必须共享初始化、split、manifest、fit 样本、batch 顺序、训练预算、early-stop、trainable
scope 和数值契约。M5 保留全部样本及 SawFM 共现，只屏蔽 SawFM 输出损失。非零差分只允许位于
`temporal_backbone.0.*` 与 `classifier.*`；其他 state 的差分范数必须为 0，否则该种子无效。

### 9.2 fit-only anchor

令 `A_s` 为该种子 label-long-tail optimization 中已经冻结的 `fit` 角色全部样本；按现有协议，
目标样本数为 2400，fit 占 80%，预期为 1920。不得为了功能约束另选 validation、early、
calibration、threshold 或 test 样本。

common-five 标签集合：

```text
C = {FMB, SinFM, MFT, VFT, ISRJ}
```

共现层：

```text
G = {SawFM-present, SawFM-absent}
```

如果任一 `label x stratum` 所需 C6 logit 尺度非有限或退化，种子标记为 invalid，不得从其他角色
借样本或事后改变公式。

### 9.3 C6-only 功能半径

对标签 `l` 和共现层 `g`，在 C6 fit-only logits 上定义：

```text
m_lg = median_{x in A_s,g}( z_l(x; theta0_s) )
r_lg = 1.4826 * median_{x in A_s,g}( |z_l(x; theta0_s) - m_lg| )
```

`r_lg` 是该标签/共现层的一个稳健 logit 标准差单位。要求 `r_lg` 有限且 `r_lg >= 1e-6`；否则
该种子功能契约无效。这里没有由正式性能选择的自由半径：原始半径就是 C6 的 `r_lg`，等价的
标准化半径固定为 `rho=1`。

### 9.4 功能距离

对任意两个块系数 `q=(q_shared,q_classifier)`，候选参数为：

```text
theta(q) = theta0_s + q_shared*d_shared_s + q_classifier*d_classifier_s
```

精确功能距离定义为：

```text
D_lg(q) = sqrt(
  mean_{x in A_s,g} [
    ( z_l(x; theta(q)) - z_l(x; theta0_s) ) / r_lg
  ]^2
)

D_max(q) = max_{l in C, g in G} D_lg(q)
```

该距离是 C6 锚定、逐标签、逐共现层、无量纲的 RMS logit displacement。使用 `max` 而不是全局
平均，避免一个标签或 SawFM 共现层的过大位移被其他条件掩盖。

### 9.5 两块功能签名与冻结投影

仅在 fit anchor 上计算两个单位块的标准化功能签名：

```text
u_shared_lg(x) = [z_l(x; theta0+d_shared)-z_l(x; theta0)] / r_lg
u_class_lg(x)  = [z_l(x; theta0+d_classifier)-z_l(x; theta0)] / r_lg
```

并构造 2x2 PSD Gram matrix：

```text
H_lg = mean_x [ [u_shared_lg(x), u_class_lg(x)]^T
                [u_shared_lg(x), u_class_lg(x)] ]
```

冻结目标系数为 `q0=(1.5,1.5)`。线性功能 trust region 为：

```text
Q = {
  q : 0 <= q_shared <= q_classifier <= 1.5,
      q^T H_lg q <= 1  for every l in C and g in G
}
```

FC-PTV 系数使用唯一固定目标求解：

```text
q_linear = argmin_{q in Q} ||q-q0||_2^2
```

这是“尽量保留 Full1.5，同时满足所有标签/共现层功能半径”的二维投影。`q_classifier >=
q_shared` 来自 v5 已冻结的机制证据：classifier-only 是主要正贡献之一，而 shared-only 有害。
该不等式不得由 v12 结果修改。

求解器使用 float64、固定初值 `q0`、固定容差和无随机重启；保存 KKT/可行性残差。正式评价前计算
精确 `D_lg(q_linear)`：

- 若 `D_max <= 1.05`，令 `q*=q_linear`；
- 若 `D_max > 1.05`、任何距离非有限或求解残差不合格，则该种子 FC-PTV 为 invalid；
- 不允许在看到正式评价指标后缩放、回退或重新求解。

`1.05` 只作为线性签名与精确前向之间的预注册数值容差，不改变标准化半径 `rho=1`。

最终模型：

```text
theta_FC-PTV_s = theta0_s + q*_shared*d_shared_s + q*_classifier*d_classifier_s
```

每个种子必须保存 `r_lg`、`H_lg`、`q_linear`、`q*`、KKT 残差、每个 `D_lg`、`D_max`、两个
块和总体参数范数，以及所有 logits 的有限性审计。

## 10. 为什么这不是普通 alpha 搜索

1. 不在候选 alpha 网格上比较 validation、early 或 test 性能；
2. 优化目标固定为距 `(1.5,1.5)` 的欧氏投影，不包含 AUPR、F1、Recall、FPR 或阈值；
3. 半径只由 C6 fit-only logits 的稳健尺度给出；
4. 所有种子共享同一公式、同一约束和同一容差；
5. 系数在正式 early/calibration/threshold 评价前写盘并计算 SHA256；
6. development-v12 只运行一次正式分析，不根据结果修改半径或系数公式；
7. `ParameterNormOnly` 明确检验“仅仅缩小全局 alpha”是否足以解释收益。

## 11. fresh development-v12 数据与身份隔离预注册

仓库审计发现 development-v7--v11 已在本预注册之前由其他冻结研究问题生成或使用，均不得评价
FC-PTV。为保持真正的前瞻性证据，本轮顺延到首个尚不存在的身份 `development_v12`。

预定数据角色：`development_v12`，train-only。预定路径：

```text
C:\LJQ_code\干扰数据\复合干扰数据0801_v12\closedset_6class_train.mat
```

冻结生成范围：

```text
sample seed:          [124000001, 124025200]
receiver_noise_seed:  [322677743, 322702942]
expected samples:     25200
expected unique receiver_noise_seed: 25200
```

生成前必须先实现 dry-run receipt；正式生成后必须完成：

- sample seed 连续性和范围审计；
- receiver_noise_seed 唯一性；
- 与 development-v2--v11 receiver 范围不重叠；
- 与 test-v2、sealed test-v3、reserved test-v4/v5 以及所有其他测试契约范围不重叠；
- 只读取历史 manifest/receipt/config 中的测试身份契约，不打开任何测试数据文件；
- 文件大小与 SHA256；
- `test_files_opened=false`；
- `locked_test_used=false`；
- `test_v2_used=false`；
- `test_v3_used=false`；
- 若存在 test-v4--v12 字段，全部为 false；
- `allow_locked_test=false`，`test_v2_consumed=true`，`test_v3_sealed=true`。

任何身份交集、哈希不一致或 test 文件打开记录都会使 v12 无效，禁止修补后在同一数据身份上重新做
正式选择。

## 12. development-v12 正式变体与公平性

正式评价至少包括：

1. `C6_Joint`；
2. `M5_GradientIsolated`；
3. `FullAlpha1p5`；
4. `FC_PTV`；
5. `ParameterNormOnly`；
6. `ClassifierOnlyAlpha2p0`（v5 冻结的最强 classifier-only 基线）；
7. `ModelAverage0p5`；
8. `CAGrad0p4TwoTask`（按 v6 冻结 ranking，canonical gradient baselines 中最强 worst-target 排名）；
9. `OrthogonalTaskSubspaceFunctionalMatched`。

为构造第 9 项，需在同一 v12 协议训练 `MaskFMB/MaskSinFM/MaskMFT/MaskVFT/MaskISRJ` 五个
端点，构造与 SawFM paired direction 正交的 other-label task subspace。控制方向不得用 v12 正式
性能选取；它使用与 v5 相同的确定性子空间构造，并在 fit-only anchor 上匹配 FC-PTV 的
`D_max`。若无法得到有限匹配控制，保留 invalid 结果并使机制特异性门槛失败。

`ParameterNormOnly` 定义为沿完整 `d` 的单一缩放，使其总参数 L2 距离与 FC-PTV 更新一致：

```text
lambda_param = ||q_shared*d_shared + q_classifier*d_classifier||_2 / ||d||_2
theta_param  = theta0 + lambda_param*d
```

所有方法共享：初始化、split、manifest、batch 顺序、epoch/patience、学习率、trainable scope、
gradient clipping、early-stop、temperature calibration、threshold selection、seeds 和数值裁剪。

角色分工冻结为：

- `fit`：训练端点及 FC-PTV 功能约束；
- `validation`：训练 early-stop，禁止用于 FC-PTV 系数；
- `calibration`：温度校准；
- `threshold`：阈值选择；
- `early`：唯一正式 development 评价角色；
- 所有 test：禁止访问。

正式种子固定为 `[2039,2040,2041,2042,2043]`。smoke 只允许使用合成/截断 train-only 数据验证
代码路径，不形成方法结论。

## 13. FC-PTV 相对 C6 的强制门槛

以下 10 项必须同时通过：

1. macro-AUPR delta `>= -0.005`；
2. macro-F1 delta `>= +0.010`；
3. target mean weak recall delta `>= +0.025`；
4. MFT、VFT、ISRJ 三个 mean weak recall delta 均 `>= 0`；
5. target FPR absolute `<= 0.20`；
6. 至少 4/5 种子的 target mean weak recall delta 为正；
7. SawFM-present macro-F1 delta `> 0`；
8. SawFM-absent macro-F1 delta `>= -0.005`；
9. MFT、VFT、ISRJ 三个 AUPR delta 均 `>= -0.010`；
10. 5/5 种子模型输出、功能距离和系数全部有限。

任一项失败，则 FC-PTV 自身门槛失败。

## 14. FC-PTV 相对 FullAlpha1p5 的强制改进门槛

以下 8 项必须同时通过：

1. `min_label_delta(FC-C6) - min_label_delta(Full1p5-C6) >= +0.005`；
2. target mean weak recall `(FC-Full1p5) >= 0`；
3. macro-F1 `(FC-Full1p5) >= -0.003`；
4. macro-AUPR `(FC-Full1p5) >= -0.003`；
5. target FPR `(FC-Full1p5) <= +0.005`；
6. 至少一个主要优势（最差目标 weak recall 或 target mean weak recall）的 paired bootstrap 95%
   CI 下界 `>0`；
7. `std_seed(FC-C6 target weak delta) < std_seed(Full1p5-C6 target weak delta)`；
8. 所有功能距离、参数距离、logits、概率、校准和阈值输出有限。

若不能超过 FullAlpha1p5，立即停止 FC-PTV；不得放宽门槛、修改半径或在新数据上重复试到通过。

## 15. 三目标标签同时改善与机制特异性

### 15.1 同时改善的统计定义

除第 13 节的三个标签均值非负外，只有同时满足以下条件，才允许声称“三个目标标签同时改善”：

1. MFT、VFT、ISRJ 的 `FC-C6` weak recall mean 全部 `>0`；
2. 每个标签至少 4/5 种子的 paired delta 非负；
3. 使用同一组 10,000 次 paired seed bootstrap 重采样索引构造 family-wise 95% max-stat
   simultaneous lower bounds，三个标签下界均 `>0`；
4. 不允许用 target mean 掩盖任一标签失败。

若只通过第 13 节而未通过 simultaneous bounds，只能表述为“均值均非负”，不能表述为
“三个标签均获得统计可靠改善”。

### 15.2 机制特异性门槛

FC-PTV 通过第 13、14 节后，还必须满足：

- 相对 `ParameterNormOnly`：target mean weak recall 或最差标签 weak recall 至少一项 `>=+0.005`
  且 paired 95% CI 下界 `>0`；
- 相对 `ClassifierOnlyAlpha2p0`：至少一项主要 weak 指标 `>=+0.005`，macro-AUPR/F1 各不劣于
  `-0.003`；
- 相对 orthogonal functional-matched control：target mean weak recall `>=+0.005`，macro-F1
  不劣于 `-0.003`；
- 5/5 种子的 `q_shared>0`，否则方法退化为 classifier-only，机制身份失败。

这些控制用于区分功能联合投影、纯参数缩小、纯 classifier 调整和任意结构化子空间扰动。

## 16. 统计报告与成本报告

对所有主要比较保存逐种子 paired delta，并使用固定 10,000 次 seed bootstrap 报告 mean、sample
std 和 95% CI。不得只报告跨种子平均值。报告至少包括：

- macro-AUPR、macro-F1、target weak recall、target FPR；
- SawFM-present/absent macro-F1；
- 每标签 AUPR；
- MFT/VFT/ISRJ 每标签 weak recall；
- 最差种子、最差标签、positive seed count；
- simultaneous lower bounds；
- `q_shared/q_classifier`、每标签/共现层 `r_lg` 与 `D_lg`；
- train seconds、peak GPU memory、梯度计算次数、额外 anchor forward 次数；
- endpoint 数量、checkpoint 大小、最终单模型推理延迟和参数量。

所有 bootstrap 随机种子、重采样索引规则和 metric path 在正式运行前冻结。

## 17. 晋级、停止和后续阶段

### 17.1 本轮决策

- 第 13、14、15 节全部通过：`promote_to_full_canonical_baselines`；
- 自身门槛通过但 Full1.5 或机制特异性失败：`stop_fc_ptv`；
- 自身门槛失败：`stop_fc_ptv`；
- 数值/身份/测试边界失败：`invalid_and_stop_fc_ptv`。

### 17.2 全通用基线阶段

仅在 v12 全部门槛通过后，才能在新的 identity-isolated development-v13 上一次性比较 PCGrad、
MGDA、CAGrad、GradNorm、loss weighting、M5、task arithmetic、model averaging 和 FC-PTV。
预留但不生成的 v13 身份范围为：

```text
sample seed:          [134000001, 134025200]
receiver_noise_seed:  [332677743, 332702942]
```

所有方法必须共享初始化、split、manifest、batch 顺序、训练预算、early-stop、calibration、
threshold、seeds 和 trainable scope，并报告时间、显存、梯度次数和推理成本。

### 17.3 鲁棒性与外部一次性测试

只有 v12 内部机制和 v13 全基线均通过后，才能依次预注册 SawFM 共现率/强度、SNR/JSR、功率分层、
label long-tail、未见组合和第二骨干。至少 80% 条件为正、最差条件无实质 macro-AUPR 退化，
才允许冻结代码、模型、阈值、校准和统计方法并执行一次外部测试。

test-v2 已消耗；test-v3 继续密封；历史 locked-test 禁止访问。外部测试后禁止修改、选择、校准或
重跑。

### 17.4 机制循环上限

FC-PTV 是新的机制循环 1。若 v12 失败，本代号立即停止。只有提出实质不同、重新完成文献审计与
新预注册的机制，才可开始机制循环 2；不得把改变 `rho`、容差、系数边界或 alpha 视为新机制。
连续两个 fresh-development 机制失败时，功能约束方向整体停止；总计最多三个新机制循环。

## 18. 实现前冻结清单

在生成 v12 或实现正式 runner 前，必须确认本文件已保存并哈希；随后才允许：

1. 新增 v12 config、MATLAB generator 和 identity auditor；
2. 实现纯函数形式的 robust scale、Gram、二维投影、exact-distance audit；
3. 编写单元测试：零方向、退化尺度、可行/不可行 q0、block support、KKT、非有限输出、角色隔离；
4. 运行不含正式证据的 smoke；
5. 在代码与门槛再次冻结后生成并审计 v12；
6. 只运行一次正式 5-seed development-v12。

禁止在本文件哈希前生成数据或实现正式 FC-PTV 模型。

## 19. 每轮强制输出结构

后续每轮报告必须包含：

1. 本轮结论；
2. 最近三轮结果；
3. 科学问题；
4. 可证伪假设；
5. 数据与身份隔离；
6. 基线和候选；
7. 方法公式；
8. 实现改动；
9. 正式结果；
10. CI 与种子一致性；
11. 每标签弱分量结果；
12. 机制分析；
13. 失败与替代解释；
14. 预注册门槛；
15. 证据等级；
16. 产物与 hash；
17. 下一轮唯一实验。

## 20. 原始文献来源

1. [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)
2. [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://proceedings.mlr.press/v162/wortsman22a.html)
3. [TIES-Merging: Resolving Interference When Merging Models](https://papers.nips.cc/paper_files/paper/2023/hash/1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html)
4. [AdaMerging: Adaptive Model Merging for Multi-Task Learning](https://arxiv.org/abs/2310.02575)
5. [Merging Models with Fisher-Weighted Averaging](https://arxiv.org/abs/2111.09832)
6. [Dataless Knowledge Fusion by Merging Weights of Language Models](https://arxiv.org/abs/2212.09849)
7. [Task Arithmetic in Trust Region](https://arxiv.org/abs/2501.15065)
8. [Functionality-Oriented LLM Merging on the Fisher--Rao Manifold](https://arxiv.org/abs/2603.04972)
9. [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
10. [Gradient Surgery for Multi-Task Learning](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
11. [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/abs/2110.14048)
12. [Multi-Task Learning as Multi-Objective Optimization](https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html)
13. [GradNorm](https://proceedings.mlr.press/v80/chen18a.html)
14. [Large-scale Multi-label Learning with Missing Labels](https://arxiv.org/abs/1307.5101)
15. [Learning a Deep ConvNet for Multi-label Classification with Partial Labels](https://arxiv.org/abs/1902.09720)
16. [A Multi-Task Gradient Descent Method for Multi-Label Learning](https://arxiv.org/abs/1911.07693)

## 21. 当前证据边界

- 本文件是 `plan/preregistration` 证据，不是模型性能证据；
- development-v4/v5/v6 仅作为历史机制依据，没有用于评价 FC-PTV；
- 本阶段没有生成 v12 数据；development-v7--v11 已在本预注册之前被其他研究问题占用；
- 本阶段没有实现或运行正式 FC-PTV；
- 没有打开任何测试数据文件；
- `locked_test_used=false`；
- `test_v2_used=false`；
- `test_v3_used=false`；
- 外部测试未授权。
