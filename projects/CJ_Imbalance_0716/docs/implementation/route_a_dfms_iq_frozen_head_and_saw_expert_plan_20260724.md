# 路线 A：DFMS-IQ 冻结分类头与 SawFM 专家下一阶段试验方案（2026-07-24）

## 1. 试验目的与证据边界

上一轮已经证明，完整 DFMS-IQ（F2）相对 PAM-IQ 提升了总体 Macro-AUPR、Hamming loss 和弱正—负样本可分性，但在原 DB 分类头与 shrinkage-F1 阈值下弱分量召回仍明显低于 B1。冻结线性 probe 接近 B1，说明 F2 表示中仍保留了相当一部分弱证据。

本轮只回答三个问题：

1. F2 的主要剩余瓶颈是否来自联合训练得到的分类边界，而不是主干表示完全丢失？
2. 弱正—难负排序目标是否能在不显著增加误报的情况下提高弱证据排序？
3. SawFM 是否需要一个不依赖相位分支的 Cartesian/幅度专用证据通道？

本轮仍属于 development 试验，`locked_test_used=false`。所有分类头、温度和阈值只使用训练、calibration 和 threshold-selection 角色；early-stop development 角色只用于最终诊断，不参与头训练、温度拟合或阈值选择。

## 2. 固定数据与检查点

- 数据：`C:\LJQ_code\干扰数据\复合干扰数据0716\closedset_6class_train.mat`
- 环境：`D:\anaconda3\envs\PY10\python.exe`
- 协议：六类 `label_lt`，样本数 2400，IF=4，SawFM 锚点 400
- 类别顺序：`FMB, SawFM, SinFM, MFT, VFT, ISRJ`
- 随机种子：`2041, 2042, 2043`
- 冻结编码器：上一轮各 seed 的 `F2_DFMS_Full/best_model.pth`
- 训练、calibration、threshold-selection、early-stop 划分必须复用对应 seed 的 split SHA256；训练清单必须复用 manifest SHA256。
- 弱分量定义：标签存在且对应分量功率不高于 10 dB。

## 3. 冻结模型矩阵

| 编号 | 主特征 | 分类头/损失 | SawFM 专家 | 作用 |
|---|---|---|---|---|
| J0 | F2 | 上一轮联合训练 DB 原头 | 无 | 既有端到端控制，不重新训练 |
| H1 | 冻结 F2 128维 | 新线性头 + DB Loss | 无 | 隔离“重新拟合头”本身的贡献 |
| H2 | 冻结 F2 128维 | 新线性头 + pos-weight BCE | 无 | 验证标签频次权重与 DB 边界差异 |
| H3 | 冻结 F2 128维 | H2 + 弱正—难负排序 | 无 | 直接提高弱正相对高分负样本的排序 |
| H4 | 冻结 F2 128维 | H3 | 有 | 验证 SawFM 的非相位局部证据补偿 |

H1—H4 的基础头均为 `Linear(128, 6)`。H4 另外从同一冻结 F2 中提取去除相位残差后的 Cartesian+幅度特征：

```text
深层复数编码器输出 (real, imag)
  ├─ F2主路：Cartesian + magnitude + 0.1~0.22×multi-lag phase
  │            → temporal block → statistics pool → 128维主特征
  └─ Saw专家：Cartesian + magnitude（不含phase）
               →共享冻结temporal block →共享冻结pool/projection → 128维专家特征

SawFM logit = main SawFM logit + 0.20 × expert logit
```

专家融合系数固定为 0.20，避免专家通过任意放大 logit 绕过“低权重”约束。专家只修改 SawFM 输出，其余五类严格使用主分类头。

## 4. 损失函数

### 4.1 H1：冻结 DB 头

复用项目 `DistributionBalancedLoss` 默认参数，对训练清单的六类阳性计数重新实例化。该组不是复用联合训练权重，而是在完全冻结 F2 后从随机初始化重新拟合线性分类边界。

### 4.2 H2：正样本加权 BCE

每类：

```text
pos_weight[c] = negative_count[c] / positive_count[c]
```

使用 `BCEWithLogitsLoss(pos_weight=...)`，不额外增加弱样本权重，以便单独判断标签频次校正的贡献。

### 4.3 H3/H4：弱正—难负排序

基础损失仍为 pos-weight BCE。每个 batch、每个标签分别取：

- 真实存在且功率不高于 10 dB 的弱正 logit；
- 当前 batch 中该标签为负的 logit；
- 从负样本中选取得分最高的 `min(弱正数, 负样本数)` 个难负样本。

排序项为：

```text
L_rank = mean softplus(margin - logit_weak + logit_hard_negative)
L = L_pos_BCE + 0.20 × L_rank
margin = 0.20
```

不把低可见度样本改写成负标签；弱分量掩码只用于已知阳性内部的排序监督。

## 5. 头训练与模型选择

| 参数 | 固定值 |
|---|---:|
| F2 编码器 | 完全冻结、`eval()` |
| train-feature split | 对 2400 个清单样本确定性 80%/20% |
| 特征标准化 | 仅用 80% head-fit 均值/标准差 |
| optimizer | AdamW |
| learning rate | 1e-3 |
| weight decay | 1e-4 |
| batch size | 256 |
| epochs | 60 |
| patience | 12 |
| dropout | 无（线性头） |
| gradient clip | 5.0 |
| checkpoint selection | head-validation Macro-AUPR；同分时取弱正—负 AUROC 更高者 |

三个 seed 的模型矩阵、超参数、训练轮数和数据角色完全一致，中途不按结果修改。

## 6. 温度与 FPR 约束阈值

每个头都在独立 calibration 角色拟合一个标量温度。阈值只在独立 threshold-selection 角色上拟合。

主阈值从普通 F1 改为逐标签约束优化：

```text
maximize    weak-positive recall
subject to  empirical negative FPR <= 0.20
grid        0.01 ... 0.99，步长0.01
tie-break   全阳性F1更高 → 阈值更接近0.5
```

若某个标签在 threshold-selection 角色没有弱正样本，则回退为同一 FPR 约束下最大化全部阳性召回，并在结果中显式记录回退。除约束阈值外，同时报告 fixed-0.5 和 per-label-F1，防止只靠降阈值制造表面召回提升。

## 7. 正式指标

阈值无关指标：

- Macro-AUPR、逐标签 AUPR；
- 弱正—负 AUROC、AUPR；
- SawFM 弱正—负 AUROC、AUPR；
- 弱正、强正、负样本分数分布。

FPR 约束主阈值指标：

- weak-component recall；
- Macro-F1、Hamming loss；
- 总体和逐标签 negative FPR；
- SawFM weak recall、negative FPR；
- 阈值从 threshold-selection 到 early-stop 的 FPR 漂移。

## 8. 冻结阶段晋级门槛

以 H1 冻结 DB 头为主要归因基线。H4 只有同时满足以下条件才允许进入末级小学习率解冻：

1. `Δ Macro-AUPR >= -0.005`；
2. `Δ weak-vs-negative AUROC >= +0.015`；
3. FPR约束阈值下 `Δ weak-component recall >= +0.050`；
4. `Δ Hamming loss <= +0.005`；
5. early-stop 总体 negative FPR 不超过 0.25，且相对 H1 增量不超过 0.02；
6. `Δ SawFM weak-vs-negative AUROC >= +0.020`，且 SawFM-AUPR 不下降；
7. 至少 2/3 seeds 的 weak-vs-negative AUROC 和弱分量召回均为正增益。

同时报告 H2、H3 相对 H1 的逐级变化，以区分“换 BCE”“加排序”和“加 Saw 专家”的独立贡献。即使 H4 未晋级，这些逐级对比仍是有效的机制证据。

## 9. 条件末级解冻方案

仅当第 8 节完整门槛通过时执行 U1：

- 初始化：对应 seed 的 H4 最优头与原 F2 检查点；
- 冻结 complex block1、block2、stage3、late-fusion 和统计投影；
- 只解冻 `complex_encoder.stage4`、`temporal_backbone` 和 H4 分类头；
- stage4/temporal learning rate：`1e-5`；
- 分类头 learning rate：`2e-4`；
- AdamW，weight decay `1e-4`；
- 10 epochs，patience 5，batch size 64，gradient clip 5；
- 损失保持 H4，不在解冻阶段改变 margin、排序权重或专家系数；
- 仍使用相同 calibration、threshold-selection 和 early-stop development 角色；
- U1 必须至少保持 H4 的 Macro-AUPR、弱分量 AUROC 和 FPR约束召回，才可进入之后的五种子确认试验。

若冻结门槛未通过，则本轮停止于 H1—H4，不执行 U1，避免用主干微调掩盖分类目标本身无效。

## 10. 输出位置

```text
outputs/tapc/route_a_dfms_iq_frozen_heads/
docs/results/route_a_dfms_iq_frozen_heads_results_20260724.md
```

每个 seed/variant 保存：配置、检查点哈希、split/manifest 哈希、训练历史、温度、三种阈值、early-stop 概率与预测、完整指标和 `locked_test_used=false` 回执。

## 11. 执行状态（2026-07-24）

- GPU smoke：1 seed × 5 variants 完成；
- 正式冻结试验：3 seeds × 5 variants 完成，其中 H1—H4 共 12 个头训练；
- threshold-selection 的 90 个标签单元均满足 FPR≤0.20；
- H4 相对 H1 的完整晋级门槛未通过；
- 按预注册规则未执行末级解冻 U1；
- locked test 未使用；
- 结果见 `docs/results/route_a_dfms_iq_frozen_heads_results_20260724.md`。
