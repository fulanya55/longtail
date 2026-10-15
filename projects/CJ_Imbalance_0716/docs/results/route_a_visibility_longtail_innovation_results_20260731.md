# Route A 标签内可见性长尾创新路线：完整实验与独立验证报告

日期：2026-07-31  
证据状态：development-v2 五种子通过；test-v2 一次性独立验证通过；禁止重跑或调参。

## 1. 最终结论

本阶段已经形成一个通过独立测试验证的创新方法：

> **标签内可见性时序专家与可见性分层均衡融合（Visibility Temporal
> Statistics Expert + Visibility-Balanced Fusion，VTSE-VBF）**。

该方法针对复合干扰中的“标签内可见性长尾”：同一标签的强分量样本远多于弱
分量样本，普通 DB/BCE 只处理标签频次，无法阻止强正样本主导分类边界。

VTSE-VBF 将任务拆为：

1. 冻结 C6 分类器判断常规标签存在；
2. VTSE 只学习 MFT/VFT/ISRJ 的“弱正 vs 标签缺失负样本”；
3. 融合器对弱正、强正、负样本三层等权优化；
4. 用平方根不确定性门控和 ±0.50 硬上限安全接回基线。

在新的独立 test-v2 25,200 条样本上，相对冻结 C6：

- 五标签 macro-AUPR：+0.00622；
- 五标签 macro-F1：+0.01087；
- MFT/VFT/ISRJ 平均弱分量召回：+0.04023；
- 三目标弱正—负 AUROC：+0.02411；
- 三目标强正召回：+0.02579；
- SawFM-present 五标签 macro-F1：+0.01118；
- SawFM-absent 五标签 macro-F1：+0.01045；
- 5/5 个种子弱召回均为正增益；
- 目标平均 FPR=0.19667，满足 0.20 约束。

development-v2 和 test-v2 的效应量高度一致，因此创新点已经具备方法、消融、
多种子、泛化和安全边界五部分证据。

## 2. 当前最终网络结构

### 2.1 冻结 F2 主干

- 复数 I/Q 输入 `[B,2,3000]`；
- delayed complex multi-scale encoder；
- Cartesian、幅度与多时延相位特征在后期残差融合；
- multi-scale temporal backbone；
- global mean/std pooling 与投影得到 128 维全局特征。

### 2.2 冻结 C6 基线头

- `Linear(128,6)`；
- pos-weight BCE；
- 全局特征、头部参数及其标准化参数均冻结。

### 2.3 VTSE 弱证据专家

从冻结 final temporal map `[B,128,T]` 提取：

- mean；
- std；
- max；
- top-8 mean。

拼接得到 512 维特征，使用 `Linear(512,3)` 分别识别 MFT、VFT、ISRJ 的弱正
与标签缺失负样本。强正样本在 VTSE 训练中被 mask，避免再次主导专家边界。

### 2.4 可见性分层均衡融合器

每个目标标签只有两个可训练标量：expert weight 与 bias，共 6 个参数。

`raw_k = softplus(alpha_k)*expert_z_k + bias_k`

`gate_k = sqrt(4*p_base_k*(1-p_base_k))`

`residual_k = 0.50*tanh(raw_k)*gate_k`

`final_logit_k = base_logit_k + residual_k`

训练时分别计算弱正、强正、负样本 BCE，再三层等权平均。非目标标签完全不变。

### 2.5 校准与阈值

- 每种子、每变体在 development calibration role 冻结标量温度；
- development threshold-selection role 按逐标签 FPR<=0.20 最大化弱召回；
- test-v2 完全复用冻结温度和阈值，没有重新校准或选择。

## 3. 本阶段逐轮试验总结

### 3.1 H5 SawFM 背景消融

问题：SawFM 作为第六输出是否压制其他五类？

结果：H5 相对 C6 的 macro-AUPR +0.00146、macro-F1 +0.00244、目标弱召回
+0.00842；5/5 种子方向为正，但远低于 +0.03 门槛。

结论：SawFM 输出竞争是轻微因素，不是主要根因。

### 3.2 有界 final-temporal 局部对比读出器

整改旧 Q1/Q2 的类别先验漂移：热图时间零中心、删除类别 bias、残差限制 ±0.35，
并加入弱正—难负排序。

L2 相对 C6：macro-F1 -0.00076、弱召回 +0.00447、弱 AUROC +0.00015。
平均残差已达 0.087、最大约 0.335，说明失败不是读出器作用太小。

结论：用 label-LT 小训练集直接学习复杂局部热图，不能稳定利用 final temporal
中的弱证据；需要先进行独立可辨识性验证。

### 3.3 stage2/stage3/temporal 可辨识性筛选

使用完整 optimization role，排除强正，仅训练弱正—负线性 probe。

| 表征 | 三目标 weak AUROC |
|---|---:|
| 冻结 C6 | 0.61067 |
| Global | 0.74202 |
| Final temporal | **0.76057** |
| Stage2 physics | 0.72757 |
| Stage3 physics | 0.70958 |
| Stage2+3 | 0.73705 |

Final temporal 的五种子增益为 +0.139～+0.174，validation–early gap=-0.0047，
所有可辨识性门槛通过。

关键结论：F2 中存在充分弱证据；之前失败的根因是训练目标和样本分层，而不是
主干完全丢失信息。这一结果确立“标签内可见性长尾”作为独立问题轴。

### 3.4 手工 VTSE 有界接回

线性 uncertainty gate：macro-F1 +0.00655、弱召回 +0.02736、弱 AUROC
+0.01283，全部方向一致但略低于效应量门槛。

平方根 gate：弱 AUROC 提升至 +0.01723，但 macro-F1 +0.00686、弱召回
+0.02687，说明简单调整作用强度无法完成 score-to-label 边界对齐。

### 3.5 可见性分层均衡融合器

只增加六个标量参数，以弱正、强正、负样本三层等权 BCE 学习 score 映射。

development-v2：

| 指标 | C6 | VTSE-VBF | 差值 |
|---|---:|---:|---:|
| macro-AUPR | 0.87681 | 0.88314 | +0.00633 |
| macro-F1 | 0.76928 | 0.77976 | +0.01048 |
| 目标弱召回 | 0.33176 | 0.37225 | +0.04048 |
| 目标弱 AUROC | 0.61067 | 0.63588 | +0.02521 |
| 目标 FPR | 0.19622 | 0.19571 | -0.00051 |
| 目标强召回 | 0.83136 | 0.85580 | +0.02444 |

九项晋级门全部通过，随后冻结完整候选。

## 4. test-v2 一次性独立结果

test-v2 基础种子与 development-v2 身份隔离，包含 25,200 条、63 种组合。
执行前验证 SHA-256，执行器先写 one-shot guard；当前
`rerun_allowed=false`，test-v2 不得再用于任何调整。

| 指标 | C6 | VTSE-VBF | 差值 |
|---|---:|---:|---:|
| macro-AUPR | 0.87550 | 0.88172 | +0.00622 |
| macro-F1 | 0.76527 | 0.77614 | +0.01087 |
| 目标弱召回 | 0.31475 | 0.35498 | +0.04023 |
| 目标弱 AUROC | 0.60490 | 0.62901 | +0.02411 |
| 目标 FPR | 0.19477 | 0.19667 | +0.00189 |
| 目标强召回 | 0.83264 | 0.85842 | +0.02579 |

逐目标标签：

| 标签 | C6 AUPR | 候选 AUPR | C6 F1 | 候选 F1 | C6 弱召回 | 候选弱召回 | 候选 FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFT | 0.90103 | 0.90989 | 0.79715 | 0.81444 | 0.35261 | 0.40298 | 0.18884 |
| VFT | 0.84523 | 0.85539 | 0.71505 | 0.73104 | 0.28092 | 0.31487 | 0.19711 |
| ISRJ | 0.86853 | 0.88061 | 0.73708 | 0.75812 | 0.31073 | 0.34710 | 0.20405 |

ISRJ 单标签 FPR 为 0.20405，略高于 0.20；三目标平均仍为 0.19667。该现象只能
作为部署风险记录，禁止再根据 test-v2 调整 ISRJ 阈值。

## 5. SCI 创新点及证据链

### 创新点一：标签内可见性长尾建模

现有 DB/长尾方法主要按标签频次重加权。本方法指出复合信号中还存在同标签内部
的强正/弱正数量和可辨识性失衡，并通过 C6 0.6107 与 VTSE 0.7606 的弱证据
AUROC 差异证明该轴独立存在。

### 创新点二：存在分类与弱可见性识别解耦

C6 保留常规分类能力，VTSE 排除强正后只学习弱正—负边界。H5、普通局部读出器
和独立 probe 的连续消融证明，收益来自监督目标解耦，而不是简单增加参数量。

### 创新点三：可见性分层均衡融合

弱正、强正和负样本三层等权，解决强正主导问题；仅用六个标量完成融合，参数
效率高且归因清晰。

### 创新点四：不确定性门控的安全残差接回

平方根置信门与 `tanh` 硬上限限制专家只修正基线不确定区域。development 和
test-v2 均表现为强正召回提高、平均 FPR 受控，证明不是简单放宽阈值。

### 证据完整性

- 负机制排除：H5、自由标签查询、top-k、排序头；
- 表征充分性：五表征、五种子独立 probe；
- 机制消融：线性 gate、平方根 gate、分层均衡融合；
- 多种子：2039--2043；
- 新 development-v2；
- 一次性 identity-isolated test-v2；
- 所有模型、校准器、阈值、哈希、预测和 receipt 保存完整。

## 6. 当前结论边界

1. 创新效果当前直接覆盖 MFT/VFT/ISRJ，不应宣称所有六标签均由 VTSE 改善。
2. SawFM 的主要创新结果仍属于 RawSaw 路线；将 RawSaw 与 VTSE-VBF 合并需要新的
   development-v3/test-v3，不能再使用当前 test-v2 选择。
3. 当前数据来自同一仿真生成体系；真实数据、参数范围外和接收机域偏移仍需外部验证。
4. ISRJ 独立测试 FPR 略高，后续只能在全新数据协议中研究风险控制。

## 7. 后续论文级验证计划

- 在新 development-v3 上做完整消融：无 VTSE、无 strong mask、普通不均衡 BCE、
  无 uncertainty gate、无 residual cap、stage2/stage3 替代专家；
- 报告参数量、FLOPs、推理时延和五种子置信区间；
- 按功率差、组合阶数、共现标签、receiver_noise_seed 分层；
- 构建参数范围外、噪声模型变化和真实/半实物数据外部集；
- 与 class-balanced loss、focal/ASL、logit adjustment、两阶段解耦和主流
  multi-label long-tail 方法公平比较；
- 若整合 RawSaw，必须新建 development-v3/test-v3 并重新冻结一次，不得回看
  当前 test-v2。

## 8. 关键产物

- development 总结：
  `outputs/tapc/route_a_visibility_balanced_fusion_v2/route_a_visibility_balanced_fusion_v2_summary.json`
- 冻结包：`frozen/20260731_route_a_visibility_balanced_fusion_v2/`
- test-v2 总结：
  `frozen/20260731_route_a_visibility_balanced_fusion_v2/test_v2_results/route_a_visibility_balanced_fusion_test_v2_summary.json`
- test-v2 receipt：
  `frozen/20260731_route_a_visibility_balanced_fusion_v2/test_v2_execution_receipt.json`
- 本报告：`docs/results/route_a_visibility_longtail_innovation_results_20260731.md`

