# 路线 A：DFMS-IQ 延迟融合多尺度主干正式筛选方案（2026-07-24）

## 1. 本轮要回答的问题

本轮不再继续调整分类器，而是验证两个主干假设：

1. **H1：深层保持复数 I/Q、延迟融合，能提高弱正—负样本可分性。**
2. **H2：在深层复数特征上加入多滞后相位支路，能在不增加误报的前提下进一步提升弱分量。**

本轮属于 development 结构筛选，`locked_test_used=false`。正式试验开始后，不根据中途结果修改模型、损失、训练预算或晋级门槛。

## 2. 冻结模型矩阵

| 编号 | 模型入口 | width | 参数量 | 作用 |
|---|---|---:|---:|---|
| B1 | `complex_1d_true` | 56 | 260,630 | 容量匹配 Base-DB 强基线 |
| P0 | `phase_multiscale` | 16 | 253,325 | 当前早期融合 PAM-IQ |
| F1 | `dfms_iq_no_phase` | 16 | 240,806 | 深层复数多尺度 + 延迟 Cartesian/幅度融合 |
| F2 | `dfms_iq` | 16 | 257,511 | F1 + lag=1/4/16 深层相位支路 |

F2 与 B1 参数量相差约 1.20%；F1 比 B1 少约 7.61%。不允许通过扩大 width 补偿首轮结果。

## 3. 数据与长尾协议

- 数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`
- 配置：`configs/student_a/six_class.yaml`
- 六标签：FMB、SawFM、SinFM、MFT、VFT、ISRJ
- 协议：`label_lt`
- 训练样本：2400
- 标签不平衡因子：4
- SawFM 尾类锚点：400
- 标签排序：`0,2,3,4,5,1`
- seeds：`2041,2042,2043`
- 同 seed 的四个模型必须共享 split SHA256 和 manifest SHA256。
- 每个 development evaluation role 最多使用300个样本。
- 禁止读取 `closedset_6class_test.mat`。

## 4. 损失、优化器和训练过程

为隔离主干贡献，四个模型完全使用相同训练设置：

| 参数 | 数值 |
|---|---:|
| classification loss | 完整 DB Loss |
| optimizer | AdamW |
| learning rate | 3e-4 |
| weight decay | 1e-4 |
| batch size | 64 |
| epochs | 20 |
| patience | 20 |
| warmup | 10% updates |
| gradient clip | 5.0 |
| dropout | 0.20 |
| AMP | 开启 |
| checkpoint selection | macro-AUPR |
| workers | 0 |

不使用 ASL 混合、弱正加权、标签查询、注意力池化、功率配对、prototype、蒸馏或参数回归。

## 5. 阈值与主要指标

所有模型保存并报告：

- fixed 0.5；
- global threshold；
- per-label F1；
- per-label F2；
- shrinkage per-label F1；
- per-label Youden。

主要模型排序指标仍为阈值无关的 macro-AUPR。主要决策点使用 shrinkage per-label F1；F2 只作为召回敏感诊断。

除原有指标外，本轮增加：

- weak-positive vs negative AUROC；
- weak-positive vs negative AUPR；
- strong-positive、weak-positive、negative 分数中位数；
- 主阈值和 F2 阈值下的 negative FPR；
- 六标签逐标签 weak-vs-negative AUROC。

## 6. 执行阶段

### 6.1 工程 smoke

四模型各执行：

- seed=2041；
- 1 epoch；
- train/eval 每角色最多64个样本；
- GPU/AMP；
- 检查输出形状、梯度、receipt、阈值文件、manifest 和 locked-test 标记。

smoke 只验证工程路径，不进入性能结论。

### 6.2 正式结构筛选

执行 `4 models × 3 seeds × 20 epochs = 12 runs`。每个 run 保存完整：

- scientific config 和配置哈希；
- environment；
- history；
- best checkpoint；
- calibration/thresholds；
- predictions；
- split/manifest hashes；
- run receipt。

### 6.3 冻结特征诊断

正式训练结束后先检查输出分数分布。如果 F1 或 F2 的弱正—负 AUROC相对 P0 有提高、但端到端弱召回没有提高，则再执行冻结 `forward_features` 的逐标签两层 MLP probe，以区分“表示已改善但线性头没利用”与“主干仍未保留弱证据”。

触发后的 probe 协议冻结为：

| 参数 | 数值 |
|---|---:|
| encoder/checkpoint | 对应 run 的 best checkpoint，完全冻结并设为 eval |
| probe training data | 对应 manifest 的2400个训练样本 |
| probe split | 训练特征确定性80%/20%拆分 |
| feature normalization | 使用 probe-train 均值/标准差 |
| linear control | `Linear(feature_dim,6)` |
| nonlinear probe | 每标签独立 `Linear(D,64)-GELU-Dropout(0.1)-Linear(64,1)` |
| loss | 基于 probe-train 阳性率的 pos-weight BCE |
| optimizer | AdamW，lr=1e-3，weight decay=1e-4 |
| batch size | 128 |
| epochs/patience | 50/10 |
| checkpoint selection | probe-validation macro-AUPR |
| evaluation | 原 development early-stop role，不调阈值 |

如果 MLP probe 相对 P0 提升、但仍不及 B1，结论应表述为“延迟融合修复了 P0 的部分表示缺陷”，不能表述为新主干已经解决弱分量问题。

## 7. 晋级规则

### 7.1 F2 相对 B1 的正式晋级门槛

必须同时满足：

1. `Δ macro-AUPR >= -0.005`；
2. `Δ SawFM-AUPR >= 0`；
3. `Δ weak-component recall >= +0.030`；
4. `Δ weak-vs-negative AUROC >= +0.020`；
5. `Δ Hamming loss <= +0.005`；
6. 至少 2/3 seeds 的 weak-vs-negative AUROC 为正增益；
7. SinFM、MFT、VFT 任一 AUPR 下降不超过0.030。

### 7.2 机制判定

- F1 优于 P0/B1，F2 继续提高：延迟融合和多滞后相位均有效。
- F1 提高、F2 退化：延迟融合有效，相位支路仍受噪声污染，后续使用 F1。
- F1/F2 都只改善阈值召回、不改善 weak-vs-negative AUROC：属于误报放大，不晋级。
- F1/F2 均不优于 P0：不再把主要失败归因于 I/Q 融合时机，返回弱正训练目标和数据可辨识性验证。

## 8. 输出位置

```text
outputs/tapc/route_a_dfms_iq_structure_screen_smoke/
outputs/tapc/route_a_dfms_iq_structure_screen/
docs/results/route_a_dfms_iq_structure_screen_results_20260724.md
```

## 9. 执行状态（2026-07-24）

- 四模型 GPU smoke：4/4完成。
- 正式端到端筛选：12/12完成。
- 冻结线性/MLP probe：12/12 checkpoints完成。
- locked test未使用。
- F1/F2完整弱分量门槛均未通过。
- F2相对P0显著修复总体表示，且冻结probe接近B1，但主工作点弱召回仍不足。

完整结果见：

`docs/results/route_a_dfms_iq_structure_screen_results_20260724.md`
