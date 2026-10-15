# 路线 A：剔除 SawFM 后五类频次长尾首轮实验报告

日期：2026-08-26  
状态：`[A]` 已完成开发验证；统一五类频次长尾创新尚未通过准入门槛

## 1. 本轮目标与数据边界

本轮只研究标签频次长尾，不研究功率长尾、弱分量可见性、半监督或开集识别。类别顺序固定为：

`FMB / SinFM / MFT / VFT / ISRJ`

训练集与测试集均删除 SawFM 阳性样本（不是把 SawFM 阳性样本当成五类负样本）。原始 0716 闭集训练/测试各 25,200 个样本；过滤后训练与测试各保留 12,400 个样本，五类过滤后正样本数均为 6,400。没有使用九档功率扫描，也没有使用 locked-test、test-v2 或 test-v3。

产物：

- [实验 Goal Prompt](../prompts/route_a_five_class_frequency_lt_v1_goal_prompt_20260826.md)
- [源过滤审计](../../../干扰数据/复合干扰数据0716/route_a_five_class_frequency_lt_v1/source_filter_audit.json)
- [LabelIF4 manifest](../../../干扰数据/复合干扰数据0716/route_a_five_class_frequency_lt_v1/label_if4_manifest.json)
- [等预算正式 receipt](../../outputs/route_a_five_class_frequency_lt_v1_equal_budget/run_receipt.json)
- [等预算实验脚本](../../experiments/run_route_a_five_class_frequency_lt_v1.py)

## 2. 当前网络结构

固定使用路线 A 已选定的 B0：`DelayedFusionMultiScaleIQClassifier (dfms_iq)`，代码位于 [backbone_adapter.py](../../cjlt/models/backbone_adapter.py) 和 [phase_multiscale.py](../../cjlt/models/phase_multiscale.py)。本轮只将输出类别由 6 改为 5，没有重做主干。

- 输入：`[B, 2, 3000]`，实部/虚部分离，`complex_rms` 归一化。
- Stage 1：复数残差卷积，1→16，kernel 7，时间长度 3000。
- Stage 2：16→32，kernel 5，下采样到 1500。
- Stage 3：32→48，多尺度 kernel `(3,7,15)`、dilation `(1,2,4)`，长度 750。
- Stage 4：48→64，同样的多尺度/空洞分支，长度 375。
- 延迟融合：Cartesian 64→128、幅度 64→128、相位差分滞后 `(1,4,16)`，在深层融合，避免 I/Q 过早混合。
- 时序骨干：128 通道多尺度深度可分离残差块，kernel `(3,7,15)`、dilation `(1,2,4)`，BatchNorm、SiLU、dropout 0.2。
- 池化：无注意力的全局 mean/std，256 维；`Linear(256→128)+LayerNorm+GELU+Dropout(0.2)`。
- 分类器：`Dropout(0.2)+Linear(128→5)`。

宽度 16、特征维度 128，五类模型参数量约 257k。该骨干在此前六类实验中是有效的多尺度延迟融合基线，本轮不把结构变化与长尾变化混在一起。

## 3. 损失函数、训练和评估协议

四个臂分别为：

| 臂 | 训练样本 | 损失 |
|---|---|---|
| B0-Focal-Balanced | SawFM 过滤后的全部五类训练池 | Focal，`gamma_pos=0, gamma_neg=3` |
| B0-Focal-LabelIF4 | LabelIF4 manifest | 同上 |
| B0-DB-LabelIF4 | LabelIF4 manifest | Distribution-Balanced Loss，训练集实际计数 |
| B0-DB-Balanced | 全部五类训练池 | DB |

LabelIF4 的目标计数为 `[4000, 3000, 2000, 1500, 1000]`，实际计数完全达到，头尾比 4:1。原始 0716 组合表高度对称，直接强行 IF20 会使组合整数约束不可行，因此本轮采用可达的 IF4，并记录这一限制。

优化器为 AdamW；backbone 学习率 `5e-5`，分类头学习率 `5e-4`，weight decay `1e-4`，梯度裁剪 `5.0`；batch 64，最大 12 epoch，early patience 4，early/test batch 256。所有臂每 epoch 固定 9,920 次抽样，长尾臂有放回采样，消除“长尾样本更少导致优化步数更少”的混杂。模型选择只看 early macro-AUPR；test 只在 checkpoint 确定后评估，阈值固定 0.5，不在 test 上调阈值。

## 4. 等预算正式结果

下表为 3 个 seed 的 test 平均值 ± 标准差（macro-AUPR）：

| 方法 | macro-AUPR | macro-AUROC | macro-F1@0.5 |
|---|---:|---:|---:|
| B0-Focal-Balanced | 0.974580 ± 0.000192 | 0.96699 | 0.90804 |
| B0-DB-Balanced | 0.973291 ± 0.000728 | 0.96536 | 0.90907 |
| B0-Focal-LabelIF4 | 0.908731 ± 0.008014 | 0.89099 | 0.81122 |
| B0-DB-LabelIF4 | 0.913237 ± 0.010672 | 0.89229 | 0.81231 |

DB 相对 Focal 的 LabelIF4 macro-AUPR 增益为：

- seed 271001：`-0.000950`
- seed 271002：`+0.010387`
- seed 271003：`+0.004080`
- 平均：`+0.004506`，2/3 seed 同向

LabelIF4 下各类平均 AUPR：

| 类别 | Focal-LabelIF4 | DB-LabelIF4 | DB−Focal |
|---|---:|---:|---:|
| FMB | 0.766823 | 0.789562 | +0.022739 |
| SinFM | 0.988014 | 0.985169 | −0.002845 |
| MFT | 0.944331 | 0.944861 | +0.000530 |
| VFT | 0.916887 | 0.913138 | −0.003749 |
| ISRJ | 0.927599 | 0.933454 | +0.005855 |

若把 VFT/ISRJ 作为当前最低频次尾部，DB 的尾部平均增益只有约 `+0.0011`。因此没有达到预注册候选门槛“macro-AUPR ≥ +0.010、尾部平均 ≥ +0.020、至少 2/3 seed 同向且头部不下降”。结论是：`[N] B0-DB-LabelIF4 未证明稳定解决五类频次长尾`。

## 5. 失败与不理想结果的原因

### 5.1 长尾训练池改变了组合覆盖，而不只是改变标签计数

为了达到 `[4000,3000,2000,1500,1000]`，组合级整数约束会删除大量共现组合。FMB 的 AUPR 从平衡控制约 0.997 降到 Focal-LabelIF4 的 0.767，说明模型遇到的是“标签频次 + 共现结构改变”的复合分布偏移，而非纯粹的单标签先验变化。DB 可以提高 FMB，却同时牺牲 SinFM/VFT，正是共享表示和共现梯度重新分配的表现。

### 5.2 DB 只改变损失权重，不能补回被长尾采样删除的组合证据

Distribution-Balanced Loss 对训练计数建模，但没有恢复缺失的组合、局部时间证据和类别条件特征。当前 B0 的共享 128 维向量和共享线性头仍然让高频/易分离类别主导表示；因此 DB 增益小且 seed 方差较大。

### 5.3 不能把 pilot 的正增益当成正式证据

未等预算 pilot 中长尾臂每 epoch 只有 3,600 个样本，平衡臂有 9,920 个样本，DB 的表面增益受优化步数混杂影响。等预算重跑后平均增益从 pilot 的约 `+0.0013`（且方差较大）变为 `+0.0045`，但仍未过门槛。pilot 已保留在 `outputs/route_a_five_class_frequency_lt_v1`，不得与正式结果合并。

### 5.4 当前有效骨干并非主要失败点

平衡控制 macro-AUPR 约 0.975，说明 B0 对五类任务具有充分的基础特征提取能力。当前主要瓶颈是长尾训练分布、共现覆盖和分类决策边界，而不是继续扩宽 backbone。

## 6. 目前阶段判断

- `[F]` 已验证：SawFM 阳性样本可被审计地从训练/测试剔除；B0 可稳定完成五类平衡识别；LabelIF4 可复现地构造并达到目标计数；等预算训练协议已实现。
- `[A]` 当前证据：五类频次长尾使 macro-AUPR 从约 0.975 降至约 0.909；DB 在 3 seed 中 2 个同向，但平均增益只有 0.0045。
- `[N]` 失败/停止：没有证明 DB 能稳定改善尾类，也没有证明“仅损失函数”足以解决频次长尾。
- `[T]` 尚未验证：采样均衡、logit adjustment、cRT、label-aware sampler、标签条件分类头。

## 7. 下一阶段改进与试验顺序

保持 B0 完全冻结，不扩宽网络，优先做以下等预算消融：

1. **全池 + label-aware sampler**：不再删除样本，只在 12,400 个 SawFM 过滤训练池上按标签正例覆盖进行有放回采样；与当前 LabelIF4 manifest 对比，区分“频次先验”与“组合覆盖损失”。
2. **logit adjustment**：在同一 B0-Focal-LabelIF4 特征上加入基于训练先验的 `logit - tau*log((1-p)/p)`，只在 early 选择 `tau∈{0.25,0.5,1.0}`，test 不调参。
3. **冻结特征 cRT**：用 B0-Focal-LabelIF4 训练的 backbone 提取特征，冻结 backbone，仅比较普通线性头、class-balanced BCE 头和 balanced cRT 头；这一步验证问题究竟在 backbone 还是分类器。
4. **联合损失消融**：在完全相同 sampler 下比较 Focal、DB、class-balanced BCE、ASL-DB 混合，最多一次改变一个因素。
5. 只有当同一方案在 5 seed 中至少 4 个同向、macro-AUPR 增益 ≥0.010、尾部两类平均增益 ≥0.020、头部下降 ≤0.010 时，才进入标签条件分类头或最后 stage 的低学习率微调。

下一轮仍不引入功率视图、不恢复 SawFM、不访问任何封存测试集。若上述方法均失败，应把论文创新点收窄为“B0 上的五类频次长尾诊断与分类器/采样校正”，不能宣称双轴长尾复合识别已被统一解决。
