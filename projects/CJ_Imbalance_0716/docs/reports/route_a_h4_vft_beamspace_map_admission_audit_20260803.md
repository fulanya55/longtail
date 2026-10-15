# Route A H4 准入审计：交错 PRI VFT 的全延迟-频率相干证据图

日期：2026-08-03  
状态：**仅完成文献与机制准入；未生成 H4 source，未训练，未访问任何测试角色。**

## 1. 触发原因

H0q/H1a/H1b 已在交错 PRI 多脉冲协议中证明固定真实时间戳相干量具有 VFT
物理可观测性；H2 与 H3 则两次否定了“把每个频率的 delay-max 相干统计压成
25 维向量，再以 25→16→1 加到 VFT logit”这一学习机制。H3 中 T1 相对同容量
置换 T2 的 weak AUROC 为 `-0.001365`，五个种子均为负。因此 H4 不能重命名
或调参 H3，必须改变**信息保留路径**。

## 2. 文献检查及证据等级

本轮只将可直接核验的书目元数据作为外部证据；未取得全文的技术细节不从标题
推断。

| 证据 | 与 H4 的关系 | 等级 |
| --- | --- | --- |
| Doerry, *Radar Doppler Processing with Nonuniform Sampling*, OSTI, 2017, DOI `10.2172/1373645` | 表明非均匀采样的雷达多普勒处理是已有物理问题；不支持任何网络细节。 | [A] |
| Harnett et al., *Experimental Evaluation of Adaptive Doppler Estimation for PRI-Staggered Radar*, IEEE RadarConf 2022, DOI `10.1109/radarconf2248738.2022.9764326` | 表明 PRI-staggered 多普勒估计已有实验研究；不支持复合干扰多标签学习结论。 | [A] |
| Doerry, *Radar Doppler processing with nonuniform PRF*, SPIE 2018, DOI `10.1117/12.2303453` | 进一步确认非均匀 PRF/PRI 处理不是新颖性来源本身。 | [A] |
| Wang et al., *Automatic radar waveform recognition based on time-frequency analysis and convolutional neural network*, ICASSP 2017, DOI `10.1109/icassp.2017.7952594` | 时间频率表示接入 CNN 是既有范式；不能声称“H4 使用二维卷积”本身创新。 | [A] |
| H0q/H1a/H1b/H2/H3 的项目可审计结果 | 在本任务里，真时间戳物理量可辨识，但 delay-max 向量的加性投影不胜过置换容量对照。 | [F] |

因此可主张的潜在贡献不是“交错 PRI”“多普勒图”或“CNN”，而只能是：在
多标签复合干扰中，以**物理锚定的全局安全头 + 目标专属、保留延迟结构的相干
证据图**，通过真/置换时间轴的信息和容量匹配消融，检验弱 VFT 是否真正获得
可归因的局部证据。该表述仍是待验证假设，不是成果。

## 3. 与 H3 的非重叠性审查

H3 的输入为

`z_f = max_d |sum_p C[p,d] exp(-j 2π f t_p)| / sum_p |C[p,d]|`，

其中 `f` 有 25 个固定频点；每个频点在输入前已经对 401 个 delay 位置取最大值。
H3 因而丢失了证据的 delay 位置、延迟带宽、跨频连续性和局部相干形状。其失败
禁止改时间戳、ROI、宽度、损失等，但不等于已经检验过一个不做 delay-max 压缩的
二维证据图。

H4 唯一拟议的新对象为固定的复相干波束空间图

`B[f,d] = sum_p C[p,d] exp(-j 2π f t_p) / (sum_p |C[p,d]| + eps)`，

保留 `f=1000:250:7000 Hz` 和 `d=100:500` 两条已物理冻结的坐标；输入为
`[Re(B), Im(B), |B|]` 的 `3×25×401` 张量。对照图使用 H3 已冻结的同一时间戳
置换。它不是一个更宽的 H3 MLP，也不允许修改 H3 的坐标或数据；其必要性来自
H3 先行的 delay-max 信息损失诊断，而非事后性能搜索。

## 4. H4 的最小可证伪路径（尚未执行）

### P0：零拟合图形物理诊断

在新的三臂、identity-isolated 640 组 source 中，固定计算真时间戳与置换时间戳
的 `B[f,d]`。不训练网络，预先定义延迟聚集比：先在每个频率保留归一化模值，再
对频率求均值；以固定 5 点局部核的最大质量除以全 delay 质量。P0 比较 weak VFT
与 absent，门槛为：真时间戳 AUROC `>=0.60`、真减置换 `>=+0.060`、paired
bootstrap 95% CI 下界 `>0`。若任一失败，停止 H4，且不实现图模型。

这不是在 H3 source 上追加分析；H4 的 `pair_seed`、receiver identity、噪声和
背景子流必须全部新建。每组仍共享所有非 VFT 成分、噪声和 VFT 参数，仅改变
VFT absent/weak/strong 三臂。

### P1：同容量真/置换图编码器

仅当 P0 通过，训练下列固定臂：

| 臂 | VFT 路径 | 允许改动 |
| --- | --- | --- |
| M0 | 与 H3 相同的多脉冲 global backbone | 无 VFT 图支路 |
| M1 | `B_true` 的小型二维深度可分证据编码器 | 只修改 VFT logit |
| M2 | `B_permuted` 的同结构、同参数编码器 | 只修改 VFT logit |

图编码器应在 P0 前冻结：`3→12` 的 `3×7` Conv2d，随后两层 depthwise-separable
`3×5` block，频率轴平均、delay 轴 attention-free log-sum-exp 汇集，`12→16→1`
VFT 残差头。M1/M2 必须同初始化策略、同参数数、同优化器和同训练 seeds；global
head、SawFM 路径及非 VFT logits 是安全路径。它保留局部 delay 证据，但不使用
分类器 query、样本对 logit-margin、阈值或校准器，避免回到已停止机制。

P1 的五种子主门槛与 H3 同样严格：M1−M2 weak AUROC 均值 `>=+0.030`、5/5 正、
paired bootstrap 下界 `>0`；weak AUPR 均值 `>=+0.025` 且下界 `>0`；并要求
M1−M0 strong AUROC 均值 `>=-0.010`、逐 seed `>=-0.020`。同时记录 M1/M2
每张图的输入 hash 和中间图摘要，证明控制只改变时间戳顺序。

## 5. 当前决策

H4 **可以进入 P0 的实现与 source-contract 审查，但尚未获准训练 P1**。P0 的
实现必须先通过：张量 shape/有限性测试、真/置换仅时间轴不同的单元测试、三臂
反事实身份审计、历史/H0q/H1a/H1b/H2/H3 identity 零重叠审计，以及 test-v3 seal
审计。任何一项不通过都只修复工程合同，不以 P0/H4 指标选择为导向。

即使 P0 与 P1 都通过，H4 仍仅是新观测协议的 VFT 组件证据：还需新六标签统一
方案、强基线、消融、鲁棒性与完全冻结后的一次独立测试，才可能接近总体目标。
