# Route A H0e：反事实可见性边际（CVMC）可识别性门

## 单一可证伪假设

已有 VTSE 的静态弱正样本重加权、固定形态特征 pair-ranking、功率不变性、
token query 和 ISRJ 相干复制图均已停止。H0e 不再改变推理期结构或输入：
**若训练时让同一背景、同一接收噪声下“仅目标分量存在/缺失”的成对记录
承担有界 logit 边际，则单视图网络应比相同计算量的随机配对边际控制更好地
排序 MFT、VFT、ISRJ 的弱正样本与同组合缺失目标的负样本。**

这检验训练期的反事实可见性监督，而不是另一个阈值、残差、特征拼接、
固定 morphology pair-ranking 或高低功率不变性方案。推理时只输入单条 I/Q。

## 新鲜受控数据

- 基础 seed `740000000`；2,400 个唯一 **pair identity**，每对的正、负两条
  view 共享 `sample_seed`、`receiver_noise_seed`、参考波形、保留分量和其参数；
  因而接收噪声及非目标组件逐点一致。
- 三个 target 各 800 对：MFT、VFT、ISRJ。每对负 view 是预定义 nuisance
  组合，正 view 仅额外加入该 target；每 target 有五种固定 nuisance 组合，
  包括 VFT/ISRJ 互为 hard nuisance 的情形。
- 所有组件基础 JNR 固定为 20 dB；正 view target 的 component offset 固定为
  -12 dB，其他组件为 0 dB。因此目标相对可见度严格为 -12 dB，是固定的
  10 dB weak positive；该元数据仅用于生成/审计/分层，绝不进入网络输入。
- pair identities 按 `receiver_noise_seed mod 10` 分为 fit/early/evaluation
  `1440/480/480` 对（各为 `2880/960/960` 单视图记录）。同一 pair 不得跨角色。
  数据只用于 H0e，不生成或读取任何 test 文件，并且须对所有历史训练 identity
  审计为零交集。

## 模型、损失和严格控制

同一 `CounterfactualVisibilityProbe`（原始 I/Q、三层 1-D 卷积、mean/max
pooling、六 logits）用于三臂，参数量完全相同。模型没有 pair id、功率、weak
mask 或参数输入。

每个 minibatch 的单视图目标 BCE 为

`L_BCE = BCEWithLogits(z, y)`。

对 MFT/VFT/ISRJ 的 fit pairs，令 `t` 是该 pair 切换的目标，固定边际损失为

`L_margin = softplus(0.5 - (z_positive,t - z_negative,t))`。

三臂固定为：

1. `B0`: `L_BCE`，无边际项；
2. `B1_shuffle`: `L_BCE + 0.5 L_margin`，但每个 epoch 将正/负 pair 在同一
   target、同一角色内以预声明的确定性排列重配；计算、参数、标签和边际项都
   相同，缺少“共享 nuisance”的信息；
3. `C_CVMC`: `L_BCE + 0.5 L_margin`，使用真实的同 background pair。

优化器为 AdamW（lr `3e-4`、weight decay `1e-4`）、batch 48 pairs、20 epoch；
早停仅按 early role 的三 target BCE，五个优化 seed 固定为 2039--2043。
每个 seed 的三臂从同一随机初始状态开始；没有校准或阈值选择。

## 主要读数与门槛

evaluation 仅报告单视图 logits。每个 target 的主要集合是所有弱正 view 与
同 nuisance-combination 的 target-absent view；主指标为 weak-vs-matched-negative
AUROC。还报告 AUPR、所有正例 AUROC、paired logit-margin success rate 和每 seed。

`C_CVMC` 相对 **B0 和 B1_shuffle 两者**均须满足：

- 三目标平均 weak AUROC 提升 `>= +0.015`，配对 bootstrap 95% CI 下界 `>0`，
  且至少 4/5 seed 为正；
- 每一 target 的均值变化 `>=0`，且任一 target 的 all-positive AUROC 不低于
  `-0.010`；
- evaluation paired margin-success 相对 B1 增加 `>=+0.03`，且 4/5 seed 为正；
- 参数、初始化、角色、训练预算、数据审计和 test-v3 封存全部一致。

通过只允许在另一份全新身份源中做六标签 CVMC 集成设计；不允许直接融合、
校准或访问 test-v3。任一门槛失败即停止 CVMC，禁止在 H0e 上调 lambda、margin、
卷积宽度、nuisance 组合或 weak offset。
