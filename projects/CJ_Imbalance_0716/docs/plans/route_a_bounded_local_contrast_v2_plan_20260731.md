# Route A：有界标签局部对比证据 development-v2 预注册方案

## 研究问题

H5 已证明删除 SawFM 输出竞争只能使 MFT/VFT/ISRJ 弱召回提高 0.0084，分类头
竞争不是主因。本轮检验：F2 池化前 `[B,128,T]` 特征中是否仍包含可利用的
标签局部弱证据，以及显式弱正—难负排序能否在不破坏全局基线的条件下提取它。

## 冻结部分与对照

- F2_DFMS_Full 五个 checkpoint、C6_PosBCE 头及其特征标准化参数全部冻结。
- 数据角色、2400 条 label-LT manifest、校准、阈值和 early diagnostic 角色与
  H5 development-v2 实验逐种子完全一致。
- B0：冻结 C6 原始输出，不训练新参数。
- L1：有界局部对比读出器，使用 pos-weight BCE。
- L2：与 L1 同初始化，加入弱正—难负排序及强正残差锚定。

## L1/L2 网络

1. 输入：F2 temporal map `[B,128,T]`。
2. `Conv1d(128,32,1,bias=False) + GroupNorm(8,32) + SiLU`。
3. 六个单位范数标签方向生成六张热图。
4. 每张热图减去自身时间均值；网络不能通过常数热图表达类别先验。
5. 每标签提取 top-8 mean、top-8 max、top-8 std、全图 std。
6. 每标签四维无 bias 线性组合产生 raw residual。
7. `residual = 0.35*tanh(raw)`，最终 `logit = frozen_C6_logit + residual`。

旧 Q1/Q2 的主要失败是自由残差学习 label-LT 类别偏置且幅度失控。本设计通过
时间零中心、删除 bias、硬幅度上限和冻结全局安全旁路逐项消除该机制。

## 损失与训练

- L1：`pos_weight_BCE + 0.05*residual_L2 + 0.20*per_label_residual_mean^2`。
- L2：L1 损失 `+0.50*weak_positive_hard_negative_rank`
  `+0.10*strong_positive_residual_L2`。
- 排序 margin=0.20；每标签以 batch 内弱正和最高分负样本配对。
- AdamW，lr=3e-4，weight decay=1e-4，batch=64，最多 40 epoch，patience=10，
  gradient clip=5.0。
- L1/L2 每种子使用相同参数初始化；差异只来自损失机制。
- 早停：五标签 macro-AUPR；并列时 MFT/VFT/ISRJ 弱正—负 AUROC。

## 校准、阈值与证据边界

- calibration role：标量温度缩放。
- threshold-selection role：逐标签 FPR<=0.20 下最大化弱召回。
- early role：仅作 development-v2 diagnostic。
- test-v2 与历史 locked test 均禁止读取；机器结果必须记录
  `test_v2_used=false`、`locked_test_used=false`。

## L2 相对 B0 的晋级门

必须全部满足：

1. 五标签 macro-AUPR 差值 >= -0.005；
2. 五标签 macro-F1 差值 >= +0.010；
3. MFT/VFT/ISRJ 平均弱召回差值 >= +0.030；
4. 三目标弱正—负 AUROC 差值 >= +0.015；
5. 三目标平均 development FPR <= 0.20；
6. 至少 4/5 种子弱召回为正增益；
7. SawFM-present macro-F1 不下降超过 0.005；
8. early role 每标签平均残差绝对值 <= 0.10，且最大绝对残差 <= 0.35。

通过后才允许以 0.1 倍学习率解冻 temporal backbone；失败则说明 stage4/temporal
特征本身的弱证据不足，下一阶段必须提前到 stage2/stage3 做独立多尺度专家。

