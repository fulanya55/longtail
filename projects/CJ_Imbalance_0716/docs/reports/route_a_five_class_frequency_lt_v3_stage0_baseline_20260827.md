# 路线 A：五类频次长尾 V3——组合保持与 B0 基线

日期：2026-08-27

## 当前状态

本阶段完成了新的身份隔离数据协议和 B0/Focal development 基线。旧的
`0716 closedset_6class_test.mat` 未打开、未用于选择，也未用于确认。损失函数
筛查和冻结分类头比较暂未启动，需在本基线收据审计后继续。

## 数据协议

- development 源：`E:\LJQ_code\干扰数据\复合干扰数据0731_v3\closedset_6class_train.mat`
- confirmation 源：`E:\LJQ_code\干扰数据\复合干扰数据0731_v2\closedset_6class_train.mat`
- 两个源均只使用 train MAT，互不共享 numeric index；SawFM 阳性样本先过滤。
- 五类顺序：`FMB / SinFM / MFT / VFT / ISRJ`。
- 使用整数规划在组合级求解目标边际 `[4000,3000,2000,1500,1000]`。
- development 和 confirmation 均精确达到目标边际，且非空五类组合 `31/31` 全部保留。
- 不使用功率长尾、SawFM 专家、半监督或旧 test 文件。

数据与哈希收据：
`E:\LJQ_code\干扰数据\route_a_five_class_frequency_lt_v3\protocol_audit.json`。

## B0 与训练设置

沿用 `DelayedFusionMultiScaleIQClassifier (dfms_iq)`，width=16，feature_dim=128，
dropout=0.2，五类线性头，参数量约 257k。输入为 `[B,2,3000]`，包含复数 RMS
归一化、四级多尺度时序残差、延迟 Cartesian/幅度/相位融合以及 mean/std 池化。

训练：AdamW；backbone lr=`5e-5`；classifier lr=`5e-4`；weight_decay=`1e-4`；
gradient clip=`5.0`；batch=`64`；每 epoch `9,920` draws；最多 18 epoch，patience=6；
5 个预注册 seed `272001–272005`。损失为 Focal，采样为 Uniform。

## development 基线结果

| seed | early macro-AUPR | early tail mean-AUPR | best epoch |
|---:|---:|---:|---:|
| 272001 | 0.975257 | 0.956910 | 18 |
| 272002 | 0.975242 | 0.956783 | 17 |
| 272003 | 0.974788 | 0.956730 | 17 |
| 272004 | 0.977149 | 0.957834 | 18 |
| 272005 | 0.976281 | 0.959330 | 16 |
| 均值 ± 样本标准差 | **0.975743 ± 0.000957** | **0.957517 ± 0.001109** | — |

该结果与上一轮完整五类 B0 约 0.976 的水平一致，说明新 `0731_v3` source 上的
B0 仍然能够稳定提取五类特征；当前没有理由因为 backbone 容量不足而扩宽网络。
这个结论只针对完整组合覆盖的 development，不等价于已证明弱证据或所有长尾
组合都已解决。

## 预注册后续顺序

1. 先在相同新 development 角色上比较 label-aware sampler 与 Uniform，保持 B0、
   优化器、预算和 seed 不变。
2. sampler 门槛通过后，才比较 Focal、DB、CB-BCE、ASL-DB；否则直接记录负结果。
3. 再进行 logit adjustment 和冻结 B0 分类头比较。
4. 只有满足 `macro-AUPR +0.010`、尾部两类平均 `+0.020`、至少 `4/5` seed 同向、
   头部下降不超过 `0.010` 且 bootstrap 下界为正，才允许进入标签条件头或
   Stage4 adapter。
5. 全部开发选择完成后，模型和阈值冻结，只对 `0731_v2` confirmation source
   执行一次正式确认；confirmation 在此之前不得用于筛选。

当前可执行脚本：
`E:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_five_class_frequency_lt_v3.py`

运行收据：
`E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_five_class_frequency_lt_v3\development_run_receipt.json`。
