# 路线 A：五类频次长尾 V3——Stage 1 sampler 验证

日期：2026-08-27

## 结论

在新的组合保持数据协议上，MaxPositive-Focal 和 MeanPositive-Focal 相对同 seed
Uniform-Focal 的增益均约 `1e-4`，远低于预注册门槛。因此 sampler 未通过，暂不
进入新的损失函数筛查、标签条件分类头或 Stage4 adapter。旧 test 和新 confirmation
均未使用。

## 固定条件

- development：`E:\LJQ_code\干扰数据\复合干扰数据0731_v3\closedset_6class_train.mat`
- 五类顺序：`FMB / SinFM / MFT / VFT / ISRJ`，SawFM 已过滤。
- 五类长尾目标：`[4000,3000,2000,1500,1000]`。
- 31 个非空组合全部保留；组合级数量由整数规划确定。
- B0：`DelayedFusionMultiScaleIQClassifier`，width=16，feature_dim=128，dropout=0.2。
- 损失：Focal；优化器 AdamW；backbone lr=`5e-5`；head lr=`5e-4`；weight_decay=`1e-4`；
  batch=64；9,920 draws/epoch；18 epoch/patience=6；seed=`272001–272005`。
- Uniform 基线使用上一阶段相同 seed、相同角色划分和相同训练预算。

## 配对 development 结果

| sampler | seed 增益（macro-AUPR） | 尾部增益均值 | 正向 seed | 最差头部类 |
|---|---:|---:|---:|---:|
| MaxPositive-Focal | **+0.000080** | **+0.000234** | 4/5 | -0.000690 |
| MeanPositive-Focal | **+0.000110** | **+0.000230** | 3/5 | -0.000513 |

逐 seed macro-AUPR 增益：

```text
MaxPositive:  +0.0000888, +0.0001270, +0.0000550, +0.0001575, -0.0000283
MeanPositive:+0.0002032, +0.0002910, -0.0000160, +0.0000953, -0.0000238
```

预注册硬门槛为：

```text
macro-AUPR 增益 ≥ 0.010
尾部两类平均增益 ≥ 0.020
至少 4/5 seed 同向
头部类别下降不超过 0.010
```

两个 sampler 虽然头部安全且 MaxPositive 达到 4/5 同向，但宏平均和尾部增益
均不接近门槛，故判定为负结果。

## 归因

这次结果与上一轮不同：新 manifest 已经保留全部 31 个组合，因此 sampler 失败
不再能解释为“组合被删除”。更合理的解释是：

1. 当前 sampler 只改变样本抽取频率，没有改变每个组合内的有效信号证据；
2. 五类长尾是由组合计数变化产生的，标签边际权重不能完全等价于组合级重采样；
3. B0 在完整池和新长尾 development 上都稳定，sampler 不是恢复弱尾召回的充分机制。

## 产物

- 脚本：`E:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_five_class_frequency_lt_v3_stage1_sampler.py`
- 汇总：`E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_five_class_frequency_lt_v3\stage1_sampler\stage1_sampler_summary.json`
- 收据：`E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_five_class_frequency_lt_v3\stage1_sampler_receipt.json`

## 下一步

按照预注册规则，sampler 未通过后仍可进行诊断性损失比较，但不能将其声明为
正式候选。下一阶段将固定 Uniform sampler，在相同新 development 上比较 Focal、
DB、CB-BCE 和 ASL-DB；若损失也未达到门槛，则停在分类器/结构扩展之前。
