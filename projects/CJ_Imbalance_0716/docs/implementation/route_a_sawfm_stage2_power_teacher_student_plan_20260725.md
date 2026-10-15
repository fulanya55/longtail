# 路线A：SawFM stage2同身份功率Teacher–Student稳定化方案（2026-07-25）

## 1. 试验动机

上一轮冻结特征充分性试验表明：

- H2仍是总体最稳定的弱证据读取器，弱正—负样本AUROC为0.6524；
- stage2物理统计只在SawFM上出现额外信息，SawFM弱AUROC从H2的0.5203提高到0.5586；
- 该增量跨seed不稳定，并存在约0.33的内部validation到独立development泛化落差；
- 硬约束残差消除了logit漂移，但没有改善弱证据排序。

因此本阶段不扩宽主干，只验证一个可证伪假设：**SawFM的stage2证据已部分存在，但需要同身份跨功率监督，才能使高可见度证据在弱视图上保持稳定。**

## 2. 数据协议修正

### 2.1 为什么不能直接使用label-LT清单构造SawFM功率对

每个seed的label-LT清单包含2,400个样本、400个SawFM阳性，但其中绝大多数SawFM只与FMB共现或单独出现。审计结果：只有1个SawFM阳性同时包含MFT/VFT/ISRJ，因此几乎无法通过改变扫频分量功率制造SawFM弱化。

本阶段保留原F2/H2模型和开发角色，但辅助功率训练身份从同一seed的合法`train_indices`中重新分层抽取，不读取early/calibration/threshold或locked test。

### 2.2 训练身份

每个seed的合法训练角色有17,640个身份：

- 合格SawFM阳性：`SawFM=1 AND (MFT OR VFT OR ISRJ)=1`，共7,840个；
- hard negative：`SawFM=0`，且去掉SawFM后的其他五标签组合与阳性完全一致，共7,840个；
- 去掉SawFM后共有28种组合。

按28种组合等额抽取：

```text
每组合40个SawFM阳性 + 40个匹配hard negative
= 1,120个阳性身份 + 1,120个负身份
```

训练身份按`receiver_noise_seed`映射到功率扫频文件，不把功率视图当成独立样本。抽取后以身份为单位固定80% fit / 20% validation。

另为随机负样本对照抽取1,120个SawFM缺失身份，不要求其他标签组合匹配。

## 3. 功率视图

0716功率扫频中FMB/SawFM/SinFM固定为30 dB，MFT/VFT/ISRJ取扫频功率：

- teacher：10 dB，SawFM相对扫频分量高20 dB；
- train students：40 dB、50 dB，SawFM分别相对扫频分量低10 dB、20 dB；
- held-out validation/development：45 dB，不参与训练损失。

validation使用训练身份的45 dB视图；独立development使用`early_stop_indices`中不同身份的45 dB视图。两者使用相同判别条件，直接计算泛化落差。

## 4. 模型结构

F2全部冻结，只读取stage2 complex map：

```text
stage2 real/imag: [B,32,1500]
AvgPool1d(4) -> concatenate(real,imag): [B,64,375]
```

新增`SawStage2LowRankEvidence`：

```text
input normalization: GroupNorm(8,64,affine=False)
low-rank residual adapter:
  Conv1d(64,8,1,bias=False)
  GELU
  Conv1d(8,64,1,bias=False), zero initialization
  residual scale = 0.20*tanh(alpha)

evidence head:
  Conv1d(64,32,1,bias=False)
  GroupNorm(8,32)
  SiLU
  depthwise Conv1d(32,32,k=15,padding=7)
  SiLU
  Conv1d(32,1,1,bias=False)

map normalization:
  map_z = (map - temporal_mean) / temporal_std
pooling:
  top16_mean + 0.25*max
classifier:
  learnable positive scale + bias -> SawFM evidence logit
```

只有adapter和证据头训练，F2、H2、stage1/stage2原参数均冻结。

## 5. 消融变体

| 变体 | Teacher | 负样本 | 跨功率一致性 | 匹配排序 |
|---|---|---|---|---|
| S0_StudentRandomNeg | 无 | 随机Saw缺失 | 无 | 无 |
| S1_TeacherRandomNeg | 10 dB | 随机Saw缺失 | score + map | 无 |
| S2_TeacherHardNeg | 10 dB | 相同其他标签组合 | score + map | 有 |

S0验证单纯增加专项头；S1归因teacher/student；S2归因组合匹配hard negative。

## 6. 损失函数

共同分类损失：

```text
L_student = BCE(student40, 1) + BCE(student50, 1)
L_negative = BCE(negative40, 0) + BCE(negative50, 0)
```

S1/S2增加：

```text
L_teacher = 0.5 * BCE(teacher10, 1)
L_score = 0.5 * SmoothL1(student_logit, stopgrad(teacher_logit))
L_map = 0.2 * (1 - cosine(student_map_z, stopgrad(teacher_map_z)))
```

S2增加：

```text
L_rank = 0.5 * softplus(0.5 - student_logit + matched_negative_logit)
```

总损失为对应项求和。正负样本数量相等，不再额外使用类别频次权重。

## 7. 训练参数

| 参数 | 数值 |
|---|---:|
| seeds | 2041, 2042, 2043 |
| optimizer | AdamW |
| learning rate | 3e-4 |
| weight decay | 1e-4 |
| batch size | 64 identity pairs |
| epochs | 30 |
| patience | 8 |
| gradient clip | 5.0 |
| adapter rank | 8 |
| top-k | 16 |
| checkpoint selection | validation@45dB AUROC，AUPR同分决胜 |

## 8. 评价指标和晋级门

核心指标：

- validation@45dB SawFM-vs-matched-negative AUROC/AUPR；
- independent development@45dB SawFM-vs-matched-negative AUROC/AUPR；
- `abs(validation AUROC - development AUROC)`；
- 三种子均值、标准差和逐seed方向；
- 相对上一轮stage2 Saw oracle 0.5586的增量。

S2完整方案必须同时满足：

1. independent development AUROC均值≥0.60；
2. 至少2/3 seeds的development AUROC≥0.60；
3. 平均绝对泛化落差≤0.15；
4. 相对S0平均AUROC提高≥0.02；
5. 至少2/3 seeds优于S0；
6. development AUPR不低于S0；
7. pair index、身份角色和locked-test审计全部通过。

只有S2通过上述门槛，才允许将证据分数标准化后以最大0.20的残差回接H2；否则停止在oracle证据头，不进行回接。

## 9. 输出

```text
outputs/tapc/route_a_sawfm_stage2_power_teacher_student/
docs/results/route_a_sawfm_stage2_power_teacher_student_results_20260725.md
```

保存抽样身份、组合计数、pair SHA256、split SHA256、F2/H2 SHA256、训练历史、检查点、逐seed预测、指标、泛化落差、晋级决定和`locked_test_used=false`回执。

## 10. 执行状态（2026-07-25）

- GPU smoke：完成，身份隔离和自动停止逻辑通过；
- 正式试验：3 seeds × S0/S1/S2完成；
- S2 oracle晋级门：通过；
- 最大0.20受限残差回接：按门槛执行完成；
- S2 development AUROC/AUPR：0.7972/0.7949；
- 受限回接AUROC增量：+0.00219，未形成有效最终融合；
- locked test未使用；
- 完整结果见`docs/results/route_a_sawfm_stage2_power_teacher_student_results_20260725.md`。
