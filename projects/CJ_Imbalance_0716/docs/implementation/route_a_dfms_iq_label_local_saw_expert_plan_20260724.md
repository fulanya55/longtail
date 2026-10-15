# 路线 A：DFMS-IQ 标签条件局部证据与早期 SawFM 专家试验方案（2026-07-24）

## 1. 目标

当前 F2 主干已经能提取较强的通用复数、多尺度和相位特征，冻结 pos-weight BCE 头 H2 达到 Macro-AUPR 0.8670、弱正—负 AUROC 0.6524。但六个标签仍共享一个全局 128 维向量，mean/std 不保留弱证据位置，且上一轮“晚期去相位”SawFM 专家与主路高度相关。

本轮不重做 F2 主干，而是在冻结 F2 和冻结 H2 基础头上逐级增加三个残差模块：

1. 标签条件时间池化；
2. 标签条件 top-k 局部证据旁路；
3. 从 stage2 提前分叉的独立 SawFM 包络/瞬时频率专家。

每个模块从零输出附近开始，最终 logit 始终保留 H2 安全基线。只有完整结构通过预注册门槛，才允许考虑解冻 F2 的末级时序层。

本轮仍为 development 结构筛选，`locked_test_used=false`。

## 2. 固定输入和冻结部分

- 数据：`C:\LJQ_code\干扰数据\复合干扰数据0716\closedset_6class_train.mat`
- 环境：`D:\anaconda3\envs\PY10\python.exe`
- 协议：六类 label-LT，2400 样本，IF=4，SawFM锚点400
- seeds：2041、2042、2043
- 主干：对应 seed 的 `F2_DFMS_Full/best_model.pth`
- 基础头：对应 seed 的 `H2_PosBCE/best_head.pth`
- F2 和 H2 基础头均完全冻结并保持 eval
- 训练、calibration、threshold-selection、early-stop 角色复用原 split SHA256 和 manifest SHA256
- 弱分量定义：标签存在且分量功率不高于10 dB

冻结 F2 提供三个接口：

```text
global feature:  [B,128]
temporal map:    [B,128,375]
stage2 complex:  real/imag [B,32,1500]
```

## 3. 消融矩阵

| 编号 | H2安全基线 | 标签条件池化 | top-k局部旁路 | stage2 Saw专家 |
|---|---|---|---|---|
| C0 | 冻结 | 无 | 无 | 无 |
| Q1 | 冻结 | 有 | 无 | 无 |
| Q2 | 冻结 | 有 | 有 | 无 |
| Q3 | 冻结 | 有 | 有 | 有 |

C0 直接复用上一轮 H2 正式结果。Q1—Q3 只训练新增残差模块，不修改 F2 和 H2 参数。

## 4. 标签条件时间池化 Q1

输入为 F2 融合后的时序图 `[B,128,375]`。

```text
key projection:   Conv1d(128,64,1)
queries:          6 × 64 learnable vectors
scores:           query · key / sqrt(64)
attention floor:  0.20 uniform + 0.80 softmax(scores)
weighted stats:   每标签 weighted mean + weighted std = 256维
projection:       Linear(256,64)-LayerNorm-GELU-Dropout(0.1)
query logit:      每标签 Linear(64,1)
```

输出：

```text
logit_Q1[c] = frozen_H2_logit[c]
              + tanh(query_scale[c]) × query_logit[c]
query_scale 初始值 = 0.05
```

attention floor 保证任何标签都至少保留20%的均匀全局证据，避免注意力再次完全压制弱区域。

## 5. 局部 top-k 证据旁路 Q2

使用 Q1 的标签条件 scores，在每个标签上选择 `k=8` 个最高分时间位置。从原始 `[B,128,375]` 时序图提取：

- top-k位置均值：128维；
- top-k位置最大值：128维；
- 拼接后投影：`Linear(256,64)-LayerNorm-GELU-Dropout(0.1)`；
- 每标签独立线性 logit。

```text
logit_Q2[c] = logit_Q1[c]
              + tanh(local_scale[c]) × local_logit[c]
local_scale 初始值 = 0.05
```

Q2 初始化自对应 seed 的 Q1 最优检查点；训练 Q2 时冻结 Q1，只训练 local projection、local classifier 和 local scale。这样 Q2-Q1 可直接归因于局部旁路。

## 6. stage2 独立 SawFM 专家 Q3

专家从 F2 block2 后、stage3 前分叉，输入冻结 stage2 复数特征 `[B,32,1500]`，不复用主路的 late fusion、temporal block、global pool 或 feature projection。

构造六组稳定描述量：

1. `magnitude`；
2. `abs(envelope difference)`；
3. 相邻相位差 cosine；
4. 相邻相位差 sine；
5. `abs(delta phase-cosine)`；
6. `abs(delta phase-sine)`。

得到 `[B,192,1500]` 后进入：

```text
Conv1d(192,32,1)-BN-SiLU
三路depthwise多尺度:
  k=5,d=1
  k=15,d=2
  k=31,d=4
concat → Conv1d(96,32,1)-BN-SiLU
AvgPool1d(4) → 长度375
独立统计池化: mean + std + max + top8-mean = 128维
MLP: Linear(128,64)-GELU-Dropout(0.1)-Linear(64,1)
```

只修改 SawFM：

```text
SawFM_logit_Q3 = SawFM_logit_Q2
                 + tanh(saw_scale) × saw_expert_logit
saw_scale 初始值 = 0.05
```

Q3 初始化自 Q2；训练时冻结 Q1/Q2，只训练 SawFM 专家和 saw scale，保证 Q3-Q2 是专家独立贡献。

## 7. 训练与选择

| 参数 | Q1 | Q2 | Q3 |
|---|---:|---:|---:|
| loss | pos-weight BCE | pos-weight BCE | pos-weight BCE |
| optimizer | AdamW | AdamW | AdamW |
| learning rate | 1e-3 | 1e-3 | 1e-3 |
| weight decay | 1e-4 | 1e-4 | 1e-4 |
| batch size | 64 | 64 | 32 |
| epochs | 40 | 30 | 30 |
| patience | 8 | 8 | 8 |
| gradient clip | 5 | 5 | 5 |
| checkpoint selection | head-validation Macro-AUPR，弱AUROC同分决胜 | 同左 | 同左 |

训练清单按 `57000+seed` 确定性80%/20%拆分；特征映射只从冻结 F2 提取，不读取锁定测试集。

## 8. 校准、阈值和指标

- 每个变体在独立 calibration 角色拟合标量温度；
- threshold-selection 使用上一轮相同规则：逐标签最大化弱正召回，negative FPR≤0.20；
- 同时保存 fixed-0.5 和 per-label-F1；
- early-stop development 角色报告：Macro-AUPR、Macro-F1、Hamming、弱召回、弱正—负 AUROC/AUPR、总体/逐标签 FPR、SawFM-AUPR、SawFM弱召回和SawFM弱正—负 AUROC；
- 保存每标签 attention entropy、top-k覆盖和各残差 scale，验证模块是否实际被使用。

## 9. 晋级门槛

### 9.1 Q1 标签条件池化相对 C0

- Macro-AUPR ≥ -0.003；
- 弱正—负 AUROC ≥ +0.010；
- 弱分量召回 ≥ +0.020；
- Hamming ≤ +0.005；
- negative FPR增量≤0.020；
- 弱AUROC和弱召回至少2/3 seeds为正。

### 9.2 Q2 局部旁路相对 Q1

- Macro-AUPR ≥ -0.003；
- 弱正—负 AUROC不下降；
- 弱分量召回 ≥ +0.020；
- negative FPR增量≤0.010；
- 至少2/3 seeds弱召回为正。

### 9.3 Q3 SawFM专家相对 Q2

- Macro-AUPR ≥ -0.003；
- SawFM-AUPR不下降；
- SawFM弱正—负 AUROC ≥ +0.020；
- SawFM弱召回 ≥ +0.030；
- SawFM negative FPR增量≤0.020；
- SawFM弱AUROC至少2/3 seeds为正。

### 9.4 完整结构 Q3 相对 C0

- Macro-AUPR ≥ -0.005；
- 总体弱正—负 AUROC ≥ +0.015；
- 总体弱召回 ≥ +0.030；
- Hamming ≤ +0.005；
- 总体 negative FPR≤0.25且增量≤0.02；
- SawFM-AUPR不下降、SawFM弱AUROC≥+0.020；
- 总体弱AUROC和弱召回至少2/3 seeds为正。

只有9.4完整通过，才允许下一轮解冻 stage4/temporal backbone。若任一子模块未通过，则保留已通过的上一级，不通过的模块停止。

## 10. 输出

```text
outputs/tapc/route_a_dfms_iq_label_local_saw/
docs/results/route_a_dfms_iq_label_local_saw_results_20260724.md
```

每个变体保存训练历史、模块检查点、源F2/H2哈希、split/manifest哈希、温度、阈值、预测、模块诊断、完整指标和 `locked_test_used=false` 回执。

## 11. 执行状态（2026-07-24）

- GPU smoke：1 seed × C0/Q1/Q2/Q3完成；
- 正式结构筛选：3 seeds × C0/Q1/Q2/Q3完成；
- 新增模块检查点：9个；
- Q1、Q2、Q3及完整结构门槛均未通过；
- 按预注册规则未解冻stage4或temporal backbone；
- locked test未使用；
- 完整结果见 `docs/results/route_a_dfms_iq_label_local_saw_results_20260724.md`。
