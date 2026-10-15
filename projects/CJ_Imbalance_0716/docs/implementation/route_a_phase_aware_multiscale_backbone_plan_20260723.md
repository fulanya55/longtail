# 路线 A：相位保持多尺度复数特征网络设计（2026-07-23）

## 1. 状态与结论边界

本阶段已完成新特征提取网络的代码实现、结构测试、梯度测试和 2400 样本 Label-LT 的 1-epoch GPU 冒烟。新网络暂命名为 **PAM-IQ（Phase-Aware Multi-scale I/Q Encoder）**。

当前只证明：网络结构、复数输入、相位特征、多尺度模块、稀疏注意力、TAPC 训练与保存链路均可运行。1-epoch 冒烟结果不能用于宣称性能优于 Base-DB；正式晋级必须经过参数匹配对照、多种子筛选和独立确认。

## 2. 参考工程给出的直接证据

参考目录：`C:\LJQ_code\pythonProject\新建文件夹\复数输入接口对比_20260715`。

### 2.1 输入接口筛选

参考工程完成了 72 次开发集训练，综合排名为：

```text
iq_2d > complex_1d > iq_2ch_1d > concat_1d
```

其中 `complex_1d + tiny` 在 10% 和 50% 数据量下取得最高 subset accuracy；说明在样本量足够时，真正的复数乘法卷积具有潜力。`iq_2d` 综合排名第一，则说明实部和虚部的同时刻交互必须显式保留。

### 2.2 不能直接照搬的部分

- `mlcnn_light` 在各输入模式下持续低于 tiny，说明简单并联 `3/5/7` 卷积后立即池化并不能充分提取复合干扰特征；
- 参考工程的 ResNet34、MobileNetV2、ShuffleNetV2 完整基线仍将实部和虚部分别送入同一个实值网络，只在分类前拼接，不是真正的深层复数交互；
- 历史 EffNetV3 也将实部和虚部分开编码，参数量约 3M～4M，并且注意力与骨干容量同时变化，不适合直接作为本项目的因果对照；
- 当前路线 A 只有 2400 个受控 Label-LT 训练样本，直接迁移 4M 以上网络容易把容量、注意力和长尾优化混在一起。

因此，新网络只借鉴有效结构思想，不直接复制历史整网或旧权重。

## 3. 当前 Base-DB backbone 的不足

当前 `complex_1d_true` 的结构是：一次复卷积后立刻取复幅值，随后三个实值卷积块，全局平均为 64 维特征，再接线性六标签头。部署参数为 21,990。

主要问题：

1. 复相位只在第一层参与计算，后续相位演化被丢弃；
2. 最后位置特征的有效感受野较短，难以同时表达切片、周期、扫频和长持续背景；
3. 全局平均池化只保留均值，不保留局部出现强度和时序波动；
4. 所有类别共享单一 64 维瓶颈，SawFM 早期能力容易被后期总体梯度覆盖；
5. 之前 A6-D2 增加解冻深度仍失败，说明不能只靠扩大可训练范围解决。

## 4. PAM-IQ 网络结构

### 4.1 总体数据流

```text
Input [B,2,3000]
  ├─ real [B,1,3000]
  └─ imag [B,1,3000]
          ↓
Complex residual block, 1 -> 16, kernel 7
          ↓
Complex average pooling, length 3000 -> 1500
          ↓
Complex residual block, 16 -> 32, kernel 5
          ↓
Phase interaction fusion
  concat(real, imag, magnitude, cos(delta phase), sin(delta phase))
  [B,160,1500] -> 1x1 projection -> [B,64,1500]
          ↓
Stage 1: multi-scale depthwise residual + ECA
  kernels/dilations = (3,1), (7,2), (15,4)
  [B,64,1500] -> [B,64,750]
          ↓
Stage 2: multi-scale depthwise residual + PCAF
  [B,64,750] -> [B,96,375]
          ↓
Stage 3: multi-scale depthwise residual + Global Context
  [B,96,375] -> [B,128,375]
          ↓
Attentive statistics pooling
  weighted mean + weighted standard deviation -> [B,256]
          ↓
projection + LayerNorm + GELU -> [B,128]
          ↓
linear multi-label classifier -> [B,6]
```

### 4.2 复数残差前端

与当前网络只使用一层复卷积不同，PAM-IQ 在两级残差块中持续使用复数乘法：

```text
y_real = W_real * x_real - W_imag * x_imag
y_imag = W_real * x_imag + W_imag * x_real
```

实部、虚部直到长度 1500 时仍保持独立状态。使用平均池化而不是分别对正负复分量做最大池化，避免符号选择破坏复关系。

### 4.3 相位增量特征

不直接使用容易发生 `pi/-pi` 跳变的 `atan2`，而使用归一化相邻复向量内积：

```text
cos(delta_phi_t) = r_t r_(t-1) + i_t i_(t-1)
sin(delta_phi_t) = i_t r_(t-1) - r_t i_(t-1)
```

它能够表达瞬时相位变化，同时保持数值连续，适合扫频、频移、调制和重复切片结构。

### 4.4 多尺度时序块

每个 stage 使用三个 depthwise 分支：

| 分支 | kernel | dilation | 有效范围 | 主要目标 |
|---|---:|---:|---:|---|
| local | 3 | 1 | 3 | 短时跳变、切片边缘 |
| medium | 7 | 2 | 13 | 局部周期与调制 |
| long | 15 | 4 | 57 | 扫频、间歇和较长持续结构 |

分支拼接后通过 `1x1 Conv` 融合并与输入残差相加。首两个阶段总下采样倍率为 4，避免过早抹掉弱成分。

### 4.5 稀疏层级注意力

- Stage 1：ECA，只做低成本通道重标定；
- Stage 2：PCAF，用多膨胀 depthwise temporal filter 形成通道门控；
- Stage 3：只使用一次 Global Context，汇聚全局复合组合背景；
- 不在每层重复堆叠注意力。

### 4.6 注意力统计池化

当前全局平均池化只能描述“平均激活”。PAM-IQ 同时输出加权均值和加权标准差：

```text
mu = sum_t a_t h_t
sigma = sqrt(sum_t a_t h_t^2 - mu^2)
```

均值描述干扰成分总体存在程度，标准差描述间歇性和时序变化，注意力权重用于保留关键片段。

## 5. 参数量与公平对照

| 模型 | backbone | width | 特征维度 | 部署参数 |
|---|---|---:|---:|---:|
| B0 | 当前 Base-DB | 16 | 64 | 21,990 |
| B1 | 当前结构容量匹配 | 56 | 224 | 260,630 |
| P0 | PAM-IQ 完整版 | 16 | 128 | 253,325 |
| P1 | PAM-IQ 无相位增量 | 16 | 128 | 249,229 |
| P2 | PAM-IQ 无稀疏注意力 | 16 | 128 | 236,007 |
| P3 | PAM-IQ 改回平均池化 | 16 | 128 | 232,780 |

B1 与 P0 参数量只差约 2.8%，是判断“收益来自容量还是结构归纳偏置”的主要公平对照。

## 6. 已实现接口

`experiments/run_tapc.py` 的 `--backbone` 可直接使用：

| 名称 | 作用 |
|---|---|
| `phase_multiscale` / `pam_iq` | 完整 PAM-IQ |
| `pam_iq_no_phase` | 删除相邻相位增量 |
| `pam_iq_no_attention` | ECA/PCAF/GC 全部替换为 Identity |
| `pam_iq_avg_pool` | 将注意力统计池化替换为全局平均池化 |

旧的 `complex_1d_true` 保持完全不变，已有结果和 checkpoint 不受影响。

## 7. 下一阶段实验方案

### 7.1 S1：容量与结构筛选

固定协议：

- 数据：0716 六类训练集；
- Label-LT：N=2400，SawFM=400，IF=4；
- seeds：2041、2042、2043，仅作结构发现；
- 随机初始化，完整 DB，20 epochs；
- 同 seed、同 split、同 manifest；
- 同时保存 macro-AUPR 最优与 SawFM-AUPR 最优检查点；
- locked test 不使用。

第一批只运行 B0、B1、P0，共 9 次训练。这样能够先回答：

1. B1 是否仅凭增加容量就能解决问题；
2. P0 相对参数匹配 B1 是否存在结构收益；
3. P0 是否减弱 SawFM 的训练后期遗忘。

P0 相对 B1 的晋级门槛：

- 平均 delta macro-AUPR `>= -0.005`；
- 平均 delta SawFM-AUPR `>= +0.010`；
- SawFM 正收益至少 2/3 seeds；
- weak-component recall 平均提高至少 `+0.02`；
- Hamming loss 不恶化超过 `+0.005`。

### 7.2 S2：结构消融

只有 P0 通过 S1 后，运行 P1/P2/P3：

- P0 vs P1：证明相位增量是否有效；
- P0 vs P2：证明稀疏层级注意力是否有效；
- P0 vs P3：证明统计池化是否有效；
- P0 vs B1：证明不是单纯参数增加。

任何模块若三种子平均贡献小于 `0.002` 且增加复杂度，则从最终模型中删除。

### 7.3 S3：独立确认与长线扩展

结构与超参数冻结后，使用未参与结构选择的新 seeds 做独立确认。只有确认通过，才将同一结构、同一宽度扩展到 VFT、ISRJ；不得按尾类重新选择 kernel、注意力或宽度。

最终 locked test 仍只在模型、训练预算、温度、阈值和统计分析全部冻结后执行一次。

## 8. 已完成验证

- 新模型结构/接口/相位敏感性/梯度测试：`11 passed`；
- 项目全量回归测试：`139 passed, 36 warnings`，警告均为已有 AMP API 弃用提示；
- width=16 部署参数：253,325；完整 Student-A 训练参数：303,629；
- RTX 5080 CUDA 前向：`[4,2,3000] -> temporal [4,128,375] -> feature [4,128] -> logits [4,6]`；
- 2400 样本 Label-LT、DB Loss、1 epoch TAPC GPU 冒烟完成；
- 冒烟训练时间约 4.88 秒，`locked_test_used=false`；
- 冒烟输出：`outputs/tapc/phase_multiscale_backbone_smoke_v2/`。

冒烟的 macro-AUPR 等性能数值只证明链路可运行，不用于模型优劣结论。

## 9. 代码与产物

- 新骨干：`cjlt/models/phase_multiscale.py`
- 构建入口：`cjlt/models/backbone_adapter.py`
- 冻结/局部适配兼容：`cjlt/models/student_a.py`
- 单元测试：`tests/test_backbone_adapter.py`
- GPU 冒烟：`outputs/tapc/phase_multiscale_backbone_smoke_v2/`
