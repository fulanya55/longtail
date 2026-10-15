# 单视图复合干扰分离与 ISRJ 周期专家验证报告

## 1. 本轮目标与证据边界

本轮针对上一阶段“单视图局部 STFT 掩膜只能恢复部分弱扫频分量”的失败原因，实施两步改进：

1. 用多尺度复数 STFT U-Net 替换局部三层掩膜网络，提高对跨时频结构的建模能力；
2. 冻结 U-Net，在其分离波形上增加 ISRJ 周期/包络自相关专家，验证剩余短板是否来自通用分类头对周期机制利用不足。

数据角色严格固定为：train 拟合模型，early 选择 U-Net checkpoint 或拟合堆叠器，calibration 在 FPR 上限 0.10 下选阈值，tune 比较专家候选。上一阶段已经读取 confirm 并据此定位 ISRJ，因此本轮专家试验属于自适应开发证据，未再次读取 confirm，也未使用 locked test。

## 2. 改进后的网络结构

### 2.1 复数 U-Net 分离器

输入为当前 10 dB 单视图 I/Q，形状为 `[B,2,3000]`。先进行 `n_fft=128, hop=32` 的 STFT，将实部、虚部和对数幅度拼接成 6 通道特征。编码器宽度为 12/24，瓶颈宽度为 48，解码器通过两级跳连恢复时频分辨率，输出两路幅度掩膜和两路相位残差。掩膜作用于原始复谱，经 ISTFT 得到分离波形，再由宽度 12 的 IQ-Grouped 分类器识别 SawFM、VFT、ISRJ。

训练只使用 50 dB 与 10 dB 的同 identity 差分构造 oracle 扫频端点作为教师波形；部署推理只输入当前 10 dB I/Q，不需要功率标签、50 dB 视图或 oracle 波形。

### 2.2 ISRJ 周期专家

U-Net 全部权重冻结。对分离波形计算 310 维机制特征：复自相关多尺度池化、包络自相关多尺度池化、频谱池化、周期峰值、峰位置及 top-8 周期峰序列。候选包括叶节点数 7/15 的 HistGradientBoosting、仅包络周期特征模型，以及将 U-Net ISRJ 分数和各周期模型分数输入二元 LogisticRegression 的堆叠头。

## 3. 损失函数与超参数

U-Net 总损失为：

`L = BCE + 0.5 * (L_cos + 0.2 L_time + 0.25 L_multires_STFT + 0.1 L_negative_energy)`。

关键超参数为 AdamW，学习率 `3e-4`，weight decay `1e-4`，CosineAnnealingLR，8 epochs，batch size 126，梯度裁剪 5.0。checkpoint 由 early 集 macro AUPR 优先、分离余弦相似度次优确定。周期专家固定为 200 boosting iterations、learning rate 0.05、L2=1.0；堆叠器 `C=1.0`、class-weight balanced。

## 4. 结果

### 4.1 U-Net 相对局部掩膜

三 seed 的 confirm 均值如下。该结果是在本轮专家开发之前得到的历史确认结果；它已经被用于定位 ISRJ，因此后续不能再把同一 confirm 当作无偏证据。

| 模型 | Macro AUPR | Macro Recall | Macro FPR | Macro F1 |
|---|---:|---:|---:|---:|
| 局部 STFT 掩膜 | 0.7732 | 0.3943 | 0.0896 | 0.5309 |
| 复数 U-Net | 0.8985 | 0.6493 | 0.0860 | 0.7461 |

U-Net 的类别召回为 SawFM 0.7323、VFT 0.6729、ISRJ 0.5427。说明多尺度分离显著有效，但通用分类头仍不能充分利用 ISRJ 周期结构。

### 4.2 冻结 U-Net 后的 ISRJ 专家（tune-only）

| ISRJ 头 | AUPR | Recall | FPR | 召回增量 | 同向 seed |
|---|---:|---:|---:|---:|---:|
| U-Net 原分类头 | 0.8563 | 0.5406 | 0.0952 | - | - |
| 周期特征 leaf-15 | 0.9264 | 0.7047 | 0.0876 | +0.1641 | 3/3 |
| U-Net + 全周期堆叠头 | **0.9293** | **0.7214** | **0.0898** | **+0.1807** | **3/3** |

最佳堆叠头达到预注册回接门槛：ISRJ 平均召回不低于 0.70、FPR 不高于 0.10、召回增益不低于 0.05，且 3/3 seeds 同方向。

将该 ISRJ 头接回三类系统后，tune 的三 seed 均值由 Macro AUPR 0.8938、Recall 0.6384、FPR 0.0955，改善为 Macro AUPR **0.9182**、Recall **0.6986**、FPR **0.0937**。整体召回距 0.70 只差 0.0014，但三个 seed 中只有 seed 2041 超过 0.70，因此不能宣称整套系统已稳定达到总体门槛。

## 5. 成果与失败原因

已验证成果：

- oracle 端点实验已经证明弱扫频信息存在；U-Net 进一步证明主要瓶颈确实是单视图前端分离，而不是信息完全不可辨识。
- 多尺度 U-Net 同时提高 AUPR、召回并降低 FPR，排除了“仅靠放宽阈值换召回”的解释。
- ISRJ 周期专家在 3/3 seeds 上稳定提升，说明剩余 ISRJ 失败主要是特征归纳偏置不匹配：通用卷积分类头弱化了长延迟周期和包络重复信息。
- 仅包络子集表现不佳，说明 ISRJ 判别需要复自相关、包络自相关和频谱的联合证据，而不是单一周期峰。

仍未完成之处：

- VFT 的历史 confirm 召回为 0.6729，仍未达到 0.70；下一短板已经从 ISRJ 转向 VFT。
- 整体 tune recall 为 0.6986，且 seed 间存在波动，尚不能申请 locked test。
- 本轮专家结果是 tune-only 自适应开发证据，需要在方法完全冻结后使用新的独立开发划分或一次正式确认验证，不能复用已查看的 confirm 进行方法选择。

## 6. 下一步决策门

保持 U-Net 与 ISRJ 堆叠头冻结，下一步只对 VFT 增加瞬时频率序列专家，输入采用分离波形的相位差/瞬时频率轨迹，不使用全局 FFT。采用同样的 train/early/calibration/tune 角色与 FPR 0.10 上限。只有当 VFT 召回达到 0.70、3/3 seeds 同方向、三类 Macro Recall 稳定超过 0.70 且 FPR 不恶化时，才冻结全部规则并申请一次性 locked test。

## 7. 审计产物

- U-Net 实现：`cj0724/conditional_unet_separator_experiment.py`
- ISRJ 专家实现：`cj0724/isrj_separator_physics_head.py`
- U-Net 完整结果：`outputs/conditional_unet_separator_full_20260725/aggregate.json`
- ISRJ 专家协议：`outputs/isrj_separator_physics_head_v2_20260725/protocol.json`
- ISRJ 专家聚合：`outputs/isrj_separator_physics_head_v2_20260725/summary.json`
- 自动化测试：63 passed

所有结果均记录 `locked_test_used=false`。
