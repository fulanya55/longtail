# EfficientNet-B1 VFT 同身份蒸馏与物理分支实验（2026-07-25）

## 最终结论

本轮按“先机制门槛、再多种子扩展”的方案执行了 E4a、E4b 和 E5。

- E4a 在线同模型参考蒸馏失败：停止梯度不能阻止共享模型参数造成教师目标漂移。
- E4b 独立冻结 E0 教师仍失败：只提升清晰视图正响应无法提供困难负样本区分证据。
- E5 VFT 物理匹配滤波分支通过全部开发晋级门槛：3/3 种子稳定改善，VFT AUPR 均值由 0.5929 提升到 0.9260，VFT FPR 由 0.4559 降至 0.1516。

E5 当前应作为 VFT-D11 的开发候选方法，替代 E0 的 VFT 输出；其他五个标签继续使用 EfficientNet-B1 的原输出。locked test 未使用，不能把本结果表述为最终测试性能。

## 冻结协议

- 数据：0716 九功率级训练文件；没有读取任何 test MAT。
- 输入：统一 `[B,2,3000]` I/Q。
- 标签顺序：`FMB, SawFM, SinFM, MFT, VFT, ISRJ`。
- 场景：VFT-D11；训练弱功率 10/15/20 dB，主评价 10 dB。
- 种子：20260724、20260725、20260726。
- 每个种子：2,400 个 optimization 身份；early-stop、calibration、development 每类组合各自独立。
- 同一身份的九功率视图不拆成独立样本。
- E0：`efficientnet_b1_1d + BCE`，其骨干、统一分类头、阈值和 checkpoint 不修改。
- 主晋级门槛：VFT AUPR 均值至少 `+0.02`、3/3 种子为正；Macro AUPR、Hamming、弱召回和 VFT FPR 满足非劣要求。

## E4：同身份参考视图定向蒸馏

### E4a 在线同模型教师

弱视图继续计算完整六标签 BCE。对真实 `VFT=1` 且 30 dB 参考 logit 高于弱视图的样本，附加停止梯度 Smooth-L1 追赶项：

```text
L = BCE(z_weak, y) + 0.2 * SmoothL1(z_weak,VFT, stopgrad(z_30dB,VFT))
```

目标最多比当前弱视图 logit 高 2.0，避免无界响应拉升。

seed 20260724 结果：

| 方法 | Macro AUPR | VFT AUPR | 弱召回 | VFT FPR |
| --- | ---: | ---: | ---: | ---: |
| E0 | 0.8170 | 0.5976 | 0.7628 | 0.4645 |
| E4a | 0.7860 | 0.5518 | 0.9958 | 1.0000 |

训练损失在约第 16 epoch 后持续升高。根因是参考分支虽停止梯度，但教师和学生共享正在更新的模型参数，目标仍随训练漂移并形成正反馈。按照机制停止门槛，没有扩展到另外两个种子。

### E4b 独立冻结 E0 教师

使用同 seed 的 E0 `best.pt` 作为独立教师，设置 `eval()`、全部参数 `requires_grad=False`，并在 protocol 保存教师路径与 SHA-256。

seed 20260724 结果：

| 方法 | Macro AUPR | VFT AUPR | 弱召回 | VFT FPR |
| --- | ---: | ---: | ---: | ---: |
| E0 | 0.8170 | 0.5976 | 0.7628 | 0.4645 |
| E4b | 0.7922 | 0.5457 | 0.9244 | 0.8347 |

冻结教师消除了训练发散，但没有改善排序。这证明 E4 失败不只是教师漂移：清晰正样本响应没有带来弱正与困难负样本之间的新可辨识信息。按照停止规则，没有继续搜索蒸馏权重。

## E5：VFT 物理匹配滤波分支

### 物理依据

VFT 生成器定义为延迟参考 LFM：

```text
x_vft(t) = s_lfm(t-delay) * exp(j*2*pi*fd*t),  fd in [1,7] kHz
```

数据参数为：采样率 10 MHz、脉宽 100 μs、带宽 5 MHz、长度 3000。E5 构造 `-8 kHz ... +8 kHz`、步长 500 Hz 的 LFM 多普勒匹配滤波器组，在全部合法延迟窗口中取归一化相关峰，并形成 72 维可部署特征：

- 各多普勒频率的归一化相关峰；
- 相对零频匹配增益；
- 最佳多普勒频率；
- 最佳延迟位置；
- 正/负多普勒峰和峰分布熵。

没有使用 `parameter_values`、VFT 真值多普勒、组合编号、身份编号或 test 数据作为分类输入。

### 物理特征充分性筛查

仅在 optimization 身份上拟合五个轻量分类器；只用 early-stop 身份选型。三个种子均选中 `HistGradientBoosting(max_leaf_nodes=7)`。

| seed | E0 VFT AUPR | 物理分支 VFT AUPR | 差值 |
| ---: | ---: | ---: | ---: |
| 20260724 | 0.5976 | 0.9224 | +0.3248 |
| 20260725 | 0.5892 | 0.9265 | +0.3373 |
| 20260726 | 0.5917 | 0.9292 | +0.3375 |

泄漏反证：

- optimization 与 development 身份交集为 0；
- 打乱 optimization 标签后，三个种子 development AUPR 为 0.453、0.511、0.529，回到约 0.508 的阳性率附近；
- 特征提取只读取 `data_legacy` 和冻结身份索引。

### E0 + 物理分支融合

在 calibration role 上从 `0,0.05,...,1` 选择物理概率权重，并重新标定 VFT 阈值。三个种子都选择 `alpha=1.0`，说明 VFT 通道直接使用物理分支比概率平均更好。其他五个标签保持 E0 原输出。

| 指标 | E0均值 | E5均值 | 差值 |
| --- | ---: | ---: | ---: |
| Macro AUPR | 0.8095 | 0.8650 | +0.0555 |
| VFT AUPR | 0.5929 | 0.9260 | +0.3332 |
| Macro F1 | 0.8114 | 0.8615 | +0.0501 |
| Hamming | 0.2317 | 0.1823 | -0.0495 |
| 弱组件召回 | 0.7730 | 0.8784 | +0.1054 |
| VFT FPR | 0.4559 | 0.1516 | -0.3043 |

所有晋级门槛均通过，且 VFT AUPR 为 3/3 种子正改善。

### 九功率级 VFT AUPR

| VFT功率/dB | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E5 VFT AUPR | 0.926 | 0.976 | 0.947 | 0.964 | 0.991 | 0.996 | 0.998 | 0.998 | 0.998 |

10 dB calibration 阈值被固定用于所有功率，因此 15/20 dB FPR 会高于 10 dB；这属于跨功率校准问题，不改变高 AUPR 所表明的排序能力。

## 标签与组合结果

10 dB 下 E5 只修改 VFT 通道：

| 标签 | E0 AUPR | E5 AUPR | E0 F1 | E5 F1 |
| --- | ---: | ---: | ---: | ---: |
| FMB | 1.0000 | 1.0000 | 0.9988 | 0.9988 |
| SawFM | 0.9999 | 0.9999 | 0.9966 | 0.9966 |
| SinFM | 1.0000 | 1.0000 | 0.9990 | 0.9990 |
| MFT | 0.6410 | 0.6410 | 0.6806 | 0.6806 |
| VFT | 0.5929 | 0.9260 | 0.5219 | 0.8224 |
| ISRJ | 0.6233 | 0.6233 | 0.6713 | 0.6713 |

63 个组合的平均 exact accuracy 从 0.1464 提升到 0.2426：36 个组合改善、19 个不变、8 个下降。仍有若干组合 exact accuracy 为 0，主要由 MFT 和 ISRJ 在 10 dB 的高 FPR造成。因此 E5 是 VFT 专项突破，不代表整个六标签组合识别问题已解决。

## 决策与下一步

1. 将 E5 提升为当前 VFT-D11 开发候选方案。
2. 冻结以下内容后，才能申请一次 locked-test 评价：匹配滤波频率网格、72维特征、HistGradientBoosting超参数、物理权重、逐种子阈值和 E0 checkpoints。
3. 在访问 locked test 前，建议用两个新开发 seed 做确认性复现，防止当前三个相邻 seed 偶然偏乐观。
4. 下一研究瓶颈转向 MFT 与 ISRJ 的低功率专项物理分支；不得把 VFT 参数直接扩展为所有类别的真值输入。

## 复现命令与产物

```powershell
conda run -n PY10 python -m cj0724.positive_reference_matrix --online-teacher --seeds 20260724
conda run -n PY10 python -m cj0724.positive_reference_matrix --seeds 20260724
conda run -n PY10 python -m cj0724.vft_physics_screen
conda run -n PY10 python -m cj0724.fuse_vft_physics
conda run -n PY10 python -m cj0724.diagnose_vft_physics_power
```

- E4运行：`outputs/vft_d11_positive_reference_20260725/`（E4a 首次运行目录保留历史方法名 `e4_positive_reference_distill`，其 protocol 中教师路径为空；E4b 为 `e4b_frozen_positive_reference_distill`）
- 物理筛查：`outputs/vft_physics_screen_20260725/`
- E5融合结果：`outputs/vft_physics_fusion_20260725/result.json`
- 九功率结果：`outputs/vft_physics_fusion_20260725/power_aggregates.csv`
- 逐标签结果：`outputs/vft_physics_fusion_20260725/per_label_10db.csv`
- 逐组合结果：`outputs/vft_physics_fusion_20260725/per_combination_10db.csv`
