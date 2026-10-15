# EfficientNet-B1：VFT 困难负样本改进实验（2026-07-25）

## 结论

本轮在冻结的 VFT-D11 开发协议上完成 E1--E3，每种方法使用匹配的 3 个种子。三种方法均未达到预先规定的晋级门槛，因此不替换 `EfficientNet-B1 + BCE` 基线，也不访问 locked test。

- E1 OHEM 显著降低 VFT FPR，但同时损失弱正样本召回，属于整体压低响应而不是提高可分性。
- E2 OHEM + ranking 进一步损害 VFT AUPR、宏 AUPR 和 Hamming，停止该方向。
- E3 专用残差专家保住了宏 AUPR、Hamming 和弱召回，失败程度小于 E1/E2，但 VFT AUPR 均值仍低于基线，且只有 1/3 种子提升，因此不晋级。

下一轮不再增加困难负样本权重。推荐验证“同身份 30 dB 参考视图 -> 10/15/20 dB 弱视图”的 VFT 正样本定向蒸馏/表征保持，使模型获得新的可辨识监督信号，而不是继续移动同一输出阈值。

## 冻结协议

- 数据：0716 九功率级训练数据，输入统一为 `[B, 2, 3000]`。
- 标签顺序：`FMB, SawFM, SinFM, MFT, VFT, ISRJ`。
- 场景：VFT-D11，同时具有标签频次长尾与低功率可见性困难。
- 训练功率：10/15/20 dB；主开发评价功率：10 dB。
- 训练样本：每个种子 2,400 个身份，功率视图不作为独立样本。
- 种子：20260724、20260725、20260726。
- 训练：30 epochs，batch size 64，AdamW，学习率 `3e-4`，weight decay `1e-4`。
- 分类器：统一 256 维适配头和六标签输出；E3 仅为 VFT 增加两个小型残差头。
- 阈值：独立 calibration role 上逐标签选择 F1 阈值；开发集只评价。
- locked test：未使用。

所有匹配种子的 manifest 与 split 哈希均保存在各自 run 目录中；不同种子有各自确定性的拆分与清单，同一 seed 的 E0--E3 哈希一致。

## 方法

### E0：BCE 基线

六标签 `BCEWithLogitsLoss`，骨干为 `efficientnet_b1_1d`。

### E1：目标标签 OHEM

保留六标签 BCE；对 VFT 阴性样本按当前 VFT logit 选择 batch 内 top-k 困难负样本，`k = ceil(2 * VFT正样本数)`，附加权重 0.5。

### E2：OHEM + pairwise ranking

在 E1 上加入 margin 0.5、权重 0.2 的 VFT 正负排序损失。

### E3：专用有界残差专家

共享 EfficientNet-B1 表征与统一六标签 general classifier；另设弱正样本头和困难负样本头，只修正 VFT logit：

```text
z_vft = z_general + 0.5 * (tanh(s_weak_positive) - tanh(s_hard_negative))
```

其他五个标签的 logit 与 general classifier 完全一致。两个专家末层零初始化，初始模型等价于 general head。

## 三种子开发结果

| 方法 | 参数量 | Macro AUPR | VFT AUPR | Macro F1 | Hamming | 弱正召回 | VFT FPR | 晋级 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E0 BCE | 6,643,398 | 0.8095 | 0.5929 | 0.8114 | 0.2317 | 0.7730 | 0.4559 | 冻结基线 |
| E1 OHEM | 6,643,398 | 0.8075 | 0.5887 | 0.7674 | 0.2328 | 0.6656 | 0.0726 | 否 |
| E2 OHEM+rank | 6,643,398 | 0.7921 | 0.5688 | 0.7550 | 0.2422 | 0.6827 | 0.0589 | 否 |
| E3 residual experts | 6,676,424 | 0.8077 | 0.5897 | 0.7939 | 0.2344 | 0.7514 | 0.2519 | 否 |

E3 相对 E0 的匹配种子均值差：Macro AUPR `-0.0018`、VFT AUPR `-0.0032`、Macro F1 `-0.0175`、Hamming `+0.0027`、弱正召回 `-0.0216`、VFT FPR `-0.2040`。VFT AUPR 仅 1/3 种子为正，未达到“均值至少 +0.02 且至少 2/3 种子为正”的主要门槛。

## 根因诊断

### 1. 额外负样本压力改善的是响应强度，不是排序

在 10 dB，E1 把 VFT 阴性均值从 0.1335 压到 0.0239，但 VFT 阳性均值也从 0.1974 压到 0.0635；阈值以下阳性比例从 0.4883 增至 0.8417。E2 更严重，阈值以下阳性比例达到 0.8852。因此 FPR 的大幅下降不能被解释为正负可分性改善。

### 2. E3 的两个专家没有形成独立专长

E3 专家分数相关系数三种子均值为 `-0.9772`，接近完全反相关。原因是两个头的监督目标基本互为补集，共享同一特征后自然退化为同一 VFT 二分类轴的正反形式，而不是两个互补证据源。

E3 的有界残差对自身 general head 的 VFT AUPR 平均仅增加 `+0.00115`；残差绝对值最大值均值已达 `0.99997`。残差在 VFT 正样本和负样本上的均值分别为 `-0.5523` 和 `-0.7051`，本质上仍主要执行负向抑制。

### 3. 表征没有获得新的可辨识信息

E3 的 general head VFT AUPR 为 0.5885，融合残差后为 0.5897，仍低于 E0 的 0.5929。九个功率级上 E3 的 VFT AUPR 均低于 E0，说明问题不是单独的 10 dB 阈值校准，而是当前监督下未学到稳定的新排序特征。

### 4. 误报组合发生迁移，而非被稳定解释

E0 的主要 VFT 阴性误报集中在含 SinFM/MFT 的组合；E3 降低总 FPR 后，最高误报转移到 `FMB+SawFM+ISRJ`、`SinFM+MFT+ISRJ`、`SinFM+ISRJ` 等不同组合。当前专家没有表现出稳定、可解释的某类干扰抑制专长。

## 下一试验门槛

建议 E4 使用同一 `receiver_noise_seed` 身份的 30 dB 参考视图与 10/15/20 dB 弱视图，但部署和评价仍只输入单个弱视图：

1. 保留 E0 六标签 BCE，不提高 VFT 阴性权重。
2. 只对真实 `VFT=1` 且参考视图预测可靠的样本，加入 reference-to-weak stop-gradient 定向蒸馏或特征保持损失。
3. 困难负样本仍由普通 BCE 约束，不把参考正响应蒸馏给 VFT 阴性。
4. 先做 1 seed 机制检查：弱视图 VFT AUPR必须提高，且阳性均值上升不能伴随阴性同幅上升。
5. 再做匹配 3 seed：VFT AUPR 均值至少 `+0.02`，至少 2/3 种子为正；Macro AUPR、Hamming、弱召回和 FPR继续沿用本轮非劣门槛。

如果 E4 仍不能改善排序，应停止通用深度特征上的损失改造，转向 VFT 专属时频/调制物理特征充分性筛查。

## 复现与产物

```powershell
conda run -n PY10 python -m cj0724.hard_negative_matrix --epochs 30
conda run -n PY10 python -m cj0724.specialized_vft_matrix --epochs 30
conda run -n PY10 python -m cj0724.aggregate_hard_negative
conda run -n PY10 python -m cj0724.diagnose_hard_negative
conda run -n PY10 python -m cj0724.diagnose_specialized_residual
```

- 聚合结果：`outputs/vft_d11_hard_negative_20260725/hard_negative_aggregates.csv`
- 匹配种子差值：`outputs/vft_d11_hard_negative_20260725/hard_negative_deltas.csv`
- 晋级判定：`outputs/vft_d11_hard_negative_20260725/decision.json`
- 九功率与组合诊断：`outputs/vft_d11_hard_negative_20260725/hard_negative_diagnostics.json`
- 专家内部诊断：`outputs/vft_d11_hard_negative_20260725/specialized_residual_diagnostics.json`

