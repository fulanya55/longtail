# MFT/VFT Oracle-Positive 参数可辨识性试验方案

## 目标

在不使用分类损失、且已知目标类别为阳性的条件下，判断 MFT `num_targets` 和 VFT `Doppler` 是否能够从当前复合波形中被学习。只有专项专家稳定超过常数基线，才允许重新接回 C2 联合模型。

## P0 数据对齐审计

1. 核对 `label_selected`、`known_label` 与注册表六类顺序。
2. 核对 `data_legacy[:, sample_index]` 与 `iq_complex[:, sample_index]`。
3. 核对 MFT/VFT 标签、参数 mask、参数列 28/49（Python 0 基 27/48 为 JNR）和 29/50（0 基 28/49 为专项目标）。
4. 按 `label_all` 活跃类别顺序解引用每个样本的 component metadata，并核对 MFT/VFT component nominal JNR 与参数列。
5. 审计只读取 root train，不读取 locked test。

任何标签-mask 错位、有效参数非有限、component 数量错位或 JNR 不一致都会阻断训练。

## P1 Oracle-positive 专家

所有训练和验证 loader 都先使用固定 development roles，再筛选 `目标标签=1 AND 参数mask=1`。分类损失完全关闭。

| 变体 | 目标 | 表征 | 输出/损失 |
|---|---|---|---|
| MFT_RAW_ORD | num_targets | 原始 I/Q 时域骨干 | 5 个有序阈值，CORAL式累计 BCE |
| MFT_ACF_ORD | num_targets | 线性自相关、多尺度卷积、top-k 延迟峰 | 5 个有序阈值，累计 BCE |
| VFT_RAW | Doppler | 原始 I/Q 时域骨干 | 标准化 SmoothL1 |
| VFT_STFT | Doppler | STFT 谱图 + 逐帧频谱质心/宽度/峰值轨迹 | 标准化 SmoothL1 |
| VFT_IFREQ | Doppler | 相邻采样相位差形成的瞬时频率序列 | 标准化 SmoothL1 |

## S1 筛选预算

- seeds：2041、2042、2043。
- 每个 seed、每个目标：训练阳性 2048，early-stop 阳性 256。
- MFT 按 3--8 六个计数等级轮询抽样；VFT 按目标值十分位轮询抽样。
- 20 epochs，patience 5，batch 64，AdamW，AMP。
- 模型选择只看参数 MAE，不使用任何分类指标。

## 常数基线与晋级门槛

每个 seed 使用对应训练阳性样本的目标中位数作为常数预测，并在相同 early-stop 阳性样本上计算 MAE。

专家重新接回 C2 必须同时满足：

1. 三 seeds 平均 MAE 相对常数基线降低至少 10%；
2. 三 seeds 平均 Pearson 相关系数至少 0.30；
3. 3/3 seeds 的 MAE 改善方向均为正；
4. 3/3 seeds 的相关系数均为正。

若专项专家失败，则不继续调联合损失权重，转向参数生成/波形可辨识性审计或受控单分量数据；若通过，只回接通过门槛的表征支路，并重新验证 C2 分类非劣。
