# 路线 A：P6 早期复数 SawFM 前端固定确认方案（2026-07-25）

## 固定项

发现阶段已经选出 `P6_RawPlusStage1`。本阶段冻结全部结构和超参数：

- 输入：RMS 归一化原始 I/Q 物理描述 + F2 第一复数残差层物理描述；
- 特征维数：948；
- 分类器：L2 Logistic Regression；
- `C=0.1`；
- fit 身份：发现阶段相同的 fit role；
- 不再使用 validation 选择模型或超参数；
- F2/H2/S2/C2/A0 全冻结；
- locked test 不使用。

## 主确认集

使用此前未参与 P6 拟合或选参的 `scale` role：每种子每个域 280 对组合匹配 SawFM 正样本/hard negative。`scale` 曾用于既有 S2 专家的尺度拟合，但 P6 不读取 S2、C2 或 A0，因此对 P6 是独立身份确认集。

同时保留原 independent development 作为发现结果复现，不把它当新的确认集。

## 对照

- C0：固定 P1 visibility，`C=0.01`；
- C1：固定 P6，`C=0.1`；
- C2：P6 输入不变，但 fit 标签按固定随机种子置乱，排除高维线性 probe 的偶然高分。

## 晋级门

C1 至少 2/3 种子同时满足：

1. scale mean-power AUROC `>=0.90`；
2. scale worst-power AUROC `>=0.85`；
3. scale 50 dB AUROC `>=0.85`；
4. 相对 C0 scale mean-power AUROC 增量 `>=0.03`；
5. independent development mean-power AUROC `>=0.90`；
6. 标签置乱 scale mean-power AUROC `<=0.60`；
7. 身份审计通过且 locked test 未使用。

通过后只允许进入“小容量 Raw+Stage1 SawFM 独立前端”实现，不允许立即替换 H2 或解冻主干。前端必须先独立训练和确认，再研究有界接回。
