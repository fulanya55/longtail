# Route A：VFT/ISRJ 浅层 STFT 拓扑专家预注册方案

## 1. 单一假设

VFT 的频率漂移和 ISRJ 的周期复位都表现为二维时频图上的局部连续结构。既有 16×16 固定摘要加线性 probe 会压缩空间邻接关系。若原始 I/Q 中确有可用证据，一个严格受限的小型 2D CNN 在 32×32 固定 log-STFT 上应提高严格匹配 weak-positive vs composition-matched-negative AUROC，并优于时间轴打乱但网络容量完全相同的对照。

## 2. 数据与输入

- development-v3；种子 2039–2043；仅 fit/validation/early；test-v3 不访问。
- 标签：VFT、ISRJ，分别训练，不共享标签真值。
- 严格双侧其余五标签组合匹配。
- STFT：complex I/Q，`n_fft=128`、`hop_length=32`、Hann、双边频谱、fftshift；每样本功率归一化后 `log1p(128*x)`，自适应平均池化为 32×32。
- 只用 fit 的全样本图像标量均值/标准差归一化。

## 3. 网络

- Conv 1→8，3×3，GroupNorm(4)，GELU，2×2 MaxPool；
- Conv 8→16，3×3，GroupNorm(4)，GELU，2×2 MaxPool；
- Conv 16→16，3×3，GroupNorm(4)，GELU；
- 全局平均池化与全局最大池化拼接为32维；
- Linear 32→1。

损失为 pos-weight BCE；AdamW，lr `1e-3`，weight decay `1e-4`，batch 128，最多60 epoch，validation AUROC early-stop patience 8，梯度裁剪5。模型与训练预算对齐/打乱完全一致。

## 4. 机制对照

- A：原始32个时间格顺序；
- S：每个种子和标签固定一组32格置换，fit/validation/early均使用同一置换，保持每个时间格的频谱与全局边际，破坏局部时间邻接和演化方向。

保存置换和 SHA-256。A/S使用同一初始化与batch shuffle种子。

## 5. 局部晋级门槛（每标签独立）

1. aligned early AUROC ≥0.620；
2. 相对严格 final-temporal ≥+0.030；
3. 相对 time-shuffled CNN ≥+0.020；
4. 至少4/5种子同时优于 final-temporal 与 shuffled；
5. validation→early AUROC gap <0.150；
6. 参数量和训练预算完全一致；
7. 身份与测试边界安全。

通过只表示该标签获得可用局部专家证据，不代表统一模型通过。失败后停止“浅层STFT拓扑”机制，不加深/加宽网络。

