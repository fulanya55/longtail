# Route A：ISRJ 联合周期特征专家预注册方案（development-v3）

## 1. 物理假设

0716生成器以10 MHz采样，ISRJ采样宽度为5–20 μs（50–200点），并进行1–5次延迟转发，因而原始复信号中应存在约100–1200点范围的复自相关和包络自相关梳状结构。此前失败的ISRJ摘要把幅度差分、相位差分分别做短统计，没有联合保留长延迟复相关、包络周期、频谱和周期峰。

若该物理证据真实存在，迁移0724项目中已经使用的310维联合周期特征，并用固定小叶节点HGB建模非线性交互，应优于冻结final-temporal、线性周期头以及破坏长周期但保留25点局部波形的块置换对照。

## 2. 固定特征

输入 `[B,2,3000]` I/Q，逐样本RMS归一化：

- complex ACF：FFT 8192，lag 1–1200，120格max pooling；
- centered-envelope ACF：lag 1–1200，120格max pooling；
- 3000点幅度频谱：60格能量池化；
- envelope ACF lag 40–600中的峰值、峰位置和top-8峰；
- 总维度310。

只用fit角色全样本计算逐维mean/std。

## 3. 模型和对照

- L：aligned特征 + LogisticRegression，`C=1`、balanced、max_iter=1000；
- H：aligned特征 + HistGradientBoosting，200轮、lr=0.05、max_leaf_nodes=7、L2=1、关闭early-stop；
- S：25点块置换后的特征 + 与H完全相同的HGB。

块置换对每个种子固定一组120块排列，fit/validation/early一致；它保留每个25点局部复波形、样本幅度和I/Q配对，但破坏50点以上的真实时间间隔。保存置换SHA-256。H/S使用同一随机种子、样本权重和预算。

## 4. 协议

- development-v3 SHA固定；种子2039–2043；严格双侧其余五标签组合匹配；
- fit拟合、validation只报告诊断、不用于模型选择、early作正式开发评估；
- 不训练编码器，不使用sample_width、JNR、weak mask或真实标签作为推理输入；
- test-v3保持sealed/unauthorized/unused。

## 5. 局部晋级门槛

H必须同时满足：

1. early AUROC≥0.620；
2. 相对严格final-temporal≥+0.030；
3. 相对块置换HGB≥+0.020；
4. 相对aligned logistic≥+0.010，证明非线性交互必要；
5. 至少4/5种子优于final-temporal和块置换；
6. validation→early AUROC gap<0.150；
7. H/S模型配置与预算完全一致；
8. 身份及测试边界安全。

通过只表示ISRJ获得可用局部专家证据；失败则停止这套310维联合周期特征在未分离复合波形上的应用，不增加HGB叶节点或搜索lag区间。

