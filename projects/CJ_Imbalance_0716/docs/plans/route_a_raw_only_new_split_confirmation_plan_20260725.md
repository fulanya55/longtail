# 路线 A：Raw-only SawFM 前端新划分确认与捷径审计（2026-07-25）

## 固定方法

- 前端：`RawStage1SawFMFrontend(mode="raw")`；
- 参数量：5521；
- 输入：RMS 归一化原始 I/Q；
- 优化器/训练预算：AdamW `3e-4`、weight decay `1e-3`、batch 64、40 epochs、patience 8；
- 不使用 F2/Stage1；
- 不使用 locked test。

## 新确认种子

使用此前未参与当前 Raw-only 选模的 split seeds `2039、2040`。每个 seed 从 train role 按其他五类组合构造 SawFM 正样本/hard negative，并分为 fit/validation/scale 三个互斥身份角色；early-stop role 构造 independent development。

## 结构消融

- R0 Full physics：幅度、幅度差分、相位增量；
- R1 Envelope-only：只保留幅度、幅度差分；
- R2 Phase-only：只保留相位增量；
- R3 Time-shuffled control：对每个样本使用固定随机时间置换，破坏局部时序但保留边缘分布。

R0 为正式候选，其余只用于机制和捷径审计。

## 晋级门

R0 在 2/2 新 seeds 同时满足：

1. development mean-power AUROC `>=0.95`；
2. development worst-power AUROC `>=0.90`；
3. scale mean-power AUROC `>=0.95`；
4. validation→development 落差 `<=0.10`；
5. time-shuffled 相对 R0 至少下降 `0.05`，证明局部时间结构有贡献；
6. locked test 未使用且身份重叠为 0。

若 R0 通过但 time-shuffled 不下降，则方法仍可作为识别前端，但不能宣称学习到 SawFM 局部时序机制；必须将贡献表述为原始 I/Q 分布特征。只有完整门通过，才允许设计与 H2 的有界接回实验。
