# Route A H1a：交错 PRI 下 MFT+ISRJ 复合背景中的弱 VFT 归因

状态：预注册；这是 H0q 通过后、完整 H1 之前的**受限复合筛查**，不是六标签开发实验。

## 单一假设

H0q 已证明真实交错时间戳包含 VFT Doppler 信息。H1a 检验该信息在两个最具相干 LFM 混淆的背景（MFT 多延迟副本、ISRJ 采样/转发副本）中，是否仍能区分“弱 VFT 存在”与“VFT 缺失”。

## 生成与配对

- 新 identities：sample seed `2700000000--2700000639`，640 对；预选历史 train identity 零交集；
- 交错时间轴、8 脉冲、LFM 与 H0q 完全相同；
- 每对共享 MFT、ISRJ、receiver noise、其 JNR、参数子流和所有背景 I/Q；
- absent：MFT + ISRJ + noise；present：完全相同背景加 VFT；
- 背景 MFT/ISRJ 的有效 JNR 固定 20 dB；VFT 有效 JNR 固定 8 dB，即相对每个背景 -12 dB，并标为弱目标；
- VFT Doppler 固定从 1--7 kHz 连续抽样；推理期不使用其真值；
- MFT、ISRJ 的核心生成公式、零边界时延与 JNR 标定从官方 0716 生成器逐项复建，但脉冲起点改为 H0q 的交错时间轴；因此该源必须标为 `project_local_staggered_pri_physics_extension`，不得称为原官方等 PRI 六类数据。

## 固定读出、对照和门

H1a 候选是 H0q 同一 LFM delay grid 与 `1000:250:7000 Hz` 网格上的全频段最大相干度。控制使用完全相同的系数、网格、delay range、归一化和计算，但将时间戳按 H0q 固定非恒等置换重配。没有训练、拟合、校准或阈值。

通过必须同时满足：candidate VFT-present/absent AUROC `>=0.62`、candidate-control AUROC `>=+0.05` 且逐对 bootstrap CI 下界 `>0`、candidate-control AUPR `>=+0.03` 且 CI 下界 `>0`、present candidate 大于 absent 的对比例 `>=0.60`、配对/身份/test-v3 审计全通过。失败关闭本固定 H1a 读出；不得调整 JNR、背景、网格、ROI、统计量或控制后重跑。

即使通过，H1a 也只授权完整五背景/六标签 H1 设计，不能接入原单脉冲模型或 test-v3。

