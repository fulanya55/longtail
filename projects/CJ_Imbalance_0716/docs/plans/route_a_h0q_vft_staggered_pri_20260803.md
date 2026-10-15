# Route A H0q：交错 PRI 下 VFT 慢时间相位一致性的零拟合物理筛查

日期：2026-08-03  
状态：预注册；在本文件、实现哈希、单元测试与身份预选全部完成前不生成数据。

## 问题与机制距离

H0i/H1j 已关闭单个 100 us 脉冲中的去斜相位残余。H0q 不改写该读出：它改变观测协议为 8 个**交错 PRI**脉冲，并只检验 VFT 在真实绝对时间戳上的慢时间 Doppler 相位一致性。固定 300 us 的均匀 PRI 会产生 3.333 kHz PRF 混叠，因此不能作为 H0q；交错 PRI 破坏了单一慢时间采样格点的混叠等价。

H0q 是新观测协议的 H0，不是原 0716 单脉冲 development-v3/test-v3 的分类实验，不训练网络、校准器或阈值，也不访问任何 test 文件。

## 一次性数据契约

- 每臂 640 条、按 pair index 对齐；预选 sample seed `2600000000--2600000639`，receiver identity `2798677742--2798678381`；
- 脉冲数 `8`，有效 LFM 脉宽 `100 us`、`fs=10 MHz`、每 PRI 300 us；相邻脉冲起点间隔固定交错为 `[300,331,300,331,300,331,300] us`；
- 两臂共享 pair seed、JNR、arrival delay、复高斯 receiver noise、LFM 和全部绝对脉冲时刻；
- 唯一干预是 VFT Doppler：low `1.0--2.5 kHz`，high `5.5--7.0 kHz`；每对使用同一个均匀随机数映射两个区间；
- 每条 VFT 波形为各有效 LFM 样点乘 `exp(j 2*pi*fD*t_absolute)`，再以整个 8 PRI 记录 RMS 缩放至 `10--30 dB` JNR；
- 输出只在 `C:\LJQ_code\干扰数据\h0q_vft_staggered_pri_20260803`；没有 train/validation/test split，H0 源不可用于 H1 训练。

## 固定候选、控制与指标

对每个脉冲，在固定 arrival-delay window `100--500` samples 上计算正确 100 us LFM 的复匹配系数。对固定 Doppler 网格 `1000:250:7000 Hz`：

\[
S(f,d)=\frac{|\sum_{p=1}^{8}c_{p,d}\exp(-j2\pi f t_p)|}{\sum_{p=1}^{8}|c_{p,d}|}.
\]

候选分数为 high-grid 最大值减 low-grid 最大值；high-Doppler 臂视为正。

控制将相同的八个 `t_p` 用固定非恒等置换 `[0,5,2,7,4,1,6,3]` 重配；复系数、delay 窗口、频率网格、归一化、计算量和所有输入完全相同。它只破坏“相位对应真实脉冲时刻”的物理关系。

不读取 Doppler/JNR/delay 真值作为推理输入；这些字段只用于生成后审计。无学习参数、无优化器、无校准、无阈值。

## 判定门与失败动作

对 640 个 identity-paired low/high 对，固定 5,000 次 pair bootstrap：

1. candidate AUROC `>=0.80`；
2. candidate − control AUROC `>=+0.20` 且 95% CI 下界 `>0`；
3. candidate − control AUPR `>=+0.15` 且 95% CI 下界 `>0`；
4. 同对 high candidate 分数大于 low 的比例 `>=0.75`；
5. 全部生成、配对、历史 train identity 与 test-v3 seal 审计通过。

任一门失败即关闭“该固定交错 PRI 相位一致性读出”，不得调整 PRI 序列、频率网格、delay ROI、对照、归一化或读出后重跑 H0q 身份。通过也仅允许设计另一个新身份的复合 H1，不允许接入原单脉冲模型或访问 test-v3。

