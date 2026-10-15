# Route A H0o：四 PRI 复合背景中的弱 MFT 跨脉冲归因

## 假设

H0n 仅证明了受控的逐 PRI 相位干预可被复相干延迟像读取。H0o 检验更有意义的部署前假设：**在官方四 PRI 六类生成器的完整五类背景中，正确 LFM 的跨 PRI 复相干延迟像对于“加入一个相对弱、相干 MFT 分量”仍比仅用相同延迟像的非相干幅度统计更具可归因性。**

这是物理 H0，不是网络 H1；不训练、校准、选阈值、融合或读取 sealed test-v3。

## 数据契约（生成前锁定）

- 新来源：`h0o_mft_crosspulse_composite_20260803`，640 对、每条 4 PRI × 3000 个 I/Q 点。
- 预选样本种子：`2000000000--2000000639`。按官方 `derive_substream_seed(sample_seed, 2, 0)` 映射，其接收机身份为 `2198677742--2198678381`；生成前扫描所有历史 `*train*.mat` 后确认零交集。
- 两臂共同使用官方 `compose_jamming_sample` 的相同 `sample_seed`、同一接收机噪声与相同五类背景：`FMB+SawFM+SinFM+VFT+ISRJ`，每一类有效 JNR 固定 20 dB。
- present 臂唯一额外加入 MFT（注册表 ID=6），其有效 JNR 为 8 dB，即相对每个最强背景成分 `−12 dB`，且标为弱分量。
- 采用官方 `create_closedset_6class_config`，仅将 `num_pulses=4`；不改动外部生成器。官方子流按 type ID 而非组合位置派生，所以五个背景分量和接收机噪声在一对中应完全一致。
- 无 train/validation/test split；该源只用于 H0o 固定统计，不能作为后续 H1 的训练数据。

## 固定读出、比较与门

保持 H0n 锁定的正确 LFM、每 PRI `[400,999]` 延迟窗口和 `top_k=8`：

1. 候选：top-8 能量加权跨 PRI 复相干度；
2. 对照：同一 top-8 的相位不变非相干幅度峰度；
3. 标签：present=1、absent=0；以 5,000 次逐对 bootstrap 检验 AUROC/AUPR 的候选−对照差。

通过条件必须同时满足：候选 AUROC ≥ 0.60、候选−对照 AUROC ≥ 0.05 且 CI 下界 > 0、候选−对照 AUPR ≥ 0.03 且 CI 下界 > 0、candidate(present)>candidate(absent) 的配对比例 ≥ 0.60、所有反事实元数据与历史身份审计通过。失败即关闭全局跨 PRI 相干延迟像用于弱 MFT 的具体读出；成功也只允许规划身份隔离的多脉冲 six-class development，不允许直接训练或宣称最终增益。
