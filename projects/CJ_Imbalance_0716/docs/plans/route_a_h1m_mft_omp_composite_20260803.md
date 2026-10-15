# Route A H1m：固定三原子 OMP 残余的六类复合验证

H0l 在新配对、单 MFT 来源上通过：正确 LFM 三原子复数 OMP 的残余能量区分 8 与 3 副本，并击败同容量相位置乱字典。H1m 固定该 OMP 字典 `[400,999]`、原子数 3、候选/控制和所有端点，不训练任何模型。

使用新标准六类、train-only 5,040 样本来源（base seed 870000000，预期 receiver IDs `1068677743--1068682782`），按 receiver mod 10 记录 fit/validation/evaluation=60/20/20。只在 evaluation 角色上，以 10 dB 实测功率定义的弱 MFT 正例对其他五标签完全相同的 MFT 缺失负例做一次评价。fit/validation 不用于拟合或选择。

主端点是候选−相位置乱控制的弱 MFT AUROC；次端点为 AUPR。固定 evaluation 行上执行 10,000 次类别分层、成对 bootstrap；五个接收机子折为 `floor(receiver_noise_seed/10) mod 5`。必须同时满足：AUROC 差≥+0.015 且 CI 下界>0；AUPR 差>0 且 CI 下界>0；至少 4/5 子折 AUROC 差为正；全部身份、角色、零拟合和 test-v3 sealed 合同通过。通过仅允许一份新的冻结复现，不允许直接融合；失败关闭三原子 OMP 残余路线且不在 H1m 身份上调字典、原子数或归一化。
