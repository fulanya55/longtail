# Route A H0n：跨脉冲 MFT 相干延迟像的受控物理筛查

## 结论

H0n 通过了所有预注册门，但其结论严格限于**受控的四 PRI 相位干预**：在正确 LFM 参考下，复延迟像的跨 PRI 相干度能够读取 MFT 分量是否被施加逐 PRI 的整体单位模相位。它证明了新的观测协议包含一个零拟合、相位敏感而非功率敏感的物理量；它**不是**六类复合识别结果，也不证明弱 MFT 已可在复合背景中归因。

| 项目 | 预注册值 | 结果 |
|---|---:|---:|
| 每臂配对数 | 640 | 640 |
| 观测 | 4 PRI × 3000 I/Q 点 | 12000 点 |
| MFT 总 JNR | [0, 8] dB | 按固定协议生成 |
| 候选统计 | 正确 LFM、top-8、能量加权复相干度 | 无可训练参数 |
| 对照统计 | 相同 top-8 的非相干幅度峰度 | 对整体 PRI 相位不敏感 |
| 候选 AUROC | ≥ 0.80 | **1.000000** |
| 幅度对照 AUROC | — | 0.500522 |
| 候选 − 对照 AUROC | ≥ 0.20 | **+0.499478** |
| 逐对 bootstrap 95% CI | 下界 > 0 | **[+0.497810, +0.501101]** |
| coherent > scrambled 配对比例 | ≥ 0.75 | **1.000000** |

候选均值为 coherent `0.999791`、scrambled `0.459888`；对照均值分别为 `14.442881`、`14.440209`。因此区分来自 PRI 间复相位一致性，不能由同一延迟像的非相干幅度统计解释。

## 数据与完整性

- 正式来源：`C:\LJQ_code\干扰数据\h0n_mft_crosspulse_pairs_20260803_r2`。
- coherent SHA-256：`3960bd82c2376b03cad90060ca7793466ba3af85867db5ccfd7db47ea2f6d617`。
- scrambled SHA-256：`4014e3cb0328d8f1cdd04964ce8b72f3898287b7bdb6f68f83b3ece8bf71ef43`。
- 审计确认：两臂所有非 I/Q 元数据、640 个 pair identity 和 receiver-noise identity 一致；与所有发现到的历史 `*train*.mat` receiver identity 零交集；sealed test-v3 未读取且仍为 `used=false`。
- 首个无后缀目录因写入前代码异常而不完整；`_r1` 因与 H1m 训练源 640/640 receiver identity 重叠而被审计拒绝、未统计。拒绝收据见 `outputs/tapc/route_a_h0n_mft_crosspulse_generation/rejected_r1_identity_overlap.json`。

## 不能作出的主张

1. 两臂都含 MFT，唯一改变是 MFT 分量的逐 PRI 整体相位；这不是 MFT-present/MFT-absent 分类。
2. 无 FMB、SawFM、SinFM、VFT 或 ISRJ 共现，因此没有复合鲁棒性证据。
3. 该协议从单 PRI 扩展为四 PRI，不能对 sealed 单脉冲 test-v3 推理，也不能替代其任何结果。
4. H0n 不含网络、训练、校准、阈值或融合，不能声称 VTSE-VBF 的端到端增益。

## 唯一允许的下一步

H0o 必须使用官方生成器在四 PRI 复合背景中配对产生 `MFT-present` 与 `MFT-absent`：其余成分、功率、生成器子种子和接收机噪声完全相同，只加入/移除相对弱的 coherent MFT。仍使用 H0n 锁定的统计、ROI、top-k 与对照，且先做身份审计；若候选相对对照不再显著，则关闭这条全局跨 PRI 相干读出，不进入多脉冲网络开发。
