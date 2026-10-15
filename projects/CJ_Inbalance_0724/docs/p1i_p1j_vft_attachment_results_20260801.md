# P1i/P1j：VFT Doppler 预测门控接入与安全校准

日期：2026-07-31  
证据等级：两轮全新 identity-safe development-confirmation；非 locked test。

## 1. 最近三轮递进

| 轮次 | 单一研究问题 | 核心结果 | 判定 |
|---|---|---:|---|
| P1h | 正确 LFM 参考与时延对齐能否恢复 VFT Doppler | MAE 改善 `46.49%`，Pearson `0.478`，3/3 数据 seed 通过 | 参数专家通过 Level-2 准入 |
| P1i | 模糊度峰可靠性门控能否安全接回参数系统 | predicted-gate NMAE 改善 `33.07%`，但最坏组合相对无条件替换恶化 `8.14%`；VFT FPR `0.19194` | 完整门槛失败 |
| P1j | 无条件参数替换配合逐功率安全分类校准能否形成闭环 | predicted-gate NMAE 改善 `53.08%`；worst-combo 改善 `25.36%`；FPR `0.05323` | 7/7 门槛通过 |

## 2. P1i：成功结果和失败原因

P1i 固定 MLCNN IQ-Grouped task-routed 三个参数 checkpoint，只改变 VFT Doppler 输出。fresh-train 每个组合 40 个身份，按 `20/10/10` 分为 fit、selection、calibration；测试在方法冻结后一次性读取。

三组对照：

1. IQ-Grouped 原 Doppler 参数头；
2. 预测 VFT 阳性后无条件使用 P1h 专家；
3. 根据模糊度峰分数决定是否使用 P1h 专家，否则保留 IQ-Grouped。

可靠性门控的结果：

- predicted-gate NMAE 相对 IQ-Grouped 改善 `33.07%`；
- 三 seed 改善 `29.39% / 39.92% / 29.91%`；
- paired-bootstrap NMAE delta `-0.2904`，95% CI `[-0.3066,-0.2716]`；
- 漏检惩罚 delta `-0.2728`，95% CI `[-0.2904,-0.2576]`。

但 P1i 两项门槛失败：

1. 可靠性门控最坏组合相对无条件专家恶化 `8.14%`；
2. 原冻结 VFT 分类阈值在新域最坏 FPR 达到 `0.19194`。

根因不是物理专家失效。无条件专家的 predicted-gate NMAE 约 `0.455–0.465`，明显优于可靠性门控的 `0.528–0.624`；其最坏组合 NMAE 也更低。单一 peak score 与误差仅弱相关，中等分数样本中仍有大量专家优于 IQ-Grouped 的情况，因此“只保留高峰值样本”造成错误拒绝。

P1i 结论：模糊度峰分数适合解释证据强度，但不足以作为硬参数路由门控；应停止继续调该阈值。

## 3. P1j 冻结方案

P1j 使用全新的 seed `700000000` 数据，仍为 20/25 dB、63 个组合、每组合 40 个身份。方法只包含两个固定动作：

1. 预测 VFT 阳性时无条件使用 P1h 对齐 Doppler 专家；
2. 在 fresh-train calibration 上，为每个模型 seed 选择同一个 VFT 阈值，要求 20/25 dB 分别满足观测 FPR `<=0.06` 且 CP95 上界 `<=0.08`。

其他五类概率、阈值和全部参数输出保持不变。

三个 VFT 阈值由原来的 `0.30 / 0.22 / 0.56` 提高为：

| Model seed | 安全阈值 | calibration 20 dB FPR/CP95 | calibration 25 dB FPR/CP95 |
|---:|---:|---:|---:|
| 2041 | 0.954 | `0.0484 / 0.0735` | `0.0516 / 0.0773` |
| 2042 | 0.935 | `0.0452 / 0.0697` | `0.0516 / 0.0773` |
| 2043 | 0.953 | `0.0484 / 0.0735` | `0.0516 / 0.0773` |

方法冻结 SHA-256：`77eb563ec8abec18916a7149936c8eb9f6add90166a30952ca254aef692701fd`。

## 4. P1j 正式结果

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| predicted-gate Doppler NMAE 改善 | `53.08%` | `>=30%` | 通过 |
| 三 seed 改善 | `53.23 / 52.96 / 53.03%` | 3/3 正向 | 通过 |
| worst-combo NMAE 改善 | `25.36%` | `>=15%` | 通过 |
| 最大测试功率 VFT FPR | `0.05323` | `<=0.10` | 通过 |
| VFT Recall 保留率 | `77.55%` | `>=75%` | 通过 |
| NMAE bootstrap delta | `-0.4627 [-0.4891,-0.4342]` | CI 上界 `<0` | 通过 |
| 漏检惩罚 vs 原阈值基线 | `-0.3016 [-0.3239,-0.2790]` | CI 上界 `<0` | 通过 |

逐 seed 的安全专家 predicted-gate NMAE 为 `0.406 / 0.409 / 0.412`；对应安全 IQ-Grouped 基线为 `0.868 / 0.870 / 0.877`。

安全阈值降低了参数 coverage，但物理专家显著降低了被保留预测中的回归误差，因此最终漏检惩罚仍优于原始低阈值基线。

## 5. 已验证结论

1. VFT Doppler 的主要不可辨识原因是未消除已知 LFM 相位和到达时延，而不是参数范围本身完全不可辨。
2. P1h 专家可以直接替换 VFT Doppler 参数头；额外的 peak-score 硬门控没有必要。
3. 分类安全和参数精度必须分开处理：逐功率置信缓冲控制是否输出 VFT 参数，物理专家控制输出后的数值精度。
4. 在全新身份数据上，这一组合同时满足参数精度、漏检惩罚、最坏组合、FPR 和统计区间门槛。

## 6. 仍未验证的完整创新

P1j 是 VFT 单分支的安全端到端闭环，但完整 Level 3 尚未成立：

- P1g 的 ISRJ Recall rescue 与 P1j 的 VFT 安全参数接入尚未在同一模型输出和同一全新测试集上联合验证；
- SawFM period 专家尚未与 P1j 在同一新域共同评价；
- MFT count 专家仍未通过准入门槛；
- 尚未对组合系统重新测量 FLOPs、显存和推理延迟；
- 当前只覆盖 20/25 dB，不是九档功率泛化。

## 7. 下一轮 P1k

冻结并联合以下已经分别通过的模块：

1. MLCNN 分类 + IQ-Grouped 参数主干；
2. P1g ISRJ 逐功率安全 rescue；
3. P1j VFT 逐功率安全阈值与无条件 Doppler 专家；
4. 已冻结 SawFM period 专家；
5. 预测标签门控下的全部 12 参数指标。

在新的 identity-safe 数据上同时比较共享主干、IQ-Grouped、无物理专家类别条件头和组合方法，并计算 paired bootstrap。只有识别 Recall/FPR、参数 NMAE/漏检惩罚、最坏组合和统计区间全部通过，才能把证据提升为 Level 3。

## 8. 产物

- P1i：`outputs/p1i_vft_reliability_gated_attachment_20260801`
- P1j：`outputs/p1j_vft_safe_attachment_20260801`
- P1i 实现：`cj0724/p1i_vft_reliability_attachment.py`
- P1j 实现：`cj0724/p1j_vft_safe_attachment.py`
- P1j 数据：`generated_data/p1j_vft_safe_attachment_seed700000000_rechunked`
- 全量测试：100 passed。

两轮均为 `test_used_once=true`、`test_fit_or_selection=false`、`confirm_used=false`、`locked_test_used=false`。
