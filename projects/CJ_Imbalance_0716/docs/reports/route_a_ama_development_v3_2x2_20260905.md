# AMA 六类 IF3 development-v3：2×2 校准/组合均衡筛查

## 1. 阶段目标与证据边界

本阶段预注册比较 `BCE / Logit Adjustment` 与普通采样/组合均衡采样的 2×2 条件。数据使用 30 dB power-sweep 新身份池，固定 F2_DFMS_Full、真实 I/Q `[B,2,3000]`、AdamW `1e-4/1e-4`、batch=32、8 epoch、梯度裁剪 5.0、5 个新种子（2050–2054）。

角色为 IF3 fit=8000、calibration=1008、check=1008、development-test-v3=4032；calibration/check 每组合16，test 每组合64。fit 的 IF=3.0、CF=7.8537，63组合完整，标签计数 `FMB=6729, AMA=2243, SawFM=3973, SinFM=3897, MFT=4069, ISRJ=3986`。四个角色身份交集均为0，train/test receiver seed 交集为0，E5未修改，locked test 未读取。

组合均衡审计发现：IF3 fit 最少组合只有41个样本，63×41=2583，小于固定预算8000，因此“8000样本全局无放回且组合均衡”在当前池中不可行。实际运行的 balanced 条件是“每个周期内按组合轮转、跨周期允许重复”的 proxy，必须按重复率解释，不能称为真正无放回 Anchor。

## 2. 主要结果（固定0.5）

development-test-v3 五种子均值±标准差：

| 条件 | Macro-AUPR | Macro-F1 | 组合准确率 | Hamming | AMA Recall | FMB FPR |
|---|---:|---:|---:|---:|---:|---:|
| BCE + ordinary | 0.999994±0.000003 | 0.999291±0.000200 | 0.995784±0.001370 | 0.000719±0.000204 | 1.000000 | 0.002117 |
| Logit + ordinary | 0.999994±0.000003 | 0.999291±0.000237 | 0.995734±0.001418 | 0.000719±0.000240 | 1.000000 | 0.000000 |
| BCE + cycle-balanced proxy | 0.999977±0.000007 | 0.999120±0.000221 | 0.994792±0.001403 | 0.000893±0.000224 | 1.000000 | 0.000000 |
| Logit + cycle-balanced proxy | 0.999975±0.000009 | 0.999072±0.000287 | 0.994494±0.001775 | 0.000942±0.000291 | 1.000000 | 0.000000 |

calibration/check 上的逐类阈值只产生约千分点级波动；ECE 接近0，说明 30 dB logits 已基本饱和，阈值和长尾损失没有可观察的有效提升。

## 3. 阶段判定

相对 BCE ordinary，四条件在 development-test 上均未满足预注册的尾 Recall +3 pp、稀有组合 +2 pp 等晋级门槛。更重要的是，这不是“模型改进失败”的有效证据，而是**实验应激强度失败**：30 dB 数据对当前 F2 主干过于容易，BCE 在第2–4轮已达到约0.99以上组合准确率，标签长尾效应被高信噪比可观测性完全掩盖。

因此：

- 阶段 A：完成了协议、可行性和 2×2 筛查，但标记为 `screening_inconclusive_due_to_saturation`；
- 阶段 B/C：不在此 30 dB 数据上继续构造困难负样本、结构解码器或专家网络，避免在零误差平台上制造伪收益；
- 阶段 D：未进入，未冻结候选，未申请 locked-test。

## 4. 工程失败与修复

Logit Adjustment 首次运行时 bias 未迁移到 CUDA，导致首个 batch 报 device mismatch；已修复并从同一 manifest、种子和预算完整重跑。失败收据位于 `outputs/route_a_ama_development_v3_2x2_20260905/failures/implementation_failures.json`，不计入科学性能结论。

## 5. 下一 gate

下一轮必须改为真正有应激强度的 development-v4：

1. 使用 10/15/20 dB 或 10–30 dB 分层的全新身份池，训练、calibration、check、test 均按功率分层；
2. 保持 IF3 标签长尾，同时将组合覆盖和组合频数单独记录；
3. 对每个功率层报告 Macro-F1、组合准确率、AMA Recall、FMB FPR、稀有组合和 Hamming；
4. 重新做 BCE、Logit Adjustment、真正可行的无放回子集（明确降低预算）三条件；
5. 只有在低/中功率出现稳定误差且 A 阶段候选通过后，才执行 B1 多目标阈值、B2 一位差异困难负样本和 B3 结构化解码。

当前 v3 的正确结论是“30 dB 无法验证长尾改进”，而不是“长尾方法已经被证明无效”。

## Receipts

- [v3 manifest](../../outputs/route_a_ama_development_v3_2x2_20260905/manifest.json)
- [v3 summary](../../outputs/route_a_ama_development_v3_2x2_20260905/summary.json)
- [v3 analysis](../../outputs/route_a_ama_development_v3_2x2_20260905/analysis.json)
- [failure receipt](../../outputs/route_a_ama_development_v3_2x2_20260905/failures/implementation_failures.json)
