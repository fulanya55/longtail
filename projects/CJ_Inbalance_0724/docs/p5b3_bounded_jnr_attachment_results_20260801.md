# P5b3：冻结 P4c 的扫描 JNR 有界附着（开发证据）

P5b3 在新身份 seed `850000000`、五档 10–30 dB 配对视图上通过全部预注册门槛。P4c 的分类分支、权重、normalizer 和阈值全部冻结；候选只是在 MFT/VFT/ISRJ 的**冻结分类决策为阳性**单元写入小型物理特征 + 分类后验估计值。

| 目标 | 冻结基线 oracle-gated NMAE | 有界附着 NMAE | 三种子平均降幅 | predicted-gated NMAE 变化 |
|---|---:|---:|---:|---:|
| MFT.JNR | 0.68252 | 0.31795 | 53.42% | -0.50674 |
| VFT.JNR | 0.72417 | 0.34224 | 52.75% | -0.52622 |
| ISRJ.JNR | 0.74213 | 0.40372 | 45.62% | -0.54908 |

MFT/ISRJ 三种子均正、bootstrap CI 下界均大于 0，10/30 dB 未出现超过 0.03 NMAE 的恶化；VFT 也改善。附着前后的分类 logits、概率和阈值决策没有被写入或重算，候选之外的所有参数单元严格保持相同。

这只是新身份开发证据，不能写为最终系统结论。三枚 early-selected 检查点与其输入/目标归一化已冻结：

- `outputs/p5b3_bounded_jnr_attachment_20260801/frozen_candidate_heads/source3041_head14041.pt`
- `outputs/p5b3_bounded_jnr_attachment_20260801/frozen_candidate_heads/source3042_head14042.pt`
- `outputs/p5b3_bounded_jnr_attachment_20260801/frozen_candidate_heads/source3043_head14043.pt`

下一步仅允许在两个全新身份种子上加载这些检查点进行无训练、无校准、无阈值调整的确认；若任一确认门槛失败，停止扩展该附着机制。

结果：`outputs/p5b3_bounded_jnr_attachment_20260801/result.json`，SHA256 `f798c4038c3860a73000e61bc91651726eb31a3431c83073e6a746ff4a2bbeb7`。`locked_test_used=false`。
