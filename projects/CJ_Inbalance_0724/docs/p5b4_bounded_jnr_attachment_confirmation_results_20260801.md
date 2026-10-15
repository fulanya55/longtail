# P5b4：冻结 JNR-only 附着的双新身份确认

P5b4 对 P5b3 冻结的三个 head、三个 P4c 基线模型、阈值和 normalizer 执行了只读确认；没有训练、early-stop、校准或结构变更。两个新的独立 receiver-noise 身份池为 `860000000` 和 `870000000`，均包含 10–30 dB 五档、每档 3,780 个严格配对身份。

两个数据池都通过全部门槛：分类和非目标参数不变，MFT/ISRJ 在每个冻结模型均为 oracle-gated 正向收益，predicted-gated NMAE 不恶化，identity-clustered bootstrap 下界大于 0，10/30 dB 没有超过 0.03 NMAE 的恶化。

| 确认身份 | MFT 三模型收益 | VFT 三模型收益 | ISRJ 三模型收益 |
|---|---:|---:|---:|
| 860000000 | 54.42% / 53.24% / 53.85% | 55.21% / 51.34% / 54.04% | 45.35% / 47.90% / 42.94% |
| 870000000 | 54.02% / 53.57% / 53.60% | 55.02% / 51.96% / 54.11% | 44.89% / 48.18% / 42.46% |

因此目前可陈述的范围是：在 10–30 dB、已知闭集的复合干扰分布内，异构 P4c 的冻结分类后验可以条件化一个小型固定物理 JNR 估计器，并在不改变分类或其他九项参数的前提下，稳定改善三项扫描 JNR。它**不**证明 35–50 dB 参数 OOD、组合不相交、多轴长尾或 locked test 上的完整系统收益。

结果：`outputs/p5b4_bounded_jnr_attachment_confirmation_20260801/result.json`。两个审计分别为 `audit_seed860000000.json` 和 `audit_seed870000000.json`。`locked_test_used=false`。
