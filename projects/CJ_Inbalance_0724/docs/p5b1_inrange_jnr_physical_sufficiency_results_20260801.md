# P5b1：10–30 dB 扫描 JNR 的物理充分性 Oracle 审计

## 冻结结论

P5b1 通过预注册的全部 11 项 O4 充分性门槛。结论严格限于：在已知真实六标签组合（oracle composition）、不向任何 probe 提供标称功率的条件下，单 PRI 原始 I/Q 中含有可稳定估计 MFT、VFT 和 ISRJ 扫描 JNR 的信息。因此，P5b 九功率中 10 dB MFT/ISRJ 的退化**不能**归因为“观测物理信息绝对不足”。

这不是部署结论：oracle 组合在推理时不可获得，P5b1 不包含分类器、阈值、路由器或端到端附着；它也没有处理 35–50 dB 参数 OOD。

## 预注册与数据边界

- 预注册：`configs/p5b1_inrange_jnr_physical_sufficiency_preregister_20260801.json`，SHA256 `de01fdcec5afe21c0852b376a39e11b89fd6535449b74cbc66c4f5f952b87a18`。
- 数据：新生成身份 seed `780000000`；10/15/20/25/30 dB 各 3,780 个身份、63 个组合、五个功率视图严格按 `receiver_noise_seed` 配对。
- 身份角色：每个组合 train/early/confirm = 36/12/12；同一身份在全部功率视图保持同一角色；与根数据、P4c、P4d、P4k、P5a、P5a2、P5b 身份池重叠为 0。
- 训练仅使用 train，early 只用于 checkpoint/O1 模型选择，confirm 仅作结论；未生成、未打开 test 或 locked test。
- 有效参数始终采用 `label=1 AND parameter_mask=1`，非激活 NaN 从不零填充为训练目标。

## 层级结果（confirm，三模型种子均值）

| 层级 | MFT.JNR NMAE / r | VFT.JNR NMAE / r | ISRJ.JNR NMAE / r |
|---|---:|---:|---:|
| O0：train median 常数 | 0.84853 / 0.00000 | 0.84853 / 0.00000 | 0.84853 / 0.00000 |
| O1：固定时域 + FFT + 自相关 + oracle 组合 | 0.18259 / 0.96515 | 0.18774 / 0.96450 | 0.20102 / 0.95907 |
| O2：冻结 MLCNN 表征 + oracle 组合 | 0.29651 / 0.92556 | 0.31419 / 0.91855 | 0.34309 / 0.90041 |
| O3：冻结 IQ-Grouped 表征 + oracle 组合 | 0.20060 / 0.95932 | 0.19607 / 0.96010 | 0.22087 / 0.95193 |
| O4：width-24 IQ-Grouped + 三个目标头 + oracle 组合 | **0.10070 / 0.99067** | **0.09226 / 0.99253** | **0.11196 / 0.98769** |

O4 对 MFT 与 ISRJ 相对 O0 的确认 NMAE 平均下降分别为 **88.14%** 与 **86.83%**；10 dB 平均下降分别为 **91.01%** 与 **90.02%**。三种子、逐组合最坏差值、VFT 控制和 identity-clustered bootstrap 均通过。MFT 三个 bootstrap gain 95% CI 下界为 0.74936/0.74704/0.73072，ISRJ 为 0.73790/0.73135/0.72003。

## 机理解释与下一道门

1. O1 已显示：在 oracle 组合条件下，简单固定的时频统计就可恢复大量扫描 JNR 信息；因此“信号不可辨识”不是主因。
2. O2 明显弱于 O1/O3，说明为分类而学得的 MLCNN 表征并非 JNR 估计的最充分表示；这不能用来否定冻结分类器的标签后验价值。
3. O4 的额外优势说明仍存在目标特异的 I/Q 信息，但当前不允许直接构造或附着深度 JNR 专家。
4. 下一问题必须是部署缺口：冻结分类器提供的软六标签后验能否替代 oracle 组合。该问题使用全新身份、独立预注册的 P5b2 完成；P5b1 的 confirm 结果不用于选择 P5b2 的阈值、结构或种子。

## 可复核产物

- 实现：`cj0724/p5b1_jnr_physical_oracle.py`
- 定向测试：`tests/test_p5b1_jnr_physical_oracle.py`
- 数据审计：`outputs/p5b1_inrange_jnr_physical_sufficiency_20260801/data_audit.json`
- 角色回执：`outputs/p5b1_inrange_jnr_physical_sufficiency_20260801/role_receipt.json`
- 完整结果：`outputs/p5b1_inrange_jnr_physical_sufficiency_20260801/result.json`，SHA256 `cce2196c08c04e07cd50a2429e14fe7f0028792c028386293ea50a29503a0c01`

`test_generated=false`，`test_opened=false`，`locked_test_used=false`。
