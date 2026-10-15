# Route-A BCE 边界阶段 1：严格标签/组合长尾筛查（2026-09-01）

## 1. 状态

本阶段已完成 GPU 运行，属于 development-v2 筛查，不读取任何 test 或 locked-test 文件，当前冻结 E5 未修改。

- GPU：NVIDIA GeForce RTX 5080
- 环境：`D:\\anaconda3\\envs\\PY10\\python.exe`，PyTorch 2.11.0+cu128
- 六类：FMB、AMA、SawFM、SinFM、MFT、ISRJ
- 输入：真实 I/Q，`[B,2,3000]`
- 骨干：F2_DFMS_Full / `dfms_iq`，宽度 16
- 初始化：C1 F2 checkpoint，仅用于统一初始化；不是 locked-test 结果
- 优化：AdamW，学习率 `1e-4`，weight decay `1e-4`，batch 32，梯度裁剪 5，8 epoch
- 分类损失：普通 BCEWithLogitsLoss
- 训练预算：每个条件 1600 个样本，3 种子筛查；label-IF=2/3 追加到 5 种子确认

原始审计与清单：

- [协议审计 JSON](../../outputs/route_a_bce_boundary_protocol_audit_20260901/audit.json)
- [development-v2 清单汇总](../../outputs/route_a_bce_boundary_v1_20260901/manifest_summary.json)
- [BCE 筛查汇总](../../outputs/route_a_bce_boundary_screen_20260901/summary.json)
- [label-IF=2/3 确认汇总](../../outputs/route_a_bce_boundary_confirm_20260901/summary.json)

## 2. 旧 P3 协议审计

原 P3 名义上使用 IF=5、CF=5，但实际清单并不是该严重度：

| seed | 实际训练数 | 实际 IF | 实际 CF |
|---:|---:|---:|---:|
| 2039 | 1331 | 1.402 | 23.50 |
| 2040 | 1371 | 1.514 | 12.50 |
| 2041 | 1288 | 1.368 | 12.25 |

因此，旧 P3 不能解释为严格的 IF=5/CF=5 双轴长尾。主要问题是组合裁剪后样本数从 4096 降至约 31%–33%，标签频率被明显拉回。

## 3. 严格清单构造

本阶段固定训练量为 1600，并将标签长尾和组合长尾分开构造。

### 3.1 标签长尾

- `label_IF1`：实际 IF 约 1.06–1.12，作为 balanced anchor；
- `label_IF2`：实际 IF=2.0；
- `label_IF3`：实际 IF=3.0。

每个条件保留 63 个组合的身份来源，但通过 MILP 控制六类标签边际计数，实际计数写入 manifest。

### 3.2 组合长尾

- `combo_CF1`：实际 CF 约 1.08；
- `combo_CF4`：实际 CF 约 3.83–3.92；
- `combo_CF8`：实际 CF 约 6.11–8.50。

组合条件同样固定 1600 个训练样本，组合计数和实际 CF 写入 manifest。由于整数计数和可用样本上限，CF=8 的实际值按清单记录，不使用名义值替代。

## 4. BCE 边界结果

### 4.1 常规身份隔离验证集

| 条件 | 种子数 | Macro-AUPR | Macro-F1 | 组合完全匹配率 | Hamming |
|---|---:|---:|---:|---:|---:|
| label_IF1 | 3 | 1.000000 | 0.999866 | 0.999186 | 0.000136 |
| label_IF2 | 5 | 0.996153 | 0.940715 | 0.765332 | 0.049642 |
| label_IF3 | 5 | 0.999733 | 0.974220 | 0.852930 | 0.026546 |
| combo_CF1 | 3 | 1.000000 | 0.999840 | 0.999023 | 0.000163 |
| combo_CF4 | 3 | 0.999999 | 0.999674 | 0.998047 | 0.000326 |
| combo_CF8 | 3 | 0.999999 | 0.999706 | 0.998210 | 0.000298 |

`label_IF2` 是当前最清晰的 BCE 瓶颈：相对于 balanced anchor，组合完全匹配率下降约 23.4 个百分点，Macro-F1 下降约 5.9 个百分点。`label_IF3` 也出现下降，但由于组合覆盖和 seed 差异，均值反而高于 IF=2，不能仅按名义 IF 单调解释。

### 4.2 逐类现象

在 `label_IF2` 的确认运行中，尾部 ISRJ 的召回可降至约 0.55，F1 约 0.71；MFT 召回约 0.84。FMB、AMA、SawFM、SinFM 仍接近饱和。说明 BCE 的主要失效不是宏观 AUPR 立即坍塌，而是尾类决策阈值和多标签组合边界先失效。

### 4.3 功率开发视图

在同一模型上，30 dB 组合率对 `label_IF2` 降至约 0.752，10 dB 和 50 dB 仍分别约 0.317 和 0.274；这些结果只用于 development 诊断，不是 locked-test 结论。

## 5. 机制判断

本阶段支持以下判断：

1. 旧 P3 的名义 IF=5 并没有形成有效的标签长尾，协议需要重构；
2. 在实际 IF=2 时，普通 BCE 已出现可重复的标签尾部和组合边界瓶颈；
3. 仅提高组合 CF 到 4 或 8，在当前常规验证分布上仍未形成明显瓶颈；
4. Macro-AUPR 接近饱和不能代表多标签组合识别没有长尾问题；
5. 当前应优先在 `label_IF2` 上比较长尾损失和重采样，而不是直接在 CF=8 上扩展复杂方法。

## 6. 下一道硬门槛

下一阶段固定使用 `label_IF2` 的五种子确认清单和相同的 BCE 训练协议，依次比较：

1. Class-Balanced BCE（effective-number）；
2. Logit Adjustment；
3. ASL/Positive-Focal；
4. Deferred Reweighting；
5. 标签感知采样；
6. 组合感知采样或有限的组合级边界损失。

每个候选先进行 3 种子筛选。只有同时满足以下条件，才进入 5 种子确认：

- tail-AUPR 提升；
- 组合完全匹配率提升；
- Macro-AUPR 不低于 BCE 锚点的预设容差；
- 30 dB 锚点不明显下降；
- 头部类别 FPR 不显著增加；
- 计算成本和参数量在预注册范围内。

在长尾方法确认前，不读取 locked test，也不修改 E5。

