# 两条路线协议纠错实验结果

日期：2026-07-22  
环境：`D:\anaconda3\envs\PY10\python.exe`  
数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`  
结论边界：四类 development pilot，`locked_test_used=false`

## 1. 结论

1. 路线 A 的主要问题已被确认：此前 Power-LT 不是标签频次长尾，不能用于判断重加权损失。本轮在真实 label-LT 上，DB Loss 在三个 seed 上都提高 macro-AUPR，平均从 ASL 的 0.94206 提高到 0.94788；ISRJ 尾类 AUPR 从 0.92746 提高到 0.93199。DB Loss 可保留为下一轮排序指标基线，但提升幅度较小，且固定 0.5 阈值下尾类召回明显下降，必须配合逐标签阈值校准后再讨论部署。
2. 路线 B 的数据协议错误已修复，初始伪标签也确实足够准确；然而即使把 `lambda_u` 降到 0.05，仍显著差于纯监督 continuation。问题已经从“伪标签本身错误”转为“强增强一致性目标与监督优化冲突、固定阈值随 teacher 漂移造成伪标签侧别失衡”。因此停止当前形式的伪标签训练，不继续增加 prototype、动态阈值或更大权重。

## 2. 路线 A：真实 label-LT

固定头尾顺序：`FMB -> SinFM -> MFT -> ISRJ`。三个 seed 的实际标签计数分别为：

| seed | FMB | SinFM | MFT | ISRJ | 样本数 |
|---:|---:|---:|---:|---:|---:|
| 2026 | 483 | 388 | 269 | 187 | 610 |
| 2027 | 469 | 382 | 265 | 199 | 597 |
| 2028 | 448 | 357 | 253 | 177 | 561 |

三 seed 汇总：

| loss | macro-AUPR mean | macro-AUPR std | ISRJ AUPR mean | ISRJ AUPR std | fixed-0.5 macro-F1 | fixed-0.5 Hamming |
|---|---:|---:|---:|---:|---:|---:|
| ASL | 0.94206 | 0.00749 | 0.92746 | 0.00484 | 0.76655 | 0.32056 |
| CB-ASL | 0.94189 | 0.00809 | 0.93104 | 0.00659 | 0.76374 | 0.32417 |
| DB Loss | **0.94788** | 0.00712 | **0.93199** | 0.00256 | **0.82044** | **0.18972** |

相对 ASL，DB Loss 的 macro-AUPR 平均提高 0.00582，尾类 AUPR 平均提高 0.00452，而且三个 seed 的方向一致。相对 CB-ASL，DB Loss 的 macro-AUPR 平均提高 0.00598，但尾类 AUPR 只提高 0.00095，尾类优势并不稳定。

需要警惕：DB Loss 的 ISRJ 固定 0.5 阈值召回均值约为 0.582，而 ASL 约为 0.905。AUPR 提升说明排序略好，但输出分数分布发生了明显变化；不能只根据固定阈值 F1/Hamming 宣称识别能力全面提升。

路线 A 决策：

- 保留 DB Loss，定位为“真实 label-LT 下的候选排序基线”；
- 下一次同路线实验只比较 ASL 与 DB Loss，并统一用 calibration role 选择逐标签阈值；
- 在完成阈值校准前，不把 DB Loss 替换成默认部署损失；
- 不再用 Power-LT 结果证明类别重加权有效。

## 3. 路线 B：协议纠错后的半监督试验

本轮完成的固定修复：

- 1024 个未标注样本按 15 种组合确定性分层抽样；四类先验均约为 0.532；
- weak augmentation 噪声改为 18～30 dB；
- 正、负伪标签损失分别归一化；
- 每个标签的正、负伪标签分别限制到每批最多 30%；
- 所有 continuation 从同一个 B0 best checkpoint 开始；
- 阈值由 calibration role 的 weak view 分位数产生；隐藏真值只用于训练后的审计。

初始 B0 审计：正伪标签 precision 四类均为 1.0；负伪标签最低为 ISRJ 的 0.9175；总接受覆盖率 0.2043。说明旧实验的主要问题确实包含未标注截断和 weak augmentation 失真，但修复后伪标签入口已经通过 0.90 精度门槛。

| variant | macro-AUPR | Δ vs B0_CONT | Hamming | Δ vs B0_CONT | max pseudo coverage |
|---|---:|---:|---:|---:|---:|
| B0_PRETRAIN | 0.96739 | - | 0.28667 | - | 0 |
| B0_CONT | **0.97944** | 0 | **0.24083** | 0 | 0 |
| B1_U005 | 0.96139 | -0.01805 | 0.32750 | +0.08667 | 0.26032 |
| B1_U010 | 0.95889 | -0.02055 | 0.30083 | +0.06000 | 0.26646 |

两档半监督权重都没有通过性能门槛。B1_U005 在 best epoch 的监督损失为 0.14062，未监督损失为 0.52889；即使乘以 0.05，未监督项仍约等于监督项的 18.8%。B1_U010 的相应比例约为 30.9%。这说明当前 strong augmentation 上的硬伪标签 BCE 不是温和正则，而是一个相当强的竞争目标。

最终 teacher 审计还显示固定阈值发生明显侧别漂移：FMB 负侧覆盖约 29.7%，SinFM 正侧覆盖约 29.1%，而 MFT 负侧没有样本通过。已接受侧的精度仍高，但各标签获得的训练信号严重不均衡。单纯降低 `lambda_u` 无法修复这种优化方向和覆盖结构问题。

路线 B 决策：

- 停止当前“EMA teacher + strong augmentation + 硬伪标签 BCE”路线；
- 不进入 prototype、distribution alignment、动态阈值或参数一致性叠加；
- 下一候选应改为不产生类别伪标签的自监督预训练，然后用相同 20% 标注集监督微调；
- 较低成本的备选是冻结 backbone，只训练分类头验证无标签表示是否提供增益；只有纯监督 continuation 仍作为比较基线。

## 4. 产物

- 冻结方案：`docs/implementation/two_route_protocol_correction_experiment_plan_20260722.md`
- 路线 A 汇总：`outputs/tapc/route_a_label_lt_losses/route_a_label_lt_loss_summary.json`
- 路线 A 各 run：`outputs/tapc/route_a_label_lt_losses/{asl,cb_asl,db}/seed_{2026,2027,2028}`
- 路线 B 汇总：`outputs/student_b/route_b_protocol_correction/seed_2026/route_b_protocol_summary.json`
- 路线 B 初始审计：`outputs/student_b/route_b_protocol_correction/seed_2026/initial_pseudo_audit.json`
- 路线 B 最终 teacher 审计：各 B1 目录下的 `best_teacher_pseudo_audit.json`

本轮没有读取或评估 locked test。
