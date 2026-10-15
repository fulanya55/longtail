# Route A H2：多脉冲六标签真实时间戳 VFT 专家结果

日期：2026-08-03  
证据等级：五随机种子、角色隔离的 H2 development 机制验证；不是原始 0716 的统一候选晋级证据。  
最终决定：**停止 H2 的“真实时间戳 VFT 专家 + 当前加性融合”机制版本；不访问 H2 external、历史 locked test、test-v2 或 test-v3。**

## 1. 问题与预注册假设

H2 只检验一个可证伪问题：在相同八脉冲 staggered-PRI 物理协议中，VFT 专家使用真实脉冲时间顺序，是否能稳定优于同参数量、将时间戳置换后的专家。它不是对 VTSE-VBF 全方法的最终验证，也不用于替代 development-v3/test-v3 的证据链。

五个模型均输出六个标签；SawFM 专家只允许修改 SawFM logit，VFT 专家只允许修改 VFT logit，其他标签走安全的共享全局头：

| 臂 | 定义 | 参数量 |
|---|---|---:|
| C0-mp | 共享多脉冲 I/Q backbone + 六标签全局头 | 26,246 |
| C1-mp | C0 + RawSaw 架构 SawFM 局部专家 | 27,671 |
| C2-mp | C0 + **真实时间戳** VFT 统计专家 | 26,679 |
| C3-mp | C1 + 真实时间戳 VFT 专家 | 28,104 |
| C4-mp | C1 + **时间戳置换** VFT 专家 | 28,104 |

C3/C4 在全部五个种子中都严格同为 28,104 参数，排除了简单参数量增益。

## 2. 数据、训练与证据边界

- 训练源：`h2_multipulse_sixlabel_train-mp-v1_20260803_r1`，5,040 views / 2,520 receiver-pair identities，SHA-256 `70810c80e550210456b8e2f46b3503847454dcca5ff07372a3dafb949f8f8585`。
- development 源：`h2_multipulse_sixlabel_development-mp-v1_20260803_r1`，2,520 views / 1,260 receiver-pair identities，SHA-256 `2079699fd99ef1141acd7294da9ce6a5692030eec855f91aa728aabedaf6c43f`。
- 身份范围互不重叠：train `2900000000–2900002519`，development `3000000000–3000001259`；功率视图按 `pair_seed` 分组。
- 正式种子：`20260803, 20260811, 20260819, 20260827, 20260904`。
- 训练：AdamW，BCEWithLogitsLoss，学习率 `3e-4`，weight decay `1e-4`，batch size `16`，24 epochs，gradient clip `5.0`，backbone width `32`，expert dimension `16`。
- 温度校准只在 `pair_seed mod 2 == 0` 的半组拟合；阈值只在另一半选择。VFT 目标为 FPR 不超过 `0.15` 时最大化弱正召回。阈值指标在整个 development 上仅作描述性记录，不能视为独立测试结果。

已完成 25/25 正式运行收据、checkpoint、预测、校准/阈值文件及 SHA-256 产物审计。所有收据均为 `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`；H2 external 保持 `external_used=false`。

## 3. 五种子结果

下表是五个种子的均值；弱 VFT 排序只在 `VFT weak positive OR VFT negative` 上计算。

| 臂 | macro-AUPR | 弱 VFT AUROC | 弱 VFT AUPR | 弱 VFT recall | VFT FPR |
|---|---:|---:|---:|---:|---:|
| C0-mp | 0.94811 | 0.79285 | 0.77372 | 0.44714 | 0.15270 |
| C1-mp | 0.94714 | 0.80964 | 0.79041 | 0.50238 | 0.15079 |
| C2-mp | 0.95033 | 0.80170 | 0.78145 | 0.49111 | 0.15540 |
| C3-mp | 0.94624 | 0.80700 | 0.78720 | 0.49079 | 0.15159 |
| C4-mp | 0.94634 | 0.80740 | 0.78782 | 0.49492 | 0.15175 |

关键配对差值（C3 减基线）如下。bootstrap CI 为五种子配对 bootstrap 95% CI；五个同号改进的精确双侧符号检验为 `p=0.0625`，故不能称为 0.05 水平显著。

| 比较 | 均值差 | 95% CI | 正增益种子 | 解释 |
|---|---:|---:|---:|---|
| C2 − C0，弱 VFT AUROC | +0.00885 | [+0.00573, +0.01255] | 5/5 | 小而稳定的单专家信号 |
| C2 − C0，弱 VFT AUPR | +0.00773 | [+0.00457, +0.01091] | 5/5 | 同上 |
| C3 − C0，弱 VFT AUROC | +0.01415 | [+0.01203, +0.01707] | 5/5 | 低于预注册 +0.030 |
| C3 − C0，弱 VFT AUPR | +0.01348 | [+0.00864, +0.01795] | 5/5 | 低于预注册 +0.025 |
| C3 − C4，弱 VFT AUROC | −0.00041 | [−0.00092, +0.00011] | 2/5 | 真实时间顺序未胜过置换容量对照 |
| C3 − C4，弱 VFT AUPR | −0.00063 | [−0.00112, −0.00013] | 2/5 | 置换对照略优，非机制收益 |
| C3 − C0，macro-AUPR | −0.00186 | [−0.00263, −0.00101] | 0/5 | 全局性能退化 |
| C3 − C0，非 VFT macro-AUPR | −0.00493 | [−0.00650, −0.00331] | 0/5 | 安全旁路没有完全避免共享表征代价 |
| C3 − C0，弱 VFT recall | +0.04365 | [+0.02841, +0.06238] | 5/5 | 仅为 development 描述性阈值结果 |
| C3 − C0，VFT FPR | −0.00111 | [−0.00302, +0.00127] | 1/5 | 未见平均 FPR 恶化，但不构成独立测试 |

## 4. 预注册门槛判定

通过：C3/C4 同参数量；所有收据未使用测试；非 VFT macro-AUPR 降幅未超过 `0.010`；SawFM 相对 C1 的 AUPR/弱召回未越过非劣界；弱召回增益与 FPR 约束均满足描述性门槛。

未通过：

1. C3 相对 C0 的弱 VFT AUROC `+0.01415 < +0.030`。
2. C3 相对 C4 的弱 VFT AUROC 为 `−0.00041`，且仅 2/5 种子为正。
3. C3 相对 C0/C4 的弱 VFT AUPR 均未达到 `+0.025`；C3−C4 AUPR CI 下界小于零。
4. C3 的 macro-AUPR 相对 C0 为负，未达到 `+0.005`。

因此 development gate 为 **false**，不得将该结构冻结为候选，也不得授权 H2 external 或 test-v3。

## 5. 专家归因与实现核验

15 个冻结 checkpoint（C2/C3/C4 × 5 seeds）均完成 development-only 贡献审计：

- 对每一个 checkpoint，`final_vft_logit = global_vft_logit + vft_expert_logit` 的最大绝对残差均不超过 `1.1920929e-7`；目标标签加性约束成立。
- 非 VFT logit 未被 VFT 专家直接修改；故性能变化不是由错误写入其他标签造成。
- 该 H2 development 的五个种子均没有 strong-VFT-positive 层。审计已将其显式记为可选层缺失，而没有伪造/零填充强正统计；弱 VFT 与 VFT-negative 两个必要层均存在。

这也给出重要限制：H2 可以检验弱 VFT 对负样本的时序可分性，却无法直接证明“强 VFT 保护”或完整的强/弱/负三层 VBF 训练机理。

## 6. 失败归因与停止规则

物理 H1b 已证明真实时间顺序相对置换含有可检测信息；本轮 C2 相对 C0 也给出小幅、5/5 同向的开发信号。但 C3 没有优于同容量 C4，且 macro-AUPR 稳定下降。这排除了“仅加入真实时间戳专家即可带来统一六标签收益”的解释。

当前证据更支持以下诊断：

1. 现有加性 VFT 专家在共享 BCE 目标下只带来很弱的局部排序扰动；其量级不足以跨过 C3/C4 因果对照。
2. C3 的共享训练代价超过该弱局部收益，表现在 5/5 macro-AUPR 均为负增益。
3. 缺失 strong-VFT 层意味着 H2 不能测试完整可见性三层训练目标，不能据此把 C2 的小信号提升为 VTSE-VBF 证据。

按预注册失败规则，禁止继续搜索时间戳置换方式、统计 ROI、专家宽度、损失权重、阈值或训练轮数；也禁止使用 H2 external、旧 locked test、test-v2 或 test-v3 来挽救此版本。

## 7. 工程验证与下一步

以下聚焦测试通过：`5 passed in 1.66s`。

- `tests/test_multipulse_vft_expert.py`
- `tests/test_summarize_route_a_h2_multipulse_development.py`
- `tests/test_audit_route_a_h2_multipulse_development_artifacts.py`

实现过程中修正了贡献审计的项目导入路径，并将“strong VFT 层缺失”改为显式的可选层记录；弱正/负样本层仍是强制检查。此修复只影响开发期只读归因审计，不影响任一训练 checkpoint、预测、校准、阈值或预注册 gate。

下一轮只能基于新的、可证伪机制：先构造同时包含弱正、强正和 matched hard-negative 的身份隔离物理源，以验证是否存在足够大的强/弱时序差异；只有该源上的低容量标签条件统计能胜过同容量置换对照，才允许重新设计统一 VTSE-VBF 候选。
