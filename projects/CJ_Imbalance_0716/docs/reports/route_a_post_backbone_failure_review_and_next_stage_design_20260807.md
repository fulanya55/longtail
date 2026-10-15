# Route A Backbone 科学停止后的失败复盘、创新证据审计与下一阶段设计

日期：2026-08-07  
证据边界：上一阶段为 fresh identity-isolated development-only；下一阶段全部内容为 `proposed / not validated`。test-v3 继续密封。

## 1. 执行摘要

上一阶段不是“网络改得不够多”，而是四类不同 backbone 增强都无法将弱分量局部信号稳定转化为三目标统一收益。M1 重新分配了频谱偏好，M2 的跨层残差被主动压制，M3 只帮助 VFT/ISRJ 并伤害 FMB，M4 的局部极值在 4/5 seed 被学习为负证据。共同结论是：当前弱分量瓶颈主要位于**成分可辨识性和监督归因**，而不是普通容量、池化或感受野不足。

此前设立的“双轴长尾”思想仍有研究价值，但 VTSE-VBF 的当前实现证据不足：C3 未战胜同参数 C4，不能排除容量解释；H3 只能证明一个冻结标签特定专家组合在标准新 identity 上可复现，H5o 又给出了联合 OOD 反例。下一阶段应保留“频次偏置与可见性退化分离”的问题定义，放弃“统一强化 backbone 就能解决可见性”的实现假设，转向新的同 identity 成分反事实观测协议。

推荐的新假设是 `IP-CVE`：用同 receiver、同 nuisance、同噪声背景下的 high/low/absent 成分三元组，直接监督“目标成分带来的边际证据”，再以小型标签特定、风险有界的 readout 接回冻结 B0。第一步必须是无最终分类器训练的可辨识性准入；准入失败则停止，不进入模型阶段。

## 2. 上一阶段逐模块失败复盘

| 模块 | 实验现象 | 直接证据 | 根因判断 | 已排除解释 |
|---|---|---|---|---|
| M1 抗混叠 | macro-AUPR 小升，但弱 AUROC 不升；SinFM/VFT 退化 | ΔAUPR `+0.004655`，Δ弱 AUROC `-0.000354` | 固定低通改变调制频谱偏好，未产生目标成分归因 | 不是参数不足；M1 无新增训练参数 |
| M2 CSFP | 几乎复现 B0 | Δ弱 AUROC `+0.000447`；有效残差约 `±0.004–0.007` | 跨层信息没有对应监督收益，优化器选择忽略它 | 不是链路未接通；梯度、参数更新和 smoke 均正常 |
| M3 EPB | VFT/ISRJ 部分改善，FMB/SinFM 退化 | Δ弱 AUROC `+0.006936`，Δ弱 recall `+0.008218` | 长感受野具有标签特异性，但共享残差产生跨调制负迁移 | 不是所有标签都缺少长程结构；MFT 几乎不受益 |
| M4 local stats | 弱召回 5/5 下降 | Δ弱 recall `-0.003301`；4/5 seed 为负残差 | 局部大响应主要由强分量或 nuisance 主导，max/top-k 缺少成分特异性 | 不是 top-k=8/16 未比较；已一次 validation 选 k=16 |

分类器、Platt 和 FPR 阈值是指标放大器，但不是这轮根因。所有候选使用相同分类器、校准和阈值协议，且排序型弱 AUROC 同样没有达到门槛。训练不足也不是主要解释：多数最佳 epoch 位于 33–36，但末段变化很小，B0 与候选的收敛形态接近。

## 3. 跨轮次统一根因链

```text
弱成分在复合背景中只贡献很小边际变化
        ↓
普通标签 BCE/Focal 只告诉模型“标签存在”，不告诉它证据位于何处、相对哪个背景成立
        ↓
强正样本和最显著共现分量支配共享表征与分类边界
        ↓
抗混叠/跨层/长感受野/top-k 读取到的是混合显著性，而非目标成分边际证据
        ↓
共享模块出现三种结果：被门控压成近零、产生标签特定收益、或引发跨标签负迁移
        ↓
校准与阈值无法恢复已经没有稳定排序的信息
```

这条链与此前证据一致：VTSE-VBF 的可见性候选未战胜同容量 C4；H0v 的原始时序诊断接近随机；P1/P2 能支持 MFT/VFT，却无法解决弱 ISRJ 在多 nuisance 背景中的归因。问题更像“条件边际证据不可辨识”，而不是“缺少一个更强的通用卷积模块”。

## 4. 既有创新点证据审计

| 创新主张 | 当前支持 | 当前反证/缺口 | 决定 |
|---|---|---|---|
| 双轴长尾：频次偏置 + 可见性退化 | 标签/组合频次与同标签 weak/strong 的机制定义不同；频次损失不能解释可见性交换 | 0716 根数据平衡，必须显式派生 LT；尚缺完整 2×2 因子证明 | 保留并重写为问题定义与实验框架 |
| VTSE：标签内可见性证据建模 | 历史部分弱指标为正；M3 显示 VFT/ISRJ 标签特定结构 | C3 未战胜 C4；backbone M1–M4 均未形成统一收益 | 放弃当前 VTSE 实现，改为反事实成分归因 |
| VBF：有界安全融合 | MFT shared-decision、部分标签冻结接回显示安全性价值 | H5o 出现 FPR/阈值运输失败；不能单独作为核心创新 | 保留为二级安全机制，不作为主创新 |
| RawSaw/标签物理专家 | RawSaw SawFM、ISRJ periodicity、VFT Ridge、MFT shared decision 有窄范围正证据 | 标签来源和泛化范围不同，不能合并成统一因果证明 | 冻结为参考基线和辅助组件 |

### 最强允许表述

频次不平衡与标签内可见性退化应被独立控制；当前标签特定专家在若干冻结协议中有效，但通用 VTSE-VBF 或强化 backbone 尚未获得同容量、信息破坏和跨分布证据链。

### 禁止表述

- 已证明统一双轴长尾网络；
- VTSE-VBF 已排除容量解释；
- 更强 backbone 已解决弱分量；
- H3 可以替代因果控制或 H5o/OOD；
- 当前开发证据足以打开 test-v3。

## 5. 创新点重构决定

主创新从“更强的通用复数 backbone”改为：

> 在频次偏置与可见性退化严格解耦的复合干扰长尾协议中，使用同 identity 的成分存在/功率反事实三元组，学习目标标签相对匹配背景的边际证据，并通过风险有界的标签特定接回保持强正与非目标安全。

建议名称：`Identity-Paired Counterfactual Visibility Evidence (IP-CVE)`。

创新由三部分构成：

1. 双轴因果化数据协议，而不是把功率称作第三种样本长尾；
2. matched component-absent 反事实监督，而不是普通 high/low consistency；
3. 只有在证据可辨识并战胜信息破坏控制后，才允许风险有界接回。

## 6. 下一阶段科学假设

对目标标签 `k∈{MFT,VFT,ISRJ}`，若 `x_low(k)` 与 `x_absent(k)` 保持所有背景、噪声和 nuisance 参数一致，仅改变目标成分存在性，那么其表征或物理观测差应包含比 unmatched/permuted 控制更稳定的目标边际证据。该证据应随 `x_high(k)>x_low(k)>x_absent(k)` 单调变化，并且不系统改变其他标签的预测。

若 matched residual 在冻结特征上都不能战胜置换控制，则当前合成观测协议不支持该创新，后续训练任何 evidence head 都缺乏科学依据。

## 7. 新数据协议（proposed / not validated）

生成新的 `route_a_counterfactual_visibility-development-v2`，identity 与 0716、development-v1/v3、H3/H5o、P1/P2 全部零重合。每个基础背景生成：

- `x_absent(k)`：移除目标成分 k；
- `x_low(k)`：加入 k，目标相对功率落入预注册低可见区间；
- `x_high(k)`：仅提高 k 的功率，其他参数不变。

必须共享：receiver_noise_seed、其他干扰参数、载频/带宽/时延 nuisance、噪声 realization 和组合背景。三元组以 group identity 切分，不能跨角色。

建议角色：fit、source-selection、validation、early、calibration、threshold。另生成一次性 `counterfactual_visibility-confirmation-v1`，只在最终冻结后使用。

同时派生并 hash 记录 frequency-balanced 与 label/combo-LT manifest，构成：

| | 高可见度 | 低可见度 |
|---|---|---|
| 频次平衡 | FB-VH | FB-VL |
| 频次长尾 | LT-VH | LT-VL |

这样才能验证频次方法和可见性方法是否分别作用于对应轴。

## 8. 模型、损失和训练设计（proposed / not validated）

### P0：冻结可辨识诊断

先冻结 B0，比较 global、Stage3/Stage4 temporal 与预注册物理观测中的 matched residual：

`r_k = feature(x_present(k)) - feature(x_absent(k))`

控制包括 unmatched receiver、component permutation、time permutation 和同维随机投影。P0 不训练最终分类器、不选阈值、不做融合。

### P1：小型标签特定 evidence readout

仅在 P0 通过的标签/来源上训练：

- B0 stem 与大部分 backbone 冻结；
- 每标签低秩 adapter 或线性 evidence map；
- 基础 B0 logits 保留；
- `Δlogit_k = 0.20*tanh(g_k)*e_k(x)`；
- 推理时只需要单个样本，absent/high 配对只用于训练。

建议总损失：

`L = L_cls + λ1 L_weak-absent-rank + λ2 L_high-low-order + λ3 L_non-target-invariance + λ4 L_hard-negative`

其中：

- `L_cls`：固定 Focal 基线；
- `L_weak-absent-rank`：要求 matched low 的目标 evidence 高于 absent；
- `L_high-low-order`：要求 high > low，但不要求二者特征完全一致；
- `L_non-target-invariance`：移除/改变 k 时，其他标签 logits 应基本不变；
- `L_hard-negative`：同组合但 k 缺失的 matched background 排名约束。

初始建议只允许一组预注册权重，例如 `λ1=1.0, λ2=0.5, λ3=0.25, λ4=0.5`；不得同源扫描。分类器、Platt 和阈值规则沿用 B0，避免同时改变多个层次。

## 9. 实验矩阵与控制

| 组别 | 目的 |
|---|---|
| B0 Focal | 统一基础对照 |
| DB/LA 或最强频次基线 | 验证频次轴 |
| ordinary high-low consistency | 排除普通配对一致性解释 |
| IP-CVE | 目标候选 |
| equal-capacity no-counterfactual | 排除参数量解释 |
| unmatched-pair | 验证 identity 匹配必要性 |
| component-permuted | 验证目标成分对应关系 |
| time-permuted | 验证真实时序/局部信息 |
| no-absent / no-order / no-invariance | 三项机制消融 |

正式评价必须同时报告 2×2 双轴四个单元、逐标签、弱/强/负、组合阶数、hard negatives、receiver cohort，以及参数/MACs/延迟/显存。

## 10. 硬门槛与停止规则

P0 准入：相对置换控制的三目标平均 weak-vs-absent AUROC `≥+0.030`，CI 下界>0，4/5 seed 为正，任一标签不低于 `-0.010`，validation-to-early gap `<0.15`。

P1 晋级：相对 B0 和 equal-capacity，macro-AUPR `≥+0.005`、弱 AUROC `≥+0.020`、弱 recall `≥+0.025`；弱指标均至少 4/5 seed 为正；FPR `≤0.20`；强 recall 下降不超过 `0.01`；目标 AUPR与安全标签不显著退化；并战胜 unmatched/component/time-permuted 控制。

失败即关闭对应标签或整个机制，不得调配对定义、margin、损失权重、阈值或 adapter 宽度后在同一来源重试。只有全部通过后才能冻结并生成 confirmation；confirmation 上不训练、不校准、不选阈值。

## 11. SCI 证据链

需要依次证明：

1. 可观测：matched counterfactual 确实含有目标边际信息；
2. 特异：信息被 identity/component/time 破坏后收益消失；
3. 有效：在固定 FPR 下提高弱排序和弱召回；
4. 安全：强正和非目标不退化；
5. 双轴独立：频次方法与可见性方法在 2×2 因子中作用不同；
6. 可泛化：新 identity confirmation 保持方向；
7. 外部有效：真实/半实物/HIL 至少一项验证。

在完成第 1–6 项之前，不能声称 IP-CVE 为可靠创新；没有外部数据时，论文必须明确限定为仿真协议内结论。

## 12. 推荐论文叙事

论文不再以“提出更强复数 backbone”为主线，而以“复合干扰中频次稀缺与可见性退化是两类不同偏差”为问题贡献；方法贡献是 identity-paired component counterfactual supervision；VBF 只作为风险控制实现；RawSaw 和标签物理专家作为既有标签特定参考。

如果 P0 或 P1 失败，应收缩为负结果论文或方法边界章节：证明 generic backbone strengthening、普通可见性头和局部显著性统计不足以解决成分归因。

## 13. 下一步执行顺序

1. 审计生成器能否严格复现同背景 absent/low/high 三元组；
2. 预注册全新 identity range、角色和所有 permutation 控制；
3. 只生成小规模 train-only smoke source，验证波形差异只来自目标成分；
4. 生成正式 development-v2 并完成 identity/hash audit；
5. 执行 P0 五 seed 可辨识诊断；
6. P0 通过才实现 P1 readout 与固定损失；
7. 执行公平基线、信息破坏控制和 2×2 双轴实验；
8. 全部通过才冻结 confirmation bundle；
9. confirmation 通过后再考虑真实/HIL 与论文终局实验；
10. test-v3 仍不授权，除非独立总目标另行满足并由用户明确授权。

## 总裁决

继续追求：双轴问题定义 + IP-CVE 成分反事实可见性证据。  
冻结为参考：B0、RawSaw、ISRJ aligned-periodicity、MFT shared-decision、VFT Ridge 和 H3 标准源结果。  
永久关闭：M1–M4 backbone 变体、当前 VTSE/C3 对 C4 未晋级机制、在已消费来源上的调阈值/扩宽/重命名重试。  
当前状态：方案已设计，尚未验证，不能写成创新已成立。
