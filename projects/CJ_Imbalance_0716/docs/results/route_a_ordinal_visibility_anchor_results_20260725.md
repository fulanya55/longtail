# 路线 A：有序可见度三锚点实验结果（2026-07-25）

## 1. 结论

本轮不晋级，也不解冻主干。显式同身份功率排序能够稳定学出 `root > 40 > 45 > 50 dB` 的可见度坐标，但三锚点单调偏置无法把该坐标转化为统一阈值下的弱分量召回；即使直接输入真实功率等级的 oracle 锚点也未达到预注册上限门槛。当前失败点已经从“可见度是否可预测”定位为“可见度到类别概率的映射结构不成立”。

## 2. 实验协议

- 环境：`D:\anaconda3\envs\PY10\python.exe`，正式运行设备为 CUDA。
- 数据：0716 六类协议；同一 `receiver_noise_seed` 的 root/40/45/50 dB 视图按身份绑定。
- 冻结模块：F2、H2、稳定 S2、C2 固定融合 `alpha=0.80`。
- 每种子每组合 30 对；expert-scale、visibility-fit、power-threshold 三个互斥身份角色。
- 种子：2041、2042、2043；最大 300 epoch；AdamW，学习率 `1e-2`，weight decay `1e-3`。
- 普通域与 40/45/50 dB 共用一个阈值，所有域 hard-negative FPR 约束为 0.20。
- locked test 未使用，身份交集为 0。

对照：

- A0：上一轮 22 参数 V2 可见度仿射校准器。
- A1：16 参数三锚点模型，不使用排序损失。
- A2：16 参数三锚点模型，加入同身份六对 pairwise 排序损失。
- A3：直接输入真实 root/40/45/50 等级的 oracle 三锚点结构上限。

## 3. 主要结果

| 方法 | 普通召回 | 普通 FPR | 三档平均召回 | 40 dB | 45 dB | 50 dB | 最大功率 FPR | 平均 oracle gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 Previous V2 | 0.7230 | 0.2048 | **0.4724** | **0.7101** | **0.5403** | 0.1668 | 0.1176 | 0.3123 |
| A1 Anchor No Ordinal | 0.7060 | 0.2311 | 0.3493 | 0.6146 | 0.3557 | 0.0776 | 0.0828 | 0.3956 |
| A2 Ordinal Anchor | 0.6519 | 0.1886 | 0.0721 | 0.2028 | 0.0136 | 0.0000 | 0.0035 | 0.6938 |
| A3 Oracle Power Anchor | 0.6318 | 0.1892 | 0.4074 | 0.5977 | 0.3736 | **0.2509** | 0.0446 | 0.3871 |

A2 在三个种子上的三档平均召回分别为 0.1156、0.0333、0.0674，3/3 决策门均失败。A3 只有种子 2041 达到 0.5748，跨种子均值 0.4074，低于预注册的 0.60 上限门槛。

## 4. 机制结果

A2 的排序机制不是失败点：

- threshold role 平均 pairwise accuracy：0.9744。
- early-development 平均 pairwise accuracy：0.9755。
- early-development 平均 Kendall tau：0.9511。
- A1 early-development pairwise accuracy：0.5189。
- A2 相对 A1 的排序增益约 0.4566。

三个种子的 A2 early 可见度均值均严格递减：

- 2041：1.302、0.466、-0.374、-1.112。
- 2042：0.984、0.342、-0.236、-0.784。
- 2043：0.644、0.217、-0.178、-0.505。

因此，有序监督确实学到了跨功率可见度坐标，而且在独立 early-development 身份上泛化稳定。

## 5. 失败原因

### 5.1 排序目标与分类目标发生方向冲突

三锚点设计把低可见度映射为更大的正偏置，但最终学习到的 A2 偏置增量仅约 `[0, 0.35, 0.67]`，不足以抵消 C2 正样本随功率降低产生的 logit 下移。同时该偏置也施加到低可见度 hard negatives，迫使训练通过更高统一阈值或压缩分数来保持 FPR，结果是正负概率间隔同步缩小。

平均正负概率间隔从 A0 的 40/45/50 dB `0.401/0.359/0.239`，降为 A2 的 `0.225/0.162/0.090`。这解释了 A2 虽然排序正确，却在统一阈值下几乎全部拒绝弱正样本。

### 5.2 三锚点只表达功率域偏移，不能表达类别证据

可见度等级同时作用于正样本和缺失 SawFM 的 hard negatives。仅凭功率/可见度锚点增加 logit，无法判断“弱 SawFM 证据”与“同功率下真正缺失 SawFM”。A3 已经给出直接证据：真实功率输入使 50 dB 召回从 A0 的 0.1668 提高到 0.2509，但三档均值仍只有 0.4074，未超过 A0，也未达到 0.60 上限门。

### 5.3 单调正偏置和单一全局尺度过度受限

A3 为保持 hard-negative FPR，学习到较大的锚点偏置（低锚点约 3.76--3.81）、更大的全局尺度（约 2.08--2.26）和负全局偏置。这种相互抵消表明，一个全局尺度加一个功率偏置无法同时对齐四个域的正负分布；需要直接建模证据可靠性或允许 unknown/reject，而不是继续增加锚点数量。

### 5.4 不是主干未解冻导致的当前失败

冻结 S2 在各功率仍有较高 oracle recall，A2 的 early 排序也稳定泛化。当前失败发生在排序坐标之后的概率映射层。此时解冻主干会把明确的决策层结构错误重新混入表征漂移，缺少晋级依据。

## 6. 决策与下一步计划

本轮 promotion gate 失败；保留 A0 Previous V2 为当前冻结决策基线，不采用 A1/A2/A3，不解冻最后 stage。

下一阶段进入预注册的 A3 失败分支：显式 `positive / negative / unknown` 决策，而不是继续做功率偏置校准。

建议实验：

1. 冻结 F2/H2/S2/C2，保留 A0 分数。
2. 使用 power-threshold 身份定义区间：在 hard-negative FPR 不超过 0.20 时确定正阈值；另设较低阈值，仅用于标记明显负样本；中间区域输出 unknown。
3. 对 40/45/50 dB 分别报告：确定正召回、unknown 覆盖率、unknown 中正样本比例、确定负误杀率、选择性风险和 coverage-risk 曲线。
4. 与 A0 单阈值、oracle 分档阈值比较；不使用真实功率作为部署输入，只将 oracle 分档阈值作为可辨识上限。
5. 晋级条件：确定正样本 FPR 不超过 0.20；50 dB 的“确定正召回 + unknown 正覆盖”不低于 0.60；确定负误杀不超过 0.10；选择性风险跨三种子稳定；locked test 继续冻结。
6. 若 selective/unknown 仍不能稳定分离弱正和 hard negative，则停止校准路线，下一步才允许在 stage2 时序证据图上训练标签条件的弱证据/缺失证据二分类头。

## 7. 产物与验证

- 设计：`docs/plans/route_a_ordinal_visibility_anchor_plan_20260725.md`
- 实现：`experiments/run_route_a_ordinal_visibility_anchor.py`
- 模型：`cjlt/models/evidence_readout.py` 中 `OrdinalVisibilityAnchorCalibrator`
- 汇总：`outputs/tapc/route_a_ordinal_visibility_anchor/route_a_ordinal_visibility_anchor_summary.json`
- 每种子保存 checkpoint、history、threshold search、predictions、metrics、source/identity audit 和 receipt。
- 测试：PY10 下全量 `171 passed`；仅有既有 AMP 弃用警告。
