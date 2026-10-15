# 长线 Goal Prompt V10：从单通道失败闭环到独立可观测量验证

将以下内容作为复合干扰联合识别与参数感知研究代理的持续目标执行。
工作区固定为 `C:\LJQ_code\CJ_Inbalance_0724`，Python 固定使用
`D:\anaconda3\envs\PY10\python.exe`。目标不是持续增加网络容量，而是用
**独立可观测量的物理充分性—最小机制对照—身份隔离验证—系统安全验收**闭环，
判断下一项创新是否真实存在。

## 0. 当前状态与不可突破的边界

当前状态为 `contract_required`，不是 `blocked_by_compute`，也不是“继续微调”。
全部既有结果均须按原证据角色保留，且始终保持：

```text
test_generated=false
test_opened=false
locked_test_used=false
```

以下结论已经冻结，不能以新随机种子、不同阈值、不同 loss 权重、更多 epoch 或
更深 router 重跑：

1. MFT→ISRJ 单通道扩展线已经满足 V7 的三机制停止规则：P3n 的条件 margin、
   P3o-T 的逆延迟 support、P3o-U 的局部 phase/STFT dynamics 均正式失败。
2. SawFM 周期的 FFT-ACF 解析专家在 OOD confirmation 中因分离器伪小峰而失去稳定
   排序；整数倍峰、跨窗硬修正、固定修正规则和 ACF loss 已失败。不能把峰规则或
   同源置信度微调包装为新方法。
3. VFT Doppler 的宽有符号网格、固定相位增量、同源 router，以及显式 VFT 分量隔离
   oracle 均未建立当前观测契约下的充分性。不得再调 Doppler 网格、相位窗、延迟对齐、
   router 或相同候选估计器的损失。
4. P2h 的预测组合、专家-主干分歧和回退已经是现有的有界可靠性对照；只把同一 I/Q
   统计改名为“选择性预测”不构成独立信息机制。

因此，**不得虚构第二个观测、把 generator component seed、精确成分、真实标签、
真实 JNR 或 paired-power 的其他视图作为部署输入**。同一 `receiver_noise_seed` 的
功率视图始终为成组反事实，不是独立样本。

## 1. 唯一待验证的科学问题

在真实可取得的、与现有单通道 compound-I/Q 信息不同的测量契约下，是否存在一个
可部署的观测量，使 SawFM 周期或 VFT Doppler 的**可识别性/风险**相对于冻结单通道
候选产生可证伪的增量？

可接受的观测契约必须由用户或实际采集系统明确提供，至少写清：

- 观测来源（例如第二接收通道、受控 PRI/时间多视图、双斜率探测参考、独立测得的
  延迟参考）；
- 采样时序、带宽、噪声和与原 I/Q 的物理关系；
- 推理时可用性及成本；
- 不依赖 component seed、标签、参数真值和 generator oracle 的证明；
- 新数据生成/采集是否真实实现该契约，而非从同一波形事后重排得到伪视图。

在这些字段没有给出前，代理只能完成审计、文档和代码接口准备，**不得生成“模拟新
观测”的正式数据或声称已完成新的机制验证**。

## 2. 自动推进状态机

每轮只处于下列一个状态；不要跳过前置门。

| 状态 | 允许动作 | 退出条件 |
| --- | --- | --- |
| `contract_required` | 审计既有失败边界，记录候选数据契约与缺失字段 | 用户/采集系统提供完整独立观测契约 |
| `contract_audited` | 冻结 generator/data-adapter 合同，审计泄漏与物理独立性 | 新观测不是 oracle/派生伪视图，且角色可按 identity 隔离 |
| `oracle_screen` | 仅在新的 train-only identities 做物理充分性/风险可分性筛查 | C1 明确超过 C0 与强控制，才准许训练 |
| `implemented` | 预注册、实现、单测、smoke、hash freeze | 代码/数据 preflight 通过 |
| `development` | 一次性三模型种子正式开发，保存预测与 bootstrap | 全部晋级门通过才生成 confirmation |
| `confirmation` | 不改任何方法，在两个新 identity roles 一次性确认 | 全部门通过才可形成候选；仍不得使用 locked test |
| `frozen_failure` | 写出失败机制、哈希和禁止回调范围 | 只有另一份独立观测契约才能开启新线 |

## 3. 新观测进入训练前的 oracle 筛查

先写入 `configs/` 的 preregistration，再生成全新、仅 train-role 的身份池。以
`receiver_noise_seed` 为 identity；角色间、历史数据与确认角色的交集必须为 0。

### 3.1 SawFM：辨识风险而非重写旧峰规则

比较：

- `S0`：冻结 P2n/P2h 的原单通道 period candidate 与安全回退；
- `S1`：同一候选加独立观测所产生的可部署 quality/risk measure；
- `S2`：维度、成本匹配但与新测量失配的控制；
- `S3`：随机置乱的新观测控制。

在不训练深度估计器之前，先检查 S1 对 `|period error|` 与 catastrophic error 的排序是否
比 S0/S2/S3 强。主门必须在三个新身份块和每个规定功率/区间同时满足：风险 AUPRC
相对随机基线至少提高 `0.15`，80% 固定 coverage 的 accepted catastrophic rate 至少降低
`30%`，且同 coverage 的 accepted-set MAE 不高于 S0 的 `0.95×`。所有比较使用 identity-
grouped paired bootstrap（至少 2,000 次）；任何控制可复制 S1 或任一 cell 失败，即停止。

### 3.2 VFT：先证明新的可识别候选，而非更深路由

比较：

- `V0`：冻结正频 grid/常数回退；
- `V1`：由新观测独立构造的 Doppler candidate；
- `V2`：错误参考或失配观测；
- `V3`：保留维度/成本的置乱观测。

V1 在 ID、near-low、near-high、far-negative、far-negative-high 五区间中必须全部满足：
相对 V0 MAE 降低至少 `20%`、Pearson 至少提高 `0.20` 且绝对 Pearson `>=0.50`；V1 相对
V2/V3 的 paired-bootstrap 95% CI 必须分别支持 MAE 下降与 Pearson 上升。若任一区间失败，
该观测不能进入分离器、router 或全 12 参数系统。

上述 Oracle 允许以标签只作离线评分，绝不允许以真实误差在候选之间作推理选择。

## 4. 仅在 Oracle 通过后的最小开发试验

1. 在数据生成前冻结 C0/C1/C2/C3、模型种子、优化器、loss、batch、epochs、warmup/
   scheduler、checkpoint 和 calibration 规则，并记录 SHA-256。
2. C0 是冻结安全系统；C1 是新观测的最小专家/风险头；C2 是等参数等 MAC 的无新观测
   控制；C3 是新观测失配/置乱控制。模型在预测时不能读取 oracle、标签或真实参数。
3. 参数损失只在 `label=1 AND parameter_mask=1 AND finite(target)` 上计算。inactive NaN
   绝不零填充。
4. 每个候选使用三个模型种子；选择与 development roles 严格分离。development 只打开一次。
5. 同时验收目标参数、full-12 predicted-gate NMAE、miss penalty、worst combination、
   每功率 FPR/recall、paired bootstrap、参数量/FLOPs/峰值显存与 batch 1/64 延迟。GPU
   非独占时延只标记“未验证”，不能报告性能数值。

开发晋级至少要求：三个 seed 的目标 NMAE 对 C0 降低 `>=10%` 且 Pearson `>=0.30`；
相对 C2/C3 三 seed 同方向、bootstrap CI 支持；分类安全、MFT AP、macro-AUPR、其他参数、
miss penalty 和 worst combination 均不劣于 preregistered tolerance。任何一项失败，冻结该
观测，不生成 confirmation，不在同一 identities 调参。

## 5. 可写入论文的证据尺度

- Oracle screen 通过：只能说新观测具有物理充分性；不能说端到端方法有效。
- development 通过：只能说存在身份隔离的开发证据。
- 两个新 identity confirmation 都通过：才可说新观测带来可部署的受限增益。
- 只有所有方法、数据、阈值、成本与 confirmation 完全冻结并获得用户单独授权后，才可
  一次性申请 locked test；本 Goal 不授权生成、打开或使用 locked test。

## 6. 每轮必须交付

报告当前状态、唯一问题、C0/C1/强控制、身份/功率配对审计、网络与训练细节、逐 seed/
逐 cell/逐参数结果、CI 和效应量、失败机制、禁止回调范围、工件绝对路径及
`test_generated/test_opened/locked_test_used`。不得把 smoke、开发、确认或历史 reference
合并成一项结论。

## 7. 立即动作

立即将 P3n、P3o-T、P3o-U 的 MFT→ISRJ 线记录为关闭；审计 P2o–P2r 与 P2u–P2w 的
SawFM/VFT 可观测性边界。若不存在真实独立观测契约，停留在 `contract_required`，向用户
索取观测/采集合同，而不是自动生成第四个同源单通道实验。
