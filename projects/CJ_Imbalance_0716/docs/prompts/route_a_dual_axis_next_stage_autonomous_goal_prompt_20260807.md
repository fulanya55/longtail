# Route A 双轴长尾下一阶段：自主改进、试验与验证 Goal Prompt

## 使用说明

将以下 Prompt 完整交给 Codex。它要求 Codex 创建一个持续执行的科研 Goal，自动完成只读审计、预注册、实现、测试、正式开发试验、统计分析、失败复盘和受限分支切换。

本 Goal 不要求强行得到正结果。若主机制被可靠否定，完整保存负证据、关闭错误路线并形成可发表的研究边界，也属于 Goal 完成。

```text
请创建并持续执行以下科研 Goal。不要设置人为 token budget。在目标达成或形成可靠负结论前保持 Goal active；每一阶段必须按硬门槛推进，
不得因结果不理想降低门槛、重复利用已查看的数据、重新打开已消费测试集，或把已关闭机制换名后继续调参。

Goal objective：

> 将 Route A 当前约 3.0/10 的“双轴长尾复合识别”证据成熟度推进到下一个科学上允许的层级：优先审计并验证部署可获得、非 oracle 的新物理观测，
> 在 H0 可辨识性通过后开发频次—可见性解耦的标签特定网络和正式 F×V 2×2 证据链；若没有合格新观测，则自动收缩到 SawFM 范围的双轴2×2验证，
> 完成新 identity confirmation、工程复现和 SCI 证据收口，或以严格控制证明该范围同样不能成立。

## 1. 角色与工作原则

你是雷达复合干扰多标签识别、复数 I/Q 网络、长尾学习、弱分量检测、概率校准、统计验证和科研证据治理方向的长期自主研究代理。

你的任务不是无限搜索网络，而是最大化每轮实验的信息增益：

1. 先验证观测中是否存在信息，再训练模型；
2. 先验证单轴，再验证组合；
3. 先在 development 选择，再在全新 confirmation 冻结验证；
4. 每轮只有一个主要机制变量；
5. 所有失败、禁止执行、未运行和不确定状态必须保留；
6. 正结果必须战胜 equal-capacity、信息破坏、阈值和选择偏差解释；
7. 负结果达到预注册停止条件时必须关闭机制，不得无限调参。

## 2. 固定环境、路径与权限

- Python：`D:\anaconda3\envs\PY10\python.exe`
- 主项目：`E:\LJQ_code\CJ_Imbalance_0716`
- 主要数据根：`E:\LJQ_code\干扰数据`
- 0716 历史数据：`E:\LJQ_code\干扰数据\复合干扰数据0716`
- E 盘其他项目只读参考；未经用户明确授权，不得修改当前 Goal 范围外的项目。
- 新数据只能写入 `E:\LJQ_code\干扰数据\` 下本 Goal 独立命名目录。
- 新代码、计划、结果、报告和 outputs 只能写入 `E:\LJQ_code\CJ_Imbalance_0716` 的独立文件/目录。
- 历史报告、manifest、receipt、冻结快照和 SHA 中的 C/D 路径属于不可变证据，不得机械替换。
- 当前 worktree 可能包含用户的大量未提交修改；先审计状态，只改本 Goal 新增文件及明确需要的活动代码，保护无关修改。
- 使用 GPU 前必须在 PY10 中验证 PyTorch/CUDA、MAT/HDF5 读取、forward、backward、短训练和产物写入。

封存边界：

- 历史 locked test 已消费；不得重跑、调参、重校准或重新选择。
- test-v2 已消费；不得重跑或用于新候选选择。
- test-v3 必须始终保持 `access_authorized=false`、`analysis_allowed=false`、`used=false`。
- 只允许读取 test-v3 的 seal/manifest 元数据以核验封存状态，不得读取 I/Q、标签、预测或指标。
- 本 Goal 不具备自动打开任何 sealed/locked test 的权限。
- Route B 参数估计和半监督证据不得合并为 Route A 识别证据。

## 3. 开始前必须完整读取的证据

至少完整读取并交叉核对：

1. `docs/reports/route_a_current_technical_state_failure_and_long_horizon_analysis_20260807.md`
2. `docs/reports/route_a_dual_axis_long_tail_evidence_gap_and_roadmap_20260807.md`
3. `docs/results/route_a_ipcve_goal_negative_closure_20260807.md`
4. `docs/results/route_a_ipcve_p0_frozen_observability_results_20260807.md`
5. `docs/reports/route_a_ipcve_post_p0_extended_observation_admission_20260807.md`
6. `docs/results/route_a_raw_cross_seed_locked_test_results_20260726.md`
7. `docs/reports/route_a_frequency_group_visibility_invariance_results_20260803.md`
8. `docs/results/route_a_backbone_stage1_b0_results_20260806.md`
9. `outputs/tapc/route_a_backbone_b0_development_v1/experiment_config.json`
10. `outputs/tapc/route_a_backbone_b0_development_v1/b0_summary.json`
11. B0 模型代码、训练 runner、test-v3 seal、当前 evidence ledger、相关正式 JSON 和 terminal/result/abort receipt。

必须固定以下事实：

- 根 0716 数据是平衡数据；正式频次长尾必须使用显式派生、可重建、记录 SHA256 的 label-LT/combination-LT manifest。
- 同一 `receiver_noise_seed` 的功率视图属于同一 identity，必须进入同一数据角色。
- 当前 B0 是 `B0_F2_DFMS_Full`，macro-AUPR `0.843746`，三目标 weak AUROC `0.555864`、weak recall
  `0.237524`、strong recall `0.771434`、FPR `0.195215`。
- 历史 RawSaw 候选在一次性 locked test 上 macro-AUPR `0.861894→0.893382`，SawFM weak recall
  `0.155255→0.512175`，SawFM FPR `0.190661→0.135177`；它只直接证明 SawFM 标签特定增量。
- Base-DB 相对 Base-ASL 在 18 个 tail rotation 上平均 macro-AUPR `+0.017870`，18/18 为正；它是频次轴开发基线，不是双轴证明。
- 频次损失不直接接收 weak/strong visibility；DB 可感知共现差异，但不会因 weak/strong 名称互换自动改变目标。
- M1-M4 generic backbone strengthening 已科学停止。
- IP-CVE P0 matched weak AUROC `0.550727`，相对 strongest control `-0.007937`、0/5 seed 正向、0/3 标签通过；
  当前单通道 matched residual、adapter、bounded residual、calibration、threshold 和统一2×2路线已经关闭。
- 当前双轴统一主张的证据成熟度约 `3.0/10`，六个决定性门槛 `0/6` 完整通过。

## 4. 不可违反的科研规则

1. 所有数值结论区分 `[F] confirmed/frozen`、`[A] bounded/development`、`[N] negative/stopped`、
   `[U] running/uncertain`、`[T] proposed/not validated`。
2. 每一正式试验前必须写预注册，固定假设、数据身份、角色、模型、参数量、损失、优化器、种子、预算、选择规则、校准、阈值、对照、指标、门槛和失败动作。
3. smoke、单 seed、短 epoch、shape、import 和 loss 下降只证明可执行性，不能作为性能或创新证据。
4. fit、validation、early、calibration、threshold、evaluation、confirmation 必须 receiver/sample identity 隔离。
5. 用于模型、loss、source、checkpoint、calibrator 或 threshold 选择的数据不能再作为 confirmation/test。
6. 同一正式 evaluation source 只允许一次预注册候选比较。机制变更必须使用新 identity development source。
7. 排序指标未通过时禁止用 Platt、阈值、FPR 放宽或 seed 筛选救援。
8. 不得重新组合 M1-M4、复活 H0e/H0f/IP-CVE、扩大 adapter/attention 或进行无机制 loss-weight 网格搜索。
9. 推理时不得输入精确模拟 absent-view、真实 weak mask、功率、pair id、目标参数或部署不可获得的 oracle 信息。
10. 每次正式运行保存 resolved config、数据/身份 manifest、SHA、环境、代码 hash、checkpoint、history、prediction、统计、成本和 terminal receipt。
11. 没有 completed/result/abort receipt 的实验只能标记 `[U]`，不得汇总为完成。
12. 任何删除或覆盖历史产物的操作都不属于本 Goal 权限。

## 5. Goal 的自动分支逻辑

本 Goal 有一个主路线 S 和一个自动回退路线 R。Stage 0 审计后必须自动选择，不要求用户逐步决定。

### 主路线 S：新物理观测的统一双轴路线

只有发现或能够生成/采集部署可获得、非 oracle、与已关闭单通道机制实质不同的第二观测时进入。合格候选包括：

- 同步主通道 + 经标定辅助接收通道；
- 同一接收机两种固定、部署时可复现的 receiver setting；
- semi-physical/HIL 中由独立硬件链产生的参考观测；
- 真实采集的同步多观测。

以下不合格：模拟器直接提供的精确 absent target、只对同一单通道做数学复制/滤波、利用标签或真实参数构造推理输入、与 MFT delay/LFM、
VFT staggered-PRI、ISRJ periodic/correlation 已关闭机制同构的换名版本。

### 回退路线 R：SawFM 范围的双轴2×2证据路线

若 Stage 0 没有发现合格新观测，禁止停在“等待数据”并继续调单通道 MFT/VFT/ISRJ。自动切换到一个仍有已验证 visibility mechanism 的窄目标：

> 使用全新 identity 和正式 `frequency × SawFM visibility` 2×2 协议，验证 Base-DB/频次方法与 RawSaw/SawFM 可见性专家的主效应、
> 独立性、组合增量和系统安全；主张明确收缩为 SawFM 范围，不外推到 MFT/VFT/ISRJ。

路线 R 不是统一六类双轴方法的替代证明，而是当前最可执行、风险最低的下一阶段正证据路线。

如果路线 R 也未过门槛，则完成可靠负结论：现有频次方法与 RawSaw 在正式共同协议下没有可证实的组合互补性，并据此关闭统一双轴模型主张。

## 6. Stage 0：当前状态、路径与观测准入审计

先只读审计，不生成正式数据、不训练模型：

1. 核验上述必读证据、正式 B0、RawSaw、Base-DB、seal 和 receipt。
2. 使用只读搜索审计 `E:\LJQ_code\干扰数据`、生成器和 E 盘参考代码，列出可能的第二观测候选。
3. 对每个候选记录：物理来源、部署可获得性、同步关系、transfer calibration、训练/推理输入、与已关闭机制的重叠、identity lineage、数据量和缺失字段。
4. 禁止仅因目录名包含 multipulse、beamspace、external 或 HIL 就认定合格；必须读取生成器/manifest/字段证明它确实提供非 oracle 独立观测。
5. 核验 PY10、CUDA、MAT/HDF5、GPU 和输出目录可用性。
6. 检查 dirty worktree，建立本 Goal 文件白名单。

生成：

- `docs/plans/route_a_dual_axis_next_stage0_admission_20260807.md`
- `outputs/route_a_dual_axis_next/stage0_admission/admission_audit.json`
- `outputs/route_a_dual_axis_next/stage0_admission/run_receipt.json`

Stage 0 决策：

- 存在至少一个合格、信息实质不同、可执行的新观测：选择最多一个进入路线 S。
- 无合格观测：记录 `no_admissible_non_oracle_observation`，自动进入路线 R。
- 不得同时正式推进 S 和 R；只允许一个主分支 active。

## 7. 路线 S：新观测统一双轴试验

### S1：新观测 Stage -1 数据合同

创建新的 `dual-axis-multiview-development-v1`，与全部历史 development/test/locked identity 零重叠。每个基础 identity 至少包含 primary 与 reference，
并在可行时包含 target high/low/absent 或明确的弱/强层级。记录同步、transfer calibration、receiver setting、背景、噪声、其他分量和 nuisance。

先生成 dry-run/smoke 数据，验证：

- 通道不是 oracle absent view；
- 主/参考身份和时间同步可验证；
- target 改变不系统改变非目标背景；
- transfer relation 可在不使用 evaluation 标签的条件下估计；
- receiver identity 不跨 role；
- 数据、manifest、SHA、生成/采集配置可重建。

合同失败：关闭该观测，返回 Stage 0 的下一候选；最多审计三个候选，只允许一个正式 materialize。三个均失败则自动进入路线 R。

### S2：零训练 H0 可辨识性

不训练最终分类器、不校准、不选阈值。比较：

- normalized cross-channel coherence；
- magnitude ratio；
- transfer-corrected complex residual；
- lagged phase difference；
- 必要的标签特定固定物理统计。

强制控制：channel swap、unmatched identity、time permutation、component permutation、transfer perturbation、raw-IQ/equal-dimension random projection。

H0 硬门：

- MFT/VFT/ISRJ 平均 weak-vs-absent AUROC 相对 strongest control `>=+0.030`；
- receiver-level paired bootstrap 95% CI 下界 `>0`；
- 至少 4/5 seeds 正向，任一标签不低于 `-0.010`；
- identity binding `>=0.60`，且明显高于 unmatched；
- high > low > absent 顺序高于破坏控制；
- development-to-early 泛化落差 `<0.15`。

分支：

- 三目标通过：进入统一 S3。
- 仅部分标签通过：只为通过标签进入标签特定 S3，统一主张同步收缩。
- 全部失败或平均门失败：关闭路线 S，不训练网络；自动进入路线 R。

### S3：冻结 B0 的低风险多观测证据头

保留当前 B0 global safety path。primary/reference 使用 weight-shared 或明确受控的轻量 complex encoder；跨观测特征只采用 S2 通过的观测量。
每个通过标签使用独立 evidence head，gate 初始化为零，接回限制：

`delta_logit_k = 0.20 * tanh(g_k) * normalized_evidence_k`

固定初始训练顺序：

1. 冻结 B0，只训练 evidence head；
2. 通过后仅解冻 stage4，stage4 lr 不超过 head lr 的 1/10；
3. 再次通过后才允许考虑 stage3；
4. 禁止重做整个 backbone。

建议初始参数：五 seeds；AdamW；head lr `5e-4`；adapter/stage4 lr `5e-5`；wd `1e-4`；effective batch `64`；
max epoch `36`；patience `8`；gradient clip `5.0`。若显存限制使用 batch 32，必须梯度累积并让全部对照保持同 effective batch。

损失：

`L = L_base + 1.0*L_weak_rank + 0.5*L_order + 0.25*L_non_target_invariance + 0.5*L_hard_negative + 0.1*L_bound`

- `L_base` 使用固定公平分类损失；
- `L_weak_rank` 只在 target present、visibility mask 有效、target finite 时计算；
- `L_order` 监督 high > low > absent；
- `L_non_target_invariance` 保护其他标签；
- `L_hard_negative` 使用组合相似但目标缺失样本；
- `L_bound` 限制 residual 和强正/FPR 风险；
- inactive/NaN 目标不得零填充参与损失。

强制对照：B0、equal-capacity no-cross-observation、unmatched、channel-swap、time/component-permuted、no-rank、no-order、no-invariance。

S3 排序晋级门：相对 B0 和 equal-capacity，macro-AUPR `>=+0.005` 且 CI 下界 `>0`；三目标平均 weak AUROC
`>=+0.020`；weak recall `>=+0.025`；至少 4/5 seed 正向；mean FPR `<=0.20`；strong recall 降幅不超过 `0.01`；
任一标签 AUPR 降幅不超过 `0.01`；必须战胜所有关键信息破坏控制。

排序未通过：禁止用 calibration/threshold 救援；关闭 S3 并进入路线 R。

### S4：正式 F×V 2×2 与组合模型

建立新的 hash-recorded manifest：

| 频次 / 可见性 | High visibility | Low visibility |
|---|---|---|
| Frequency-balanced | FB-VH | FB-VL |
| Label/combination-LT | LT-VH | LT-VL |

比较：B0、frequency-only、visibility-only、frequency+visibility、equal-capacity、threshold-only、信息破坏 controls。

固定 estimand：

- `Delta_F`：frequency-only - B0，在 LT-VH/LT-VL；
- `Delta_V`：visibility-only - B0，在 FB-VL/LT-VL；
- `Delta_FV`：combined - best single-axis，在 LT-VL；
- `Delta_Risk`：FB-VH、强正、FPR、非目标标签和常见组合的最坏退化。

组合硬门：

- Delta_F 在 LT 条件至少 4/5 seed 为正，FB 条件 macro-AUPR 降幅不超过 `0.005`；
- Delta_V 通过 S3 弱指标门，VH strong recall 不退化；
- Delta_FV 在 LT-VL 上 macro-AUPR `>=+0.005` 或三目标 weak recall `>=+0.020`，CI 下界 `>0`；
- equal-capacity 和 threshold-only 不能复现；
- exact match、Hamming、FPR、ECE/Brier 和逐标签安全合格。

若两主效应存在但 Delta_FV 不通过，只能声称双轴评估/解耦框架，不能声称联合模型创新。

### S5：全新 confirmation 与冻结

全部结构、权重、checkpoint rule、calibrator、threshold、manifest 规则、依赖和统计脚本冻结后，才创建
`dual-axis-multiview-confirmation-v1`。一次性运行，不训练、不重校准、不移动阈值。

confirmation 必须保持主效应方向，weak AUROC 落差 `<0.15`、macro-AUPR 落差 `<0.02`，并保持全部安全门。失败后关闭候选，不得重跑。

## 8. 路线 R：SawFM 范围双轴2×2试验

### R1：正式问题和数据协议

创建全新 `dual-axis-sawfm-development-v1`，与历史 RawSaw development/locked test、B0 development、test-v2/test-v3 identity 零重叠。

双轴范围明确为：

- F：六标签 label/combo frequency imbalance；
- V：SawFM 在同 identity high/low visibility 下的标签内可见性；
- 不对 MFT/VFT/ISRJ 声称可见性方法有效。

构造 FB-VH、FB-VL、LT-VH、LT-VL。SawFM high/low 必须由预注册物理阈值定义，同一 receiver/noise/background 配对并进入相同 role。
频次 manifest 至少包含一个 label-LT 和一个 combination-LT 设置；开发阶段只允许预先选定一个作为主结果，另一个作为稳健性验证。

### R2：公平模型矩阵

使用当前 B0 谱系和冻结 RawSaw 结构定义，所有模型共享输入、identity split、训练预算、seed、checkpoint、calibration 和 threshold 规则：

1. `M0_B0_Focal`：当前公平锚点；
2. `MF_B0_Frequency`：在 development 上预注册确定的 DB 或其他最强频次方法；
3. `MV_B0_RawSaw`：B0 + 仅替换 SawFM 的 RawSaw 专家；
4. `MFV_B0_Frequency_RawSaw`：frequency method + RawSaw；
5. `MEC_EqualCapacity`：与 RawSaw 增量参数匹配，但不使用 raw phase/time-order 的对照；
6. `MTS_ThresholdOnly`：不改排序，只执行相同校准/阈值协议；
7. RawSaw 信息破坏消融：phase-only、envelope-only、time-shuffled；只使用 development，不能重复历史 locked test。

RawSaw 保持既有结构先验：raw I/Q RMS normalization、hidden 16、kernels 7/31/63、top-k 16、dropout 0.1、约 5,521 参数。
允许为新输入合同做兼容性修复，不允许先看正式结果后改变结构。

### R3：频次、可见性和组合效应

主指标：

- 频次轴：macro-AUPR、tail-label AUPR、tail-combination exact match/Hamming；
- SawFM 可见性轴：weak-vs-negative AUROC、weak recall、strong recall、FPR、ECE/Brier；
- 系统安全：其余五标签逐位不变性/退化、组合阶数、hard negatives、receiver cohort；
- 工程：参数、MACs、GPU/CPU latency、显存、吞吐。

R 路线硬门：

- `MF-M0` 在 LT 条件 macro-AUPR `>=+0.005`，至少 4/5 seeds 正向，FB 条件降幅不超过 `0.005`；
- `MV-M0` 在 VL 条件 SawFM weak AUROC `>=+0.020`、weak recall `>=+0.050`，SawFM FPR `<=0.20`，strong recall 降幅不超过 `0.01`；
- `MFV-best(MF,MV)` 在 LT-VL 上 macro-AUPR `>=+0.005` 或 SawFM weak recall `>=+0.020`，CI 下界 `>0`；
- MFV 必须战胜 MEC 和 MTS；
- 其余五标签任一 AUPR 降幅不超过 `0.01`，不得出现系统性 FPR/强正退化；
- phase/time-order 消融方向必须与历史机制证据一致。

若 MF 与 MV 分别通过、MFV 不通过：形成“两个轴可区分但无需联合结构”的结果，不声称组合创新。
若 MV 不能在新 identity 复现：关闭 RawSaw 跨分布主张，保留历史 locked result 但不得扩展。

### R4：SawFM 范围 confirmation

R3 通过后冻结全部内容，创建 `dual-axis-sawfm-confirmation-v1`。一次性评价，禁止训练、重校准、重选阈值。
confirmation 主效应方向必须保持，macro-AUPR 落差 `<0.02`、SawFM weak AUROC 落差 `<0.15`，FPR/强正/非目标安全门继续满足。

R4 通过后可形成 SawFM 范围的双轴证据链，但必须在标题、摘要和结论中明确 label-scoped，不得写成六类统一 visibility solution。

## 9. Calibration、threshold 与统计协议

所有分支统一：

- 五个正式 seeds；模型臂共享初始化策略、batch identity 顺序、更新步数和 checkpoint 规则；
- calibration role 每标签独立 Platt，建议沿用 `C=1e6`、`lbfgs`、`max_iter=1000`；
- threshold role 搜索 `0.05:0.01:0.95`，在负样本 FPR `<=0.20` 下最大化 weak recall；
- 排序门先于 calibration/threshold；
- 同时报告原始 logits、校准概率、固定 0.5 和受约束阈值；
- receiver-level paired bootstrap，报告 95% CI、effect size、seed signs 和逐标签结果；
- 不只报告平均数，至少报告 weak/strong/negative、组合阶数、hard negatives、receiver cohort 和失败样本。

## 10. 自动实现和运行循环

每一 Stage 自动执行：

1. 读取最新计划、结果 JSON、checkpoint、terminal、prediction、calibration、threshold、receipt 和 evidence ledger。
2. 用状态 `success / bounded / fail / stopped / running-uncertain / proposed` 报告当前证据，禁止乐观合并。
3. 写本 Stage 预注册和唯一科学假设。
4. 实现最小必要代码；优先复用项目基础设施，禁止复活已关闭机制。
5. 依次执行：`py_compile` → focused pytest → MAT/HDF5 contract → CPU/GPU forward/backward → 短训练 smoke → 产物审计。
6. smoke 通过后才运行正式五 seed development。
7. 长进程必须轮询 terminal、进程和 receipt；不能因出现 checkpoint 就宣布完成。
8. 正式运行结束后计算逐 seed/标签/强弱/负/FPR/校准/子群/bootstrap/成本。
9. 对照预注册硬门给出 pass/fail，不得结果后修改门槛。
10. 通过则自动进入下一 Stage；失败则按分支规则关闭、收缩或进入 R，不得无界搜索。
11. 更新 evidence ledger 和中文报告，并给出下一唯一动作。

允许的自动修复仅包括：路径、编码、字段、shape、mask、设备、确定性、receipt 和明确实现 bug。算法机制失败后不允许在同一正式 source 上调整超参数重试。

## 11. 产物规范

统一命名：

- Goal 总计划：`docs/plans/route_a_dual_axis_next_*`
- 预注册：`docs/plans/route_a_dual_axis_next_*_preregistration_*.md`
- 结果：`docs/results/route_a_dual_axis_next_*_results_*.md`
- 审计与总结：`docs/reports/route_a_dual_axis_next_*`
- 机器产物：`outputs/route_a_dual_axis_next/*`
- 冻结件：`frozen/route_a_dual_axis_next_*`
- 新数据：`E:\LJQ_code\干扰数据\dual_axis_*`

每次正式运行至少保存：

- preregistration 和 resolved config；
- dataset/role/identity manifest、overlap audit、SHA256；
- 环境、CUDA/PyTorch、source hash、dirty-worktree 说明；
- 每 seed checkpoint、history、predictions、metrics；
- calibration、threshold、bootstrap 和 effect estimand；
- 参数、MACs、latency、memory、throughput；
- terminal log、started/completed/aborted receipt；
- `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`；
- 中文专业报告和 machine-readable summary。

## 12. Goal 完成标准

满足以下任一终局，可以将 Goal 标记 complete。

### 终局 S：统一或部分标签的新观测证据链完成

1. 至少一个非 oracle 新观测通过 S2；
2. visibility model 战胜 B0、equal-capacity 和信息破坏 controls；
3. 频次轴与可见性轴在正式2×2中作用可区分；
4. 若声称组合创新，Delta_FV 必须通过；
5. 全新 confirmation 一次性通过；
6. 统计、消融、OOD、安全、成本和复现 bundle 完整；
7. 主张范围与实际通过标签一致。

### 终局 R：SawFM 范围双轴证据链完成

1. frequency-only 和 RawSaw visibility-only 在正式共同协议中分别通过；
2. 明确判断 combined 是否具有独立增量；
3. SawFM 范围全新 confirmation 通过；
4. 其余标签安全、统计、成本、消融和复现完整；
5. 论文主张明确限定 SawFM，不冒充统一六类方法。

### 终局 N：可靠负结论完成

1. 新观测候选均不合格或 S2 被严格控制否定；
2. 路线 R 也未能在新 identity 共同协议中复现/形成独立作用；
3. 所有失败证据、控制、统计、receipt 和适用边界完整；
4. 明确关闭统一双轴模型路线，给出可保留的 RawSaw 历史结论和论文范围收缩方案。

不得因为实验困难、运行缓慢或希望更多调参而长期保持伪 active。只有缺少用户才能提供的外部硬件/数据且路线 R 也已完成时，才报告真实外部阻塞。

## 13. 每轮对用户的汇报格式

每轮简洁但必须包含：

1. 本轮实际完成内容；
2. 当前 active 分支和 Stage；
3. 数据角色与是否为 development/confirmation；
4. 关键数值及其对照；
5. 哪些硬门通过/失败；
6. 当前允许和禁止的论文表述；
7. 新产物的 E 盘绝对路径；
8. 下一唯一动作；
9. locked test/test-v2/test-v3 使用状态，默认全部必须为 false。

## 14. 启动指令

现在创建 Goal 并开始 Stage 0。先完整读取必读证据，完成 worktree/PY10/test-v3 seal/B0/RawSaw/Base-DB 审计，并对 E 盘数据和生成器执行
非 oracle 第二观测准入检查。不要立即训练模型。Stage 0 完成后自动选择路线 S 或路线 R，写出预注册，再按门槛持续执行。
```

