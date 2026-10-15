# Goal Prompt：Route A 频次轴重建与双轴受控再组合自动试验

## 1. Goal Objective

在不复用已消费 R2 evaluation、locked test、test-v2 或 test-v3 的前提下，创建全新 identity 的 `dual-axis-frequency-development-v2`，验证 **Risk-Bounded Co-occurrence Classifier Retraining（RB-C cRT）** 是否能建立达到最小科学效应的 frequency-only 主效应。只有 frequency-only 严格通过后，才允许在同一新 identity-paired 协议中重新训练固定结构 RawSaw，检验 frequency 与 SawFM visibility 的正交组合是否稳定优于两个单轴。

本 Goal 不以“必须得到正结果”为完成条件。严格负结论、停止回执和论文范围收缩同样是合法终局。

## 2. Fixed Environment and Live Paths

- 项目：`E:\LJQ_code\CJ_Imbalance_0716`；
- Python：`D:\anaconda3\envs\PY10\python.exe`；
- 新数据根：`E:\LJQ_code\干扰数据\dual-axis-frequency-development-v2`；
- 参考代码：`E:\LJQ_code` 下与数据生成、B0、长尾 manifest、RawSaw、校准、阈值和统计相关程序；
- 当前冻结证据：`outputs/route_a_dual_axis_next/terminal_N_closure`；
- 新输出根：`outputs/route_a_frequency_reconstruction_v2`；
- 新报告根：`docs/plans`、`docs/results`、`docs/reports`。

所有活跃代码、数据和输出使用 E 盘路径。历史 C/D 路径只能作为报告中的历史证据文本，不得作为新运行源。

## 3. Immutable Evidence Boundary

1. `route_a_dual_axis_next` 的 R2 evaluation 已消费，不得读取其 sample-level predictions 进行新方法、loss、adapter、threshold、seed 或 source 选择。
2. locked test、test-v2 和 test-v3 的样本、标签、预测、指标禁止访问；只能核验 seal/receipt metadata。
3. 不得改写或覆盖 terminal N bundle、历史 manifests、checkpoint、hash 或报告。
4. 根 0716 是平衡源；任何长尾主张必须来自显式派生、可重建、保存 SHA256 的 manifest。
5. high/low power views 必须按 receiver_noise_seed 和完整 component seeds 作为同一 identity 配对，不能拆分到不同角色。
6. evaluation source 每个预注册候选集合只允许使用一次；结果后改机制必须生成新的 identity source。

## 4. Scientific Hypotheses

### H-F：频次轴主假设

当前 DB 失败的主要原因不是 backbone 容量不足，而是多标签共现条件下，标签计数重权不能稳定分离稀有标签正样本与组合相似 hard negatives。冻结 128 维 B0 representation 后，使用 label-balanced classification、co-occurrence-matched ranking 和风险边界重训练分类器，应在 LT 条件形成稳定 frequency main effect，同时保持 FB 和非尾标签安全。

### H-V：可见度复制假设

只有 H-F 通过后才检验。固定 RawSaw 结构在新 identity VH/VL 上应再次改善 SawFM weak-vs-negative 排序，并战胜 equal-capacity 和信息破坏 controls。

### H-J：联合系统假设

只有 H-F 和 H-V 分别通过后才检验。RB-C cRT 与 RawSaw 的组合应在 LT-VL 上稳定优于两个最佳单轴。若 combined 优于 best single-axis 但 factorial interaction 不为正，只允许称“正交模块互补”，不得称“协同”。

## 5. Stage 0：只读审计与预注册

自动执行：

1. 核验 PY10、CUDA、当前 worktree 和 terminal N receipts；
2. 核验 test-v3 seal 的 `access_authorized=false`、`analysis_allowed=false`、`used=false`；
3. 读取生成器、现有数据 seed lineage 和 R2 数据 hashes，只做 overlap audit；
4. 分配与历史完全不重叠的新 receiver/component seed range；
5. 写正式预注册，冻结数据规模、角色、manifest、模型、参数量、优化器、seeds、预算、选择规则、统计、门槛和失败动作；
6. 在预注册完成前不得生成正式数据或训练模型。

若无法证明新 seeds 与历史不重叠，写 abort receipt，并停止等待用户提供可用 seed namespace。

## 6. Stage 1：新 identity 数据与长尾协议

### 6.1 Paired views

- VH：SawFM 30 dB；
- VL：SawFM 20 dB；
- 除 SawFM 功率外，I/Q 长度、receiver noise、组件参数和随机种子完全对齐；
- 每个 identity 只属于一个角色。

### 6.2 Roles

推荐比例 60/10/10/10/10：

- optimization；
- early/model-selection；
- calibration；
- threshold；
- evaluation。

evaluation 不少于 2,500 identities；具体总量根据显存和生成成本在预注册中冻结。

### 6.3 Frequency manifests

- FB safety；
- primary label-LT IF20；
- primary combination-LT IF20；
- secondary IF10 和 IF50；
- 每个 manifest 保存目标计数、实际计数、组合计数、identity ids、生成算法、seed 和 SHA256；
- 禁止把 paired power views 当作独立样本扩充计数。

数据准入门：角色零重叠、view identity 完全对齐、labels 一致、source/manifest hash 完整、LT 实际 IF 与预注册误差不超过 5%。否则只允许修工程问题，不得训练。

## 7. Stage 2：Frequency-only 模型

### 7.1 Shared backbone

保持当前 `dfms_iq`：

- input `[B,2,3000]`；
- width 16；
- feature 128；
- dropout 0.2；
- classifier `Linear(128,6)`；
- 不扩宽 backbone，不新增全局 attention。

### 7.2 Formal arms

1. `F0_B0_Focal`：当前公平基线；
2. `F1_B0_DB`：当前失败机制的参考复现；
3. `F2_Ordinary_cRT`：冻结 B0 feature，按普通 BCE 同预算重训 head；
4. `F3_RBC_cRT`：主候选；
5. `F3_no_rank`；
6. `F3_no_anchor`；
7. `F3_random_pair_rank`：参数和 pair 数匹配，但 negative 随机选择。

主结果必须比较 `F3_RBC_cRT` 与 `best(F0,F1,F2)`；消融只解释机制，不能替代主比较。

### 7.3 RB-C cRT

先训练 F0 representation，再冻结 backbone 和 feature projection。F3 从 F0 classifier 权重初始化，只训练六标签 head：

`z = W h + b, h in R^128`

`L = L_label_balanced + 0.25*L_cooccurrence_rank + 1e-3*L_anchor`

- `L_label_balanced`：每个标签在一个 epoch 内正样本和负样本总权重各为 0.5，再对六标签平均；
- `L_cooccurrence_rank`：对每个稀有标签，将正样本与“其他五标签 Jaccard 相似度最高、但该标签缺失”的 hard negative 配对；使用 `softplus(0.2-z_pos+z_neg)`；
- `L_anchor`：`||W-W0||²/(||W0||²+eps) + 0.1||b-b0||²/(||b0||²+eps)`；
- inactive/NaN 参数不得进入任何损失；
- 模型输入中不得加入标签频次、真实 weak mask、功率、pair id 或其他 oracle 信息。

### 7.4 Optional low-rank structural candidate

只允许一个预注册 secondary：stage4 projection 上 rank-4 low-rank adapter，其他 backbone 冻结。它必须与 head-only 候选同时在 preregistration 中定义，并只用 early role按固定规则选择一个候选进入唯一 evaluation。不得在看到 evaluation 后启用 adapter 救援。

### 7.5 Hyperparameters

- seeds：五个全新固定 seeds；
- F0/F1：AdamW，backbone lr `5e-5`，classifier lr `5e-4`，wd `1e-4`；
- cRT heads：AdamW，lr `1e-3`，wd `1e-4`；
- optional adapter lr `1e-5`；
- effective batch 64；缓存 feature 训练 head 时 batch 256；
- max epoch：core 36、head 100；
- patience：core 8、head 15；
- gradient clip 5.0；
- early key：primary IF20 label-LT 与 combination-LT macro-AUPR 平均，FB safety 为次级 lexicographic key；
- 不进行 loss-weight、margin 或 adapter-rank 网格搜索。

## 8. Frequency Formal Gate

排序指标先于校准和阈值。receiver-identity paired bootstrap 至少 5,000 次。

F3 相对 `best(F0,F1,F2)` 必须同时满足：

1. primary IF20 label-LT 与 combination-LT 的 macro-AUPR 平均差 `>=+0.005`；
2. 95% CI 下界 `>0`；
3. 至少 4/5 seeds 正向；
4. 三个最少标签的平均 AUPR 差 `>=+0.010`；
5. FB macro-AUPR 差 `>=-0.005`；
6. 任一标签 AUPR 差 `>=-0.010`；
7. IF10/IF50 方向不反转，至少各 3/5 seeds 非负；
8. 主候选必须胜 ordinary cRT 和 random-pair control，证明收益不只是重新训练或额外 pair loss。

若任一主门失败：

- 标记 `frequency_reconstruction_failed`；
- 保存所有负结果和 receiver bootstrap；
- 不训练 RawSaw、不进入 2×2；
- 关闭当前 classifier-decoupling 频次路线；
- Goal 以可靠负终局完成。

## 9. Stage 3：Visibility Replication（条件执行）

仅当 Frequency Formal Gate 全部通过时执行。

- RawSaw 结构冻结：raw I/Q、hidden 16、kernels 7/31/63、top-k 16、dropout 0.1；
- 在新 optimization/early identities 上重新训练权重，不复用 R2 权重；
- controls：equal-parameter envelope-only、phase-only、time-shuffled、ordinary core SawFM head；
- calibration：paired VH+VL calibration role 上 per-label Platt，`C=1e6`、lbfgs、max_iter 1000；
- threshold：threshold role 上 `0.05:0.01:0.95`，FPR `<=0.20` 下最大 weak recall。

Visibility gate：

- weak AUROC 相对 core `>=+0.020`，CI 下界 `>0`；
- weak recall `>=+0.050`；
- candidate point FPR `<=0.20`，且 bootstrap 97.5% 上界 `<=0.21`；
- strong recall 差 `>=-0.01`；
- 4/5 seeds 正向；
- 胜 equal-capacity 和全部关键信息破坏 controls。

失败则停止 combined，保留 frequency-only 结果。

## 10. Stage 4：新 2×2 与联合判定（条件执行）

正式模型：

- `M0 = F0_B0_Focal`；
- `MF = 通过的 F3_RBC_cRT`；
- `MV = M0 + RawSaw`；
- `MFV = MF + RawSaw`。

固定 estimands：

- `Delta_F`：MF-M0；
- `Delta_V`：MV-M0；
- `Delta_joint`：MFV-best(MF,MV)；
- `Interaction = MFV-MF-MV+M0`。

联合系统门：

1. Delta_F 和 Delta_V 重复通过各自门槛；
2. LT-VL 上 Delta_joint macro-AUPR `>=+0.005`，CI 下界 `>0`；
3. 4/5 seeds 正向；
4. 任一标签 AUPR 降幅不超过 0.01；
5. strong recall 和 FPR 安全；
6. MFV 胜 equal-capacity、threshold-only 和信息破坏 controls。

论文语言分支：

- Delta_joint 通过且 Interaction CI 下界 `>0`：允许 `[A] synergistic dual-axis candidate`；
- Delta_joint 通过但 Interaction 不通过：只允许 `[A] orthogonal modular complementarity`；
- Delta_joint 不通过：不允许联合模型主张，只保留通过的单轴。

## 11. Stage 5：Fresh Confirmation（条件执行）

只有 Stage 4 通过后：

1. 冻结 source code、model、checkpoint 选择、calibrator、threshold、seeds、统计和 claim；
2. 创建完全不重叠的 `dual-axis-confirmation-v2`；
3. 不使用 confirmation 做任何选择；
4. 一次性运行五 seeds 或预注册的 frozen ensemble；
5. 主要效应至少保留 development 的 70%，门槛方向不反转，FPR 和非目标安全通过；
6. 生成 terminal/result/abort receipt。

confirmation 失败则降级为 development-only，不允许继续调参后重复 confirmation。

## 12. Engineering and Reproducibility Loop

每个 Stage 自动执行：

1. 写 preregistration 和 resolved config；
2. 实现最小代码；
3. `py_compile`；
4. focused pytest；
5. MAT/HDF5 contract；
6. CPU/GPU forward/backward；
7. short smoke；
8. artifact/hash/role audit；
9. 正式运行；
10. 轮询进程和 receipt，不能凭 checkpoint 宣布完成；
11. receiver-level paired bootstrap、逐 seed、逐标签、LT severity、组合阶数、FPR、校准和成本分析；
12. 更新 evidence ledger 和中文报告。

允许自动修复：路径、编码、字段、shape、mask、device、determinism、receipt 和明确实现 bug。算法门失败后禁止在同一 evaluation source 调参重试。

## 13. Required Artifacts

每个正式 Stage 必须保存：

- preregistration、resolved config；
- source/role/identity/manifest SHA256；
- overlap audit；
- environment、CUDA、PyTorch 和 dirty-worktree 说明；
- checkpoint、history、prediction；
- calibration、threshold、per-seed/per-label metrics；
- receiver bootstrap samples 和 summary；
- 参数、MACs、latency、memory、throughput；
- completed/result/abort receipt；
- `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`；
- 中文专业报告和 machine-readable evidence ledger。

## 14. Goal Terminal Conditions

满足任一终局即可 complete：

### Terminal S：统一双轴候选完成

- frequency、visibility、joint 均通过；
- controls 胜出；
- fresh confirmation 通过；
- 安全、成本和复现 bundle 完整；
- 主张明确区分 modular complementarity 与 synergy。

### Terminal P：部分单轴成果完成

- frequency 或 visibility 至少一个正式通过；
- combined 未通过或另一轴失败；
- 完成该单轴的 controls/confirmation 或明确 development-only 边界；
- 论文范围收缩完整。

### Terminal N：可靠负结论完成

- frequency reconstruction 未达到门槛，或后续 visibility/joint 被严格否定；
- 所有失败证据、统计、controls 状态、receipt 和适用边界完整；
- 明确关闭对应机制，不在同一 source 无限调参。

## 15. Start Instruction

现在开始 Stage 0。先只读核验 terminal N bundle、PY10、worktree、test-v3 seal、生成器与历史 seed lineage，写新的 frequency-development-v2 预注册。不得立即训练，不得读取 R2 sample-level evaluation predictions，不得创建 confirmation。Stage 0 完成后按硬门自动继续，直到 Terminal S、P 或 N。

