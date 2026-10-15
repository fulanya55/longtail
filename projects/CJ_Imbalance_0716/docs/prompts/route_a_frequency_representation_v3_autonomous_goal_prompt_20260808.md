# Goal Prompt：Route A Frequency Representation V3 全新身份受限表征适配自动试验

## 1. Goal Objective

在完全不复用 `dual-axis-frequency-development-v2` evaluation predictions、旧 R2 evaluation、locked test、test-v2 或 test-v3 做任何选择的前提下，创建全新 identity 的 `dual-axis-frequency-representation-v3`，验证“受限 stage4 表征适配 + label-conditioned evidence projection + feature-space hard-negative ranking”是否能建立可靠的 label-frequency 主效应。

本 Goal 不以获得正结果为完成条件。Terminal S/P/N 均为合法终局；若 frequency gate 失败，必须停止 visibility、joint 和 confirmation，不得在同一 evaluation source 调参重试。

## 2. Fixed Environment

- 项目：`E:\LJQ_code\CJ_Imbalance_0716`
- Python：`D:\anaconda3\envs\PY10\python.exe`
- 新数据根：`E:\LJQ_code\干扰数据\dual-axis-frequency-representation-v3`
- 新输出根：`E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_frequency_representation_v3`
- 参考负终局：`outputs/route_a_frequency_reconstruction_v2/terminal_N_closure`
- 所有活动代码、数据和输出使用 E 盘；C/D 路径只能作为历史文本证据。

## 3. Immutable Evidence Boundary

1. 禁止读取 v2、旧 R2、locked test、test-v2、test-v3 的 sample-level predictions、labels 或 metrics 做模型、loss、adapter、pair、seed、threshold 或 source 选择。
2. 只允许读取 v2 Terminal N 的 aggregate summary、receipt、hash 和失败结论。
3. 创建与全部历史 sample-seed/receiver-seed namespace 无重叠的新 seed range，并保存生成器与 JSON range overlap audit。
4. 在正式数据生成前完成 preregistration；生成后不得改变主候选、损失权重、rank、margin、adapter rank、early key 或 gate。
5. 新数据必须 train-only source 后再按 identity 派生 optimization/early/calibration/threshold/evaluation；paired views 不得作为独立样本计数。

## 4. Scientific Hypothesis

V2 失败表明：冻结 128 维 B0 feature 上，线性 head-only cRT 无法恢复 label-frequency 条件边界；标签 Jaccard hard pairs 与 random pairs 无差异。新假设 H-R 为：

> Label-frequency 的可恢复信息存在于 B0 stage4/feature projection 附近，但被共享投影压缩。仅在 stage4 后加入低秩受限适配，并通过每标签 evidence projection 与 feature-space matched negatives 训练，可改善尾标签局部可分性，同时保持 FB 和非目标标签安全。

## 5. Stage 0：审计与预注册

自动执行：

1. 核验 PY10、CUDA、RTX 5080、dirty worktree、v2 Terminal N hashes 和 test-v3 seal metadata。
2. 审计所有已知 generator/config/audit seed ranges，分配新 sample-seed range；无法证明无重叠则 abort。
3. 写正式 preregistration，冻结数据规模、角色、manifest、模型、参数、seeds、预算、early key、bootstrap、controls、gate、失败动作和成本上限。
4. 写协议一致性单元测试，明确：普通 BCE comparator、DB、head-only v2、adapter、label projection、hard-pair 和 random-pair 的损失公式不可混淆。

## 6. Stage 1：新 identity 数据与难度修复

### 6.1 数据

- 生成全新的 VH/VL paired views，I/Q `[2,3000]`，六标签 63 个非空组合。
- 建议每组合至少 500 identity，evaluation 不少于 3,150 identity。
- VH 所有分量 30 dB；VL 仅复合样本 SawFM 20 dB，其余分量 30 dB。
- 保存 sample seed、receiver seed、component seeds、labels、parameters、power truth 和 SHA256。

### 6.2 Roles

固定 optimization/early/calibration/threshold/evaluation = 60/10/10/10/10，按每组合分层；所有角色 identity 零重叠，VH/VL 同 identity 同角色。

### 6.3 Frequency manifests

- FB：六标签正样本精确平衡；
- Label-IF10/20/50：主证据；每标签必须同时有正负样本，固定 sample count，实际 IF 误差 <=5%；
- Combo-IF10/20/50：必须增加困难度，禁止所有正式 baseline 在 smoke/readiness 上 macro-AUPR 已超过 0.985；若天花板门失败，只允许在正式训练前重建数据难度。
- 所有 manifest 固定 2,400 draws/epoch，保存 target/actual counts、combo counts、seed、indices 和 SHA256。

## 7. Stage 2：Representation Identifiability Gate

在任何正式端到端候选训练前，只使用 optimization/early：

1. 训练 B0 Focal 与 DB 基线。
2. 冻结 B0，缓存 stage3、stage4 和 128 维 global feature。
3. 比较以下训练-only probes：
   - global-128 linear probe；
   - stage4 pooled probe；
   - rank-4 adapter probe；
   - label-conditioned projection probe；
   - feature-nearest hard-negative probe；
   - random-neighbor control。
4. identifiability gate：Label-IF20 三个最少标签的 early AUPR 平均提升 `>=+0.015`，5 seeds 至少 4 个正向；feature-neighbor 必须胜 random-neighbor；FB 平均下降不得超过 0.005。

未通过则直接 Terminal N，不运行完整候选。

## 8. Stage 3：Formal Models

共享 backbone：B0 `dfms_iq`，stem 与 stage1–3 冻结。

正式 arms：

1. `G0_B0_Focal`；
2. `G1_B0_DB`；
3. `G2_PlainBCE_cRT`：冻结 128 feature、普通 BCE、同预算；
4. `G3_V2_HeadOnly_RBC`：复现 v2 head-only 机制，作为失败机制对照；
5. `G4_Rank4_Adapter`：只训练 stage4 后 rank-4 residual adapter；
6. `G5_FCLPA`：主候选，rank-4 adapter + 六个 label-conditioned evidence projections + feature-space ranking + anchor；
7. `G5_no_adapter`；
8. `G5_no_label_projection`；
9. `G5_random_neighbor`；
10. `G5_capacity_matched_MLP`。

主候选结构建议：

- global safety path：原 128 维 feature；
- adapter：`h' = h + U(Vh)`，rank=4，残差初始为 0；
- 每标签 evidence projection：`e_k = GELU(P_k h')`，维度 16；
- logit：原线性 logit + 有界 label residual，初始 residual scale=0；
- feature hard negative：目标标签缺失、其他标签 Jaccard 位于同一预注册分层，同时在冻结 training feature 中 cosine 最近；
- 禁止输入 label frequency、weak mask、power、pair id 或 oracle parameters。

损失固定：

`L = L_label_balanced + 0.20 L_feature_rank + 5e-4 L_anchor + 1e-4 L_adapter_norm`

margin、权重和 rank 不做网格搜索。普通 BCE comparator 必须真正使用普通 BCE，不得复用 label-balanced 实现。

## 9. Hyperparameters

- seeds：五个全新固定 seeds；
- B0：AdamW，backbone lr `5e-5`，classifier lr `5e-4`，wd `1e-4`；
- adapter/projection：AdamW，adapter lr `1e-5`，label projection/head lr `3e-4`，wd `1e-4`；
- effective batch 64；cached probe/head batch 256；
- core max 36 epoch/patience 8；adapter max 60/patience 10；head max 100/patience 15；
- gradient clip 5.0；
- early key：Label-IF20 三尾标签 mean AUPR，macro-AUPR 次级，FB safety 第三；
- 每个 manifest 每 epoch 2,400 uniform draws。

## 10. Formal Gate

G5 相对逐 seed/cell `best(G0,G1,G2,G3,G4)` 必须全部满足：

1. Label-IF20 macro-AUPR 平均差 `>=+0.010`；
2. Label-IF20 三个最少标签平均 AUPR 差 `>=+0.020`；
3. Combination-IF20 平均差 `>=+0.003`，且场景无天花板；
4. receiver-identity paired bootstrap 5,000 次，95% CI lower `>0`；
5. 至少 4/5 seeds 的 Label-IF20 和合并 primary 均为正；
6. FB macro 差 `>=−0.005`；任一标签平均下降不超过 0.010；
7. IF10/IF50 各至少 3/5 seeds 非负；
8. 主候选胜 `G5_random_neighbor`、`G5_no_adapter`、`G5_no_label_projection` 和容量匹配 MLP；
9. 参数增量 <=5%，batch-1 latency 增量 <=15%，batch-64 throughput 降幅 <=15%。

任一主门失败：写 `frequency_representation_failed`，保存全部负结果，关闭当前机制，不进入 visibility/joint。

## 11. Conditional Dual-axis Chain

只有 frequency representation gate 全部通过，才允许：

1. 在同一新身份协议上重新训练固定 RawSaw 和 equal-capacity/information-destruction controls；
2. 通过 visibility gate 后运行新 2×2；
3. 区分 modular complementarity 与 positive interaction；
4. 冻结全部代码、模型、阈值和 claim，创建完全新 identity confirmation-v3，一次性验证。

## 12. Required Artifacts

- preregistration/resolved config/source hashes/seed overlap audit；
- roles 与全部 manifests；
- compile/focused pytest/GPU smoke/readiness receipts；
- per-seed checkpoint/history/prediction；
- probe、per-label、per-regime、bootstrap、controls、cost；
- calibration/threshold 仅在 ranking gate 通过后执行；
- 中文报告、machine-readable evidence ledger、terminal S/P/N receipt；
- 所有产物固定 `locked_test_used=false,test_v2_used=false,test_v3_used=false`。

## 13. Start Instruction

现在从 Stage 0 开始。先只读核验 v2 Terminal N、PY10、CUDA、worktree、test-v3 seal 和全部历史 seed lineage；写新 seed namespace 与正式 preregistration。在 preregistration 完成前不得生成正式数据或训练模型。之后按硬门自动推进，直到 Terminal S、P 或 N。
