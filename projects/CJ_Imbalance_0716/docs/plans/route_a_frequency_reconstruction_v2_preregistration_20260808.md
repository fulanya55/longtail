# Route A Frequency Reconstruction V2 正式预注册

日期：2026-08-08  
分支：`frequency_reconstruction_v2`  
数据角色：全新 identity、development-only  
主假设：`H-F`  
当前授权：Stage 0 审计、Stage 1 数据生成/manifest、Stage 2 frequency-only；RawSaw、联合 2×2 和 confirmation 均为条件授权。

## 1. 不可变证据边界

1. 活跃根为 `E:\LJQ_code\CJ_Imbalance_0716`，Python 为 `D:\anaconda3\envs\PY10\python.exe`。
2. 上一轮 `route_a_dual_axis_next` 已封存为 terminal N。其 sample-level evaluation predictions 不得用于本轮任何选择。
3. locked test、test-v2、test-v3 的样本、标签、预测和指标禁止访问；只允许核验 receipt/seal metadata。
4. 新数据、角色、manifest、模型、loss、预算、early key、统计和失败动作在正式数据生成前冻结。
5. 本轮不扩宽 backbone，不复活 generic attention/backbone screen，不搜索 DB 权重、ranking weight、margin、anchor weight或 adapter rank。

## 2. 新数据合同

- 数据根：`E:\LJQ_code\干扰数据\dual-axis-frequency-development-v2`；
- 生成器：`experiments/generate_route_a_frequency_development_v2.m`；
- generator base seed：`3930000000`；
- receiver identity range：`3930000001..3930025200`；
- 63 个非空六标签组合，每组合 400，共 25,200 identities；
- label order：FMB、SawFM、SinFM、MFT、VFT、ISRJ；
- VH：所有激活分量 30 dB；
- VL：复合样本中的 SawFM 为 20 dB，其余激活分量 30 dB；
- 两视图共享完整 component seeds、receiver noise、labels、parameters、sample order；
- train-only，不生成 test；existing policy 为 `error`。

角色按每组合固定配额切分，role seed `3930100001`：

| Role | 每组合 | 总数 |
|---|---:|---:|
| optimization | 240 | 15,120 |
| early | 40 | 2,520 |
| calibration | 40 | 2,520 |
| threshold | 40 | 2,520 |
| evaluation | 40 | 2,520 |

所有角色必须 identity 零重叠，VH/VL 同 identity 必须属于同一角色。

## 3. Frequency manifests

所有 manifest 只从 optimization role 派生，source split SHA256 固定。不同样本量通过每 epoch 固定 2,400 uniform draws 统一更新预算；paired power views 不增加样本计数。

### 3.1 FB safety

- seed `3930100010`；
- 2,400 identities；
- 六标签正样本计数精确为 `[1200,1200,1200,1200,1200,1200]`。

### 3.2 Label-LT

固定 tail order：FMB > SinFM > MFT > VFT > ISRJ > SawFM。target counts：

| Severity | Seed | Target counts（原标签顺序） | Target IF | 预期选中数 |
|---|---:|---|---:|---:|
| IF10 | 3930100011 | `[1800,180,1136,717,452,285]` | 10 | 约 1,835 |
| IF20 primary | 3930100012 | `[1500,75,824,453,249,137]` | 20 | 约 1,425 |
| IF50 | 3930100013 | `[800,16,366,167,77,35]` | 50 | 约 789 |

label counts 相对 target 容差 5%；实际 IF 与 target IF 相对误差不得超过 5%。

### 3.3 Combination-LT

| Severity | Seed | min samples/combo | 预期选中数 |
|---|---:|---:|---:|
| IF10 | 3930100021 | 3 | 约 5,948 |
| IF20 primary | 3930100022 | 3 | 约 4,845 |
| IF50 | 3930100023 | 3 | 约 3,851；离散后 IF 约 48 |

combo IF 相对 target 容差 5%。

## 4. Stage 2 唯一科学假设

当前 DB 的主要失败不是 backbone 缺少容量，而是按标签计数重权无法区分稀有标签正样本与“其他标签组合相似、目标标签缺失”的 hard negative。冻结 B0 的 128 维 representation 后，RB-C cRT 应通过 label-balanced head、co-occurrence ranking 和风险边界形成稳定 frequency main effect。

## 5. 模型与公平预算

Shared backbone：`dfms_iq`，input `[B,2,3000]`、width 16、feature 128、dropout 0.2、`Linear(128,6)`，参数 257,511。

训练 seeds：`260821,260822,260823,260824,260825`。

正式 arms：

1. `F0_B0_Focal`；
2. `F1_B0_DB`；
3. `F2_Ordinary_cRT`；
4. `F3_RBC_cRT`；
5. `F3_no_rank`；
6. `F3_no_anchor`；
7. `F3_random_pair_rank`。

F0/F1 在每个 training manifest 上从同一初始 state 开始。每 epoch 固定 2,400 draws，effective batch 64，max 36 epoch，patience 8；AdamW，backbone lr `5e-5`、classifier lr `5e-4`、weight decay `1e-4`、clip 5.0。F0 使用 gamma_pos 0/gamma_neg 3 的 Focal；F1 使用当前 DB 默认实现。两者沿用上一正式协议的 reduction：五个非 SawFM 标签均值加 0.2 倍 SawFM 项。

cRT 使用对应 manifest/seed 的最佳 F0 checkpoint，冻结所有 feature 参数并缓存 128 维 feature。head 从 F0 classifier 初始化，AdamW lr `1e-3`、weight decay `1e-4`、batch 256、max 100 epoch、patience 15、clip 5.0。

本预注册不启用 optional stage4 adapter；若 head-only 正式失败，不能在同一 evaluation source 启用 adapter 救援。

## 6. RB-C cRT loss

`L = L_label_balanced + 0.25 L_cooccurrence_rank + 1e-3 L_anchor`

- `L_label_balanced`：对每个标签，epoch 内 positive 和 negative 的总权重各归一为 0.5，再对六标签平均；
- `L_cooccurrence_rank`：每个 positive 与其他五标签 Jaccard 最大、但当前目标标签为 0 的 negative 配对，`softplus(0.2-z_pos+z_neg)`；
- `L_anchor`：相对 F0 head 的归一化权重/偏置漂移；
- `F3_random_pair_rank` 使用完全相同 pair 数，但在合法 negatives 中由冻结 seed 均匀抽取；
- 模型输入不得含 frequency、weak mask、power、pair id 或其他 oracle 字段。

## 7. Regimes 与选择规则

Primary：Label-IF20、Combo-IF20、FB。所有七个 arms 均运行。

Secondary：Label-IF10、Combo-IF10、Label-IF50、Combo-IF50。只运行 F0/F1/F2/F3，用于方向稳健性，不参与主候选选择。

训练和 early selection 只使用 VH 30 dB，隔离 visibility 因素。checkpoint early key 为 macro-AUPR，SawFM AUPR 为固定 tie-break；不得使用 calibration、threshold 或 evaluation 选择模型。

正式 evaluation 对预注册 arm 集合只运行一次。排名指标基于 raw logits；之后才允许在独立 calibration/threshold roles 上执行 Platt 和阈值分析。

## 8. Frequency hard gate

F3 相对 replicatewise/per-seed `best(F0,F1,F2)` 必须全部满足：

1. Label-IF20 与 Combo-IF20 macro-AUPR 差的五 seed 平均 `>=+0.005`；
2. receiver-identity paired bootstrap 5,000 次，95% CI 下界 `>0`；
3. 至少 4/5 seeds 的 primary 两 cell 平均差为正；
4. 每个 primary cell 的三个最少标签平均 AUPR 差再跨 cell/seed 平均 `>=+0.010`；
5. FB macro-AUPR 差 `>=-0.005`；
6. primary/FB 任一标签的平均 AUPR 差 `>=-0.010`；
7. IF10 与 IF50 的 label/combo 平均方向各至少 3/5 seeds 非负；
8. F3 在 primary 上分别胜 F2 Ordinary cRT 与 F3 random-pair control，平均差均 `>0` 且至少 4/5 seeds 正向。

若任一条件失败，写 `frequency_reconstruction_failed` terminal receipt，停止 RawSaw 和 2×2，不在同一 evaluation 上调参。

## 9. 条件式后续

只有 frequency hard gate 全部通过，才授权 Goal Prompt 中 Stage 3–5：新 identity RawSaw replication、完整 controls、新 2×2、joint/interaction 判定和 fresh confirmation。后续所有结构、阈值和门槛仍以 Goal Prompt 为准，且必须另写 execution addendum 后才能运行。

## 10. 必需产物

- Stage 0 environment/worktree/seal/seed audit 与 receipt；
- generator dry-run/formal receipts；
- 两视图文件/manifest SHA256；
- roles 和 IF10/20/50 manifests；
- code/config/source hashes；
- per-seed checkpoints/history/predictions；
- calibration/threshold；
- receiver bootstrap samples/summary；
- 参数/MACs/latency/memory/throughput；
- 中文报告和 machine-readable evidence ledger；
- 所有产物固定 `locked_test_used=false,test_v2_used=false,test_v3_used=false`。

