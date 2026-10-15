# 路线 A：Stage2 决策可靠性可辨识性筛选结果（2026-07-25）

## 1. 结论

本阶段没有通过晋级门，结论为：

> 冻结 Stage2 时序特征能够预测一部分 A0 决策错误，但没有提供超出已有 10 维 visibility 统计的稳定独立信息；在 50 dB 弱正域，选择性风险仍然不可接受。因此停止 Stage2 reliability 扩展，不训练 `exist / reliability` 双头，不解冻主干，继续保留 C2 作为 SawFM 排序分数、H2/Base-DB 作为正式二元决策基线。

正式决策：`stop_stage2_reliability_extension_keep_c2_ranking_h2_decision`

## 2. 实验是否按预注册执行

- 种子：2041、2042、2043
- 功率：40、45、50 dB
- F2、H2、S2、C2、A0：全部冻结
- A0 阈值：沿用上一阶段冻结值
- probe：L2 Logistic Regression，`C={0.01,0.1,1,10}` 仅由 validation 选择
- early：只用于独立评价
- 身份重叠：0
- locked test：未使用
- 晋级门：0/3 种子通过

## 3. 聚合结果

| 方法 | 输入 | validation AUROC | early AUROC | early AUPR | validation→early 落差 | 可行拒识阈值种子数 |
|---|---|---:|---:|---:|---:|---:|
| R0 | 常数 | 0.5000 | 0.5000 | 0.2829 | 0.0000 | 0/3 |
| R1 | A0 margin | 0.6964 | 0.6706 | 0.4016 | 0.0257 | 3/3 |
| R2 | 10维统计 + margin | **0.7610** | **0.7314** | **0.4559** | 0.0296 | 3/3 |
| R3 | Stage2 时序 + margin | 0.7355 | 0.7014 | 0.4383 | 0.0341 | 3/3 |
| R4 | 统计 + Stage2 时序 + margin | 0.7379 | 0.7109 | 0.4496 | 0.0270 | 3/3 |

关键比较：

- R3 比 R1 高 `+0.0308` AUROC，说明 Stage2 时序确实含有一定可靠性线索；
- R3 比 R2 低 `-0.0300`，说明这些线索没有超过已有 visibility 统计；
- R4 比 R2 低 `-0.0205`，融合后也没有形成独立增益；
- 三个种子的 R4 相对 R2 增量分别为 `-0.0208、+0.0023、-0.0431`，均未达到预注册的 `+0.03`。

因此不能把“AUROC 大于 0.65”单独解释为 Stage2 reliability 可用。它主要复现了已有置信度和可见性统计中的信息，而不是发现了新的时序证据。

## 4. 选择性决策为何仍失败

R4 虽然在三个种子都能从 validation 找到满足覆盖率下限的拒识阈值，但 early 的最坏选择性风险分别为：

| 种子 | 普通覆盖率 | 平均功率覆盖率 | 最大选择性风险 | 结果 |
|---:|---:|---:|---:|---|
| 2041 | 0.7100 | 0.7687 | 0.4034 | 失败 |
| 2042 | 0.7067 | 0.7767 | 0.4094 | 失败 |
| 2043 | 0.7400 | 0.7926 | 0.3290 | 失败 |

失败主要集中在 50 dB：

- seed 2041：覆盖率 0.8980，风险 0.4034，确认漏检率 0.7041；
- seed 2042：覆盖率 0.8550，风险 0.4094，确认漏检率 0.7000；
- seed 2043：覆盖率 0.8245，风险 0.3290，确认漏检率 0.4681。

可靠性 probe 会拒绝一部分普通域、40 dB 和 45 dB 样本，但对 50 dB 的大量弱正错误仍给出“可靠”判断。换言之，当前 Stage2 表征不能区分：

1. 真正缺失 SawFM 的负样本；
2. SawFM 存在但已经低于当前特征可见度的 50 dB 弱正样本。

这与前几轮的机制结论一致：问题不只是阈值或分类边界，而是弱正证据在共享特征中的可识别性不足。

## 5. 晋级门审计

R4 三个种子都满足：

- early AUROC 不低于 0.65；
- validation→early 落差不大于 0.15；
- validation 拒识阈值可行；
- early 普通覆盖率与平均功率覆盖率下限。

但三个种子都失败于：

- R4 相对 R2 AUROC 增量不低于 0.03；
- 最大选择性风险不高于 0.15。

所以总体晋级结果为 `0/3`，不得训练新可靠性头。

## 6. 对下一步的约束

### 立即停止

- 不再对 A0 做普通阈值、双阈值或功率 oracle 阈值搜索；
- 不训练基于同一 Stage2 特征的 tiny temporal reliability head；
- 不解冻 F2 stage3/stage4；
- 不把 C2/A0 提升为正式二元决策方法。

### 保留

- H2/Base-DB：六类阶段主要二元决策基线；
- C2：SawFM 排序、候选检索和机制分析分数；
- R2 visibility probe：仅作为错误分析工具，不作为部署决策头。

### 若继续 SawFM 路线，必须改变证据来源

后续只有一条仍符合证据链的探索方向：从更早且尚未被共享融合抹平的复数特征分叉，建立真正独立的 SawFM 输入/表征，并先做正样本可辨识性 oracle 筛选。新分支必须在独立 development 上同时证明：

1. 弱正/负样本 AUROC 明显超过 R2 visibility 基线；
2. 50 dB 正样本召回改善不是由 FPR 失控换取；
3. 跨种子、跨功率泛化落差可控；
4. 达标前不接回六类联合分类器。

## 7. 实验产物

- 预注册：`docs/plans/route_a_stage2_reliability_identifiability_plan_20260725.md`
- 程序：`experiments/run_route_a_stage2_reliability_identifiability.py`
- 测试：`tests/test_stage2_reliability_identifiability.py`
- 正式汇总：`outputs/tapc/route_a_stage2_reliability_identifiability/route_a_stage2_reliability_identifiability_summary.json`
- 每种子：`seed_2041`、`seed_2042`、`seed_2043` 下的模型选择、threshold search、early predictions、metrics 与 receipts
- locked test：未使用
