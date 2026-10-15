# Route A 双轴长尾复合识别：当前证据差距、已验证内容与长线规划

日期：2026-08-08  
项目：`E:\LJQ_code\CJ_Imbalance_0716`  
依据 Prompt：`docs/prompts/route_a_dual_axis_evidence_gap_long_horizon_analysis_prompt_20260808.md`  
当前证据终态：`terminal_N_complete`  
性质：对冻结结果的证据审计与 `[T]` 长线规划，本报告未训练新模型、未生成新数据、未访问任何冻结测试集。

## 1. 执行摘要

当前项目已经完成了严格的 identity-paired F×V 2×2 development 协议、五 seed 正式矩阵、receiver-level bootstrap、成本与复现审计，工程和实验治理已经成熟；但统一六类双轴正向主张仍未成立。最强正结论是 RawSaw 在全新 development 上稳定解决了 **SawFM 单标签**的低可见度问题，weak AUROC `+0.3169`、weak recall `+0.4208`。最强负结论是当前 DB 频次轴只取得 `+0.000984` macro-AUPR，且 CI 上界仍低于 `+0.005` 门槛；MFV 的组合 weak recall 增量严格为 `0`，macro 增量也只有 `+0.002650`。因此主瓶颈已从“缺少正式协议”转移为“频次轴机制无足够效应、可见度证据只覆盖 1/6 标签、组合结构没有新增 SawFM 决策”。

若以统一六类双轴创新所需的 12 环证据链计，当前为 **4.0/12，约 33%**；若只看六个决定性正向门，当前为 **1/6 完整通过、1/6 部分通过、2/6 失败、2/6 缺失**。这不是模型准确率或录用概率，而是证据完整度。总体差距仍为“大”。

## 2. 当前方法与证据地图

### 2.1 频次轴

- 基线：`M0_B0_Focal`；
- 候选：同结构 `MF_B0_DB`；
- 网络参数完全相同，均为 257,511；
- 差异只来自 Focal 与 Distribution-Balanced Loss；
- 在正式新 identity LT 条件下，DB 没有达到最小科学效应。

### 2.2 可见度轴

- 候选：`MV_B0_RawSaw`；
- RawSaw 使用原始 I/Q 的幅度、幅度差分、相位增量和 7/31/63 多尺度卷积；
- 只替换 SawFM logit，其余五标签 bitwise 保持 B0 输出；
- 增加 5,521 参数和 7,442,848 MACs；
- 在新 development 上稳定改善 SawFM weak-vs-negative 排序和召回。

### 2.3 组合模型

- `MFV_B0_DB_RawSaw` 的非 SawFM 输出来自 MF；
- SawFM 输出与 MV 使用同一个 RawSaw checkpoint；
- 这是一种正交模块拼接，不存在新的 SawFM 交互通道；
- 因而组合 weak recall 相对 MV 必然为 0，组合 macro 增量只能来自其余五标签的 DB 净变化。

### 2.4 数据与统计

- 新 paired development：25,200 identities；
- optimization 15,120；early/calibration/threshold/evaluation 各 2,520；
- 五 seeds × FB/LT × VH/VL，共 20 个正式 cell；
- 5,000 次 receiver-identity paired bootstrap；
- locked test、test-v2、test-v3 使用状态均为 false。

## 3. 效应距离量化

| 证据 | 实际结果 | 门槛 | 还差多少 | 门槛完成比例 | CI 对门槛的判断 |
|---|---:|---:|---:|---:|---|
| Frequency LT macro-AUPR | +0.000984 | +0.005 | **+0.004016** | 19.7% | CI 上界 +0.001821，仅为门槛 36.4%，真实效应很可能过小 |
| Visibility weak AUROC | +0.316928 | +0.020 | 超出 +0.296928 | 1584.6% | CI 下界 +0.304463，稳定通过 |
| Visibility weak recall | +0.420806 | +0.050 | 超出 +0.370806 | 841.6% | CI 下界 +0.409558，稳定通过 |
| Visibility candidate FPR | 0.185887 | <=0.20 | 余量 0.014113 | 通过 | CI 上界 0.200078，接近并略越约束边界 |
| Combined macro-AUPR | +0.002650 | +0.005 | **+0.002350** | 53.0% | CI 上界 +0.003613，仅为门槛 72.3%，不是单纯统计功效不足 |
| Combined weak recall | 0 | +0.020 | **+0.020** | 0% | CI `[0,0]`，属于结构性零增量 |

关键判断：

1. 频次轴不是“差一点显著”，而是 receiver CI 整体远低于最小科学效应；
2. combined macro 虽稳定大于零，但没有达到事先规定的有意义增量；
3. combined weak recall 的缺口不是训练噪声，而是由共享 RawSaw 输出造成的结构性零；
4. 可见度轴本身并非短板，但只覆盖 SawFM，不能代表六类统一 visibility method。

## 4. 12 环正向证据链评分

评分规则：完整正向通过 1.0；范围受限但具有正式证据 0.5；失败或缺失 0。负结果提高科研认知质量，但不增加“统一正向主张”的完成度。

| 环节 | 状态 | 得分 | 当前证据 | 关键缺口 |
|---|---|---:|---|---|
| 1. 双轴定义与 estimand | pass | 1.0 | 已明确 F、V、ΔF、ΔV、combined | 无 |
| 2. Identity-safe 2×2 development | pass | 1.0 | 新 25,200 paired identities、角色隔离 | 仍缺 confirmation source |
| 3. 公平 B0、统计和复现 | pass | 1.0 | 同初始化/预算、20 receipts、bootstrap、成本 | 无主要缺口 |
| 4. Frequency-only 主效应 | fail | 0 | +0.000984，4/5 正向 | 未达到 +0.005，标签间抵消 |
| 5. Visibility-only 主效应 | partial | 0.5 | SawFM 大幅通过 | 只覆盖 1/6 标签，FPR CI 临界 |
| 6. Combined 独立增量 | fail | 0 | macro +0.002650；weak recall 0 | 未达到最小效应，无新交互通道 |
| 7. Equal-capacity/破坏 controls | missing by stop | 0 | 因频次门失败而预注册停止 | 正向统一主张没有 controls 胜出证据 |
| 8. 六标签多弱目标覆盖 | missing | 0 | 非 SawFM 只做安全旁路 | MFT/VFT/ISRJ 等弱证据未解决 |
| 9. 校准、FPR、安全与成本 | partial | 0.5 | 有独立角色、成本、point safety | FPR CI 临界；MFT safety CI 下界略低于 -0.01 |
| 10. Fresh confirmation | missing | 0 | 门失败，未创建 | 完全缺失 |
| 11. Cross-receiver/generator/OOD | missing | 0 | 当前为单生成 lineage | 完全缺失 |
| 12. Real/semi-physical/HIL | missing | 0 | 无 | 完全缺失 |
|  | **总计** | **4.0/12** | **约 33%** | **统一正向证据仍缺约 67%** |

六个决定性门可简化为：

| 决定性门 | 状态 |
|---|---|
| 正式 F×V 2×2 协议 | pass |
| Frequency-only | fail |
| 跨标签 Visibility-only | partial：仅 SawFM |
| Combined independent gain | fail |
| Fresh confirmation | missing |
| External/real/HIL | missing |

即 **1/6 完整通过**。

## 5. 已验证、已否定和未验证事项

### 5.1 已验证

1. `[F/A]` 可以构造 receiver identity 对齐、角色隔离且 SHA 固定的 F×V 2×2 development 协议。
2. `[A]` RawSaw 使用小幅参数和计算增量即可保留 SawFM 弱相位—时间证据，说明 SawFM 的主要问题是观测归纳偏置，而不是单纯 backbone 容量不足。
3. `[A]` SawFM visibility-only 在五 seed、新 identity、receiver bootstrap 下稳定通过。
4. `[A]` 当前模块路由能保证非 SawFM logits 不被 RawSaw 改变。
5. `[F]` 训练、校准、阈值、evaluation、receipt、成本和代码哈希链可复现。

### 5.2 已否定或严格停止

1. `[N]` 当前 DB 不能作为正式频次轴创新；其收益集中于 VFT/部分标签，并被 SawFM、MFT、SinFM 损失抵消。
2. `[N]` 当前 MFV 没有证明频次—可见度协同；weak recall 增量结构性为零。
3. `[N]` 在同一 R2 evaluation 上继续调 DB 权重、epoch、阈值或 seed 不被允许。
4. `[N]` Generic backbone 继续扩宽没有新增准入证据。
5. `[N stopped]` MEC、MTS 和信息破坏 controls 不是“失败结果”，而是因为 frequency gate 失败而未获运行授权。

### 5.3 尚未验证

1. 能否通过分类器解耦而不是 loss 重权获得稳定 frequency main effect；
2. RawSaw 是否能跨 generator、receiver 或真实采集复现；
3. 是否存在适用于 MFT/VFT/ISRJ 的非 oracle 可见度观测；
4. combined 是否能在全新数据上稳定优于两个单轴；
5. 是否存在正的统计 interaction，而非简单模块相加；
6. fresh confirmation、OOD、real/HIL 和部署校准运输。

## 6. 根因与当前主瓶颈

### 6.1 主瓶颈一：频次轴机制错位

DB 在 loss 层按标签计数重权，但当前失败来自多标签共现下的排序边界和标签间负迁移。它没有将“稀有标签正样本”与“组合相似但该标签缺失的 hard negatives”直接分开，因此 VFT 增益可以伴随 MFT/SawFM/SinFM 损失。继续搜索 DB 权重只会在同一权衡面移动，不会形成新的可证伪机制。

### 6.2 主瓶颈二：可见度证据覆盖不足

SawFM 的 raw phase/time-order 先验有效，但其物理结构不能直接推广到所有干扰类型。当前 V 轴是 1/6 标签特定专家，不是统一可见度表示。

### 6.3 主瓶颈三：组合只有路由，没有联合决策

MFV 对 SawFM 完全复用 MV，对其他标签完全复用 MF。它可以产生模块互补，但不能产生 SawFM 的新增交互收益。若未来仅证明 combined 优于 best single-axis，只能称“正交模块互补”；只有 factorial interaction CI 下界大于零时才允许使用“协同”。

### 6.4 非主瓶颈：训练时长和全局 backbone 宽度

频次效应的 receiver CI 上界只有 +0.001821；combined CI 上界只有 +0.003613。现有证据更支持机制效应不足，而不是简单增加 epoch 或宽度即可解决。

## 7. 三类论文目标的距离

| 论文目标 | 当前成熟度 | 距离 | 最低补实验集合 |
|---|---:|---|---|
| 双轴问题定义与评估框架 | 约 7/10 | 小到中等 | 文献对齐、公开可复建 manifest、再现一个不同方法、强调负结果和 estimand |
| SawFM 标签特定弱可见度方法 | 约 6.5/10 | 中等 | fresh cross-generator/receiver confirmation、equal-capacity、phase/envelope/time-shuffle、主流 raw-IQ baseline、FPR 运输 |
| 统一六类双轴复合识别 | 约 4/10 | 大 | 新 frequency mechanism、跨标签 V 轴、combined gain、controls、fresh confirmation、OOD/HIL |

最现实的短期 SCI 收口是 SawFM 标签特定方法或双轴评估框架；统一六类双轴方法仍属于长线目标。

## 8. 0–18 个月长线试验路线图

| 时间 | 唯一科学问题 | 数据与模型 | 通过门 | 失败动作 |
|---|---|---|---|---|
| 0–3 月 | 冻结 backbone 后，分类器解耦能否建立真正 frequency main effect？ | 全新 `dual-axis-frequency-development-v2`；IF10/20/50 label-LT + combination-LT；RB-cRT | IF20 上相对 best(B0,DB,普通cRT) macro-AUPR >=+0.005、CI>0、4/5 正向、FB 安全 | 关闭当前频次路线，不接回 RawSaw |
| 3–6 月 | 新频次头能否与 RawSaw 正交组合？ | 同一新 identity 的 VH/VL；重新训练固定结构 RawSaw；2×2 | ΔF、ΔV 分别通过；combined vs best single >=+0.005 | 只保留两个单轴或 SawFM 论文，不称联合模型 |
| 6–12 月 | 正向结果能否战胜容量、阈值和信息破坏解释？ | equal-capacity、ordinary-cRT、no-rank、no-bound、envelope、phase、time-shuffle、threshold-only | 主要候选胜全部关键 controls，receiver CI>0，非目标安全 | 机制收缩或关闭；禁止选择性报告 |
| 12–18 月 | 能否跨 source 确认并外部运输？ | 全新 confirmation；跨 generator/receiver；semi-physical/HIL | 完全冻结一次性通过；性能与 FPR 运输落差在预注册范围 | 降级为 development 方法或边界论文 |

## 9. 下一阶段改进方案

下一阶段只做 **频次轴重建**，暂不扩展多标签 visibility，也不继续扩大 backbone。

### 9.1 新数据

- 创建全新 identity 的 `dual-axis-frequency-development-v2`；
- receiver/component seeds 与 R2 完全不重叠；
- 同时生成 VH 30 dB 和 VL 20 dB，为未来组合保留 paired views；
- primary：IF20 label-LT 与 combination-LT；secondary：IF10/IF50；
- 保留 FB safety；
- optimization、early、calibration、threshold、evaluation identity 隔离；
- R2 evaluation、locked test、test-v2、test-v3 均不得参与选择。

### 9.2 Risk-Bounded Co-occurrence cRT（RB-C cRT）

保持 B0 的 128 维特征和主干结构不变，先训练 B0-Focal representation，再冻结 backbone，重建六个标签的线性分类器：

`z = W h + b, h in R^128`

建议损失：

`L = L_label_balanced + 0.25 L_cooccurrence_rank + 1e-3 L_anchor`

- `L_label_balanced`：每个标签的正、负项在 epoch 内各贡献 0.5，避免标签频次直接主导梯度；
- `L_cooccurrence_rank`：对稀有标签正样本与“其余标签组合相似但目标标签缺失”的 hard negative 做 margin=0.2 的 pairwise ranking；
- `L_anchor`：限制新 head 相对 B0 head 的归一化漂移，保护 FB 和非尾标签；
- 不把 weak/strong mask、功率或 pair id输入模型；
- 先 head-only；仅允许一个预注册的 stage4 rank-4 low-rank adapter 作为 secondary 候选，不扩宽主干。

### 9.3 对照

- B0-Focal；
- B0-DB；
- ordinary BCE cRT，同训练预算；
- RB-C cRT；
- no-rank；
- no-anchor；
- equal-parameter random-pair rank control。

只有 frequency gate 通过后，才允许重新训练固定 RawSaw 并进入新 2×2。

### 9.4 为什么该方案风险更低

1. 直接针对已观测到的“标签间抵消”和“共现 hard negative”问题；
2. 冻结 backbone，可区分分类器边界问题与特征不足；
3. 参数量几乎不增加，公平对照清楚；
4. 若失败，可以明确关闭 classifier-decoupling 频次路线；
5. 不需要复用已消费的 evaluation，也不依赖部署不可获得的 oracle 信息。

## 10. 当前论文表述边界

允许：

- 已建立可复现的 identity-safe F×V 评估协议；
- RawSaw 在新 development 上对 SawFM 弱可见度具有稳定增益；
- 当前 DB 和模块化 MFV 未达到预注册频次与组合门槛；
- 当前结果支持频次偏置和标签内可见度退化是不同机制。

禁止：

- 已验证统一六类双轴长尾模型；
- 频次轴与可见度轴存在协同；
- combined 在 confirmation/test 上有效；
- 用 bootstrap CI 大于零替代最小科学效应门槛；
- 通过同一 R2 evaluation 继续调参挽救结果。

未来只有在新 development-v2、controls、fresh confirmation 和外部运输全部通过后，才允许声称统一双轴方法。

## 11. 下一唯一动作

冻结当前 terminal N bundle，启动一个全新的 **frequency-axis reconstruction Goal**。先审计生成器和 seed lineage，写预注册并生成新 identity development-v2；首个正式问题只回答 RB-C cRT 是否能在 IF20 label/combination-LT 上稳定超过 B0、DB 和 ordinary cRT。该门失败即停止，不重新接回 RawSaw。

