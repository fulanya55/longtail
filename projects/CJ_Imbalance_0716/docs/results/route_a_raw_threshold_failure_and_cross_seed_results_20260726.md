# Route A Raw SawFM 阈值失败分析与跨种子改进结果（2026-07-26）

## 1. 本轮结论

当前失败不是 Raw SawFM 特征排序能力不足，而是逐 seed 阈值在 selection role 上紧贴 FPR 上限，迁移到 `ordinary_early` 后缺少统计安全裕量。简单把逐 seed 搜索预算从 0.20 收紧到 0.15 仍不能解决 seed 2039；改成“由其他四个 seed 共同决定目标 seed 阈值”的 LOSO 共享阈值后，开发集跨种子稳健性门槛全部通过。

新的冻结候选为：

`RawSawCrossSeedFPR15 + H2Others`

locked test 未访问，仍需单独明确授权。

## 2. 失败原因证据链

### 2.1 不是特征排序失败

当前 `C0` 五种子 Raw SawFM ordinary AUPR 均值约为 0.9547；seed 2039 ordinary AUPR 为 0.9558、召回为 0.9130，功率开发集 50 dB 召回为 1.0000。说明模型已经把正负样本大体排好序，主要问题发生在把连续分数转换为二值结果的阈值环节。

### 2.2 当前阈值缺少外推裕量

原阈值在 threshold role 上的 ordinary FPR 为 0.1818–0.1986，几乎都贴着 0.20 上限。seed 2039 的 threshold-role FPR 为 0.1944，但 ordinary-early FPR 上升到 0.2806。这是有限样本选择误差和 role 分布差异的典型表现。

### 2.3 逐 seed 收紧仍不稳健

预注册主方案 `T1_FPR15_PerSeed`：

- 五种子平均 ordinary FPR：0.1588；
- seed 2039 ordinary FPR：0.2374，仍超过 0.20；
- 五种子平均 ordinary recall：0.8587，相对 C0 的 0.8813 下降 0.0226，超过允许下降 0.02；
- 50 dB 平均召回：0.9679；
- 路由 macro-F1：0.8030，路由 5/5 seed 通过。

因此 T1 正式失败。它证明只在同一 seed 内收紧经验 FPR，不足以保证新 role 上的 FPR。

### 2.4 次要风险

- Raw frontend 协议不完全统一：2039/2040 为 `new_split_R0`，2041–2043 为 `original_N0`；
- 每个 seed 单独拟合 Platt，系数和分数尺度存在差异；
- ordinary early 已参与本轮开发分析，不能被当成最终独立测试。

## 3. 改进设计

主要方案 `C1_LOSO_FPR15_Shared` 对每个目标 seed：

1. 排除目标 seed 的全部 threshold-search 信息；
2. 在其余四个 seed 的公共阈值网格上搜索；
3. 要求四个来源 seed 的 ordinary 和 power FPR 均不超过 0.15；
4. 按平均功率召回、平均 ordinary 召回、50 dB 召回和阈值依次择优；
5. 将选出的阈值用于 held-out seed 的 ordinary early、power development 和六类标签路由。

网络权重、Platt 校准器、H2 分类头均未改变。

## 4. 正式结果

| 指标 | C0 当前阈值 | C1 LOSO-FPR15 | 变化 |
|---|---:|---:|---:|
| ordinary FPR | 0.1960 | 0.1426 | -0.0534 |
| ordinary recall | 0.8813 | 0.8448 | -0.0365 |
| 50 dB recall | 0.9758 | 0.9657 | -0.0101 |
| 六类 macro-F1 | 0.8027 | 0.8026 | -0.0001 |
| exact match | 0.2747 | 0.2807 | +0.0060 |
| 路由通过数 | 5/5 | 5/5 | 不变 |

逐 seed 主要结果：

| seed | LOSO 阈值 | ordinary FPR | ordinary recall | 50 dB recall | 路由 |
|---:|---:|---:|---:|---:|---:|
| 2039 | 0.48 | 0.1871 | 0.8634 | 0.9897 | 通过 |
| 2040 | 0.45 | 0.1519 | 0.8873 | 0.9907 | 通过 |
| 2041 | 0.48 | 0.0596 | 0.7919 | 0.9388 | 通过 |
| 2042 | 0.48 | 0.1479 | 0.8038 | 0.9200 | 通过 |
| 2043 | 0.48 | 0.1667 | 0.8778 | 0.9894 | 通过 |

10 项预注册门槛全部通过。阈值范围为 0.45–0.48，其中 4/5 seed 为 0.48。

## 5. 解释与边界

改进的核心不是提升网络表征，而是用跨 seed 一致性约束代替单 seed 经验最优阈值。代价是 ordinary recall 平均下降 3.65 个百分点；收益是 FPR 平均下降 5.34 个百分点，且六类整体 macro-F1 几乎不变。这说明 Raw SawFM 专家原有排序能力可以保留，部署决策边界需要更保守、更跨 seed 稳健。

本轮结果只能支持：

> LOSO 阈值规则在现有五个开发 seed 上具有跨种子稳健性。

不能支持：

> 已在独立 locked test 上证明泛化有效。

## 6. 冻结产物

- 阈值安全裕量总结：`outputs/tapc/route_a_raw_threshold_safety_margin/route_a_raw_threshold_safety_margin_summary.json`
- LOSO 正式总结：`outputs/tapc/route_a_raw_cross_seed_threshold/route_a_raw_cross_seed_threshold_summary.json`
- 新冻结推理包：`outputs/tapc/route_a_raw_saw_cross_seed_locked_test_preflight/frozen_inference_bundle.json`
- preflight 审计：`outputs/tapc/route_a_raw_saw_cross_seed_locked_test_preflight/preflight_audit.json`
- 执行收据：`outputs/tapc/route_a_raw_saw_cross_seed_locked_test_preflight/run_receipt.json`

关键哈希：

- 基础五种子总结：`1c456ff579f0fa2efccba202152f5f2ca117ca215a7d84131447d4086d491efc`
- LOSO 总结：`5ff8eb70ba1f76a006a2f25aa244e5c33823fea714a5360843b258d150265d61`
- 新冻结推理包：`14d3667dd545f5eaa8d1d69881050fa05da529ddec7b2a48d7af724c65fb3cea`
- preflight 审计：`3008ffb12630901313c099c882e9a47ed5bfa09eca4f9a195b5fd2c50e25a7a4`

## 7. 下一步工作计划

1. 保持网络、Platt 和 LOSO 阈值全部冻结，不再根据开发集修改候选；
2. 获得明确授权后，只执行一次 locked-test 评估，并同时报告 C0 与新 C1，但以新 C1 为预先指定主候选；
3. locked-test 主门槛重点检查 SawFM FPR、弱分量召回、六类 macro-F1 和 exact match；
4. 若 locked test 失败，不再继续事后调阈值，而是回到开发阶段统一五个 seed 的 Raw frontend 训练协议，并增加全新 seed/split；
5. 若 locked test 通过，补充阈值敏感性曲线、置信区间以及 Base-DB/H2/RawSaw 路由消融，形成论文证据链。

