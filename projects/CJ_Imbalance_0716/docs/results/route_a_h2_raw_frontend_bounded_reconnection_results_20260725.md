# 路线 A：H2 与 Raw SawFM 前端有界接回结果（2026-07-25）

## 1. 结论

本阶段没有通过晋级门，正式结论为：

> Raw-only SawFM 前端具有很强的独立排序能力，但最大 0.20 的 H2 有界残差无法克服 H2 原有的错误排序；重新选择阈值后，只获得很小的弱正召回增益，且个别种子出现普通域或功率域 FPR 泛化违约。因此 Raw 前端暂时保留为独立机制结果，不接回 H2 六类决策。

正式决策：`keep_raw_frontend_as_independent_mechanism_do_not_reconnect_h2`。

## 2. 实验协议

- seeds：2041、2042、2043；
- F2、H2、Raw frontend 全部冻结；
- Raw frontend checkpoint：各 seed 的 `N0_RawOnly`；
- 其他五类 logits 和阈值完全保持 H2 原值；
- 接回只修改 SawFM 一列；
- threshold role：选择残差形式和 SawFM 阈值；
- ordinary early + power development：独立评价；
- 最大残差幅度：0.20 logit；
- locked test：未使用；
- 身份重叠：0。

候选包括：

- H2 重选阈值基线；
- `alpha=0.05/0.10/0.20` 的双向有界残差；
- 9 个 H2 低置信 + Raw 高证据的 hard-rescue gate。

## 3. 聚合结果

| 方法 | Mean-power recall | Min-power recall | 50 dB recall | Max power FPR | Ordinary SawFM FPR | Macro-AUPR | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 H2 | 0.0346 | 0.0068 | 0.0693 | 0.1685 | 0.1731 | 0.8670 | 0.7468 |
| Chosen bounded | 0.0550 | 0.0240 | 0.0999 | 0.1685 | 0.1666 | 0.8699 | 0.7516 |
| 增量 | +0.0205 | +0.0172 | +0.0306 | +0.0000 | -0.0065 | +0.0028 | +0.0048 |

新方法没有损害宏指标，甚至略微改善普通域结果；但主要目标——弱正召回——远低于预注册要求：

- mean-power recall 要求 `+0.10`，实际仅 `+0.0205`；
- 50 dB recall 要求 `+0.15`，实际仅 `+0.0306`。

晋级种子数为 `0/3`。

## 4. 逐种子结果

| Seed | 选择方法 | Threshold | Mean recall | 50 dB recall | Max power FPR | Ordinary SawFM FPR | 结果 |
|---:|---|---:|---:|---:|---:|---:|---|
| 2041 | rescue `g=-0.5,r=0.90` | 0.13 | 0.0748 | 0.1020 | 0.1633 | 0.1258 | 召回增量不足 |
| 2042 | rescue `g=0.0,r=0.70` | 0.11 | 0.0300 | 0.0700 | 0.1400 | 0.2324 | 普通域 FPR 违约 |
| 2043 | rescue `g=-0.5,r=0.70` | 0.14 | 0.0603 | 0.1277 | 0.2021 | 0.1417 | 功率域 FPR 违约 |

三个种子都优先选择 hard rescue，而不是连续 signed residual。这说明在当前分数尺度下，只有对少数 H2 低置信样本执行离散救援才有可见效果；连续 `alpha<=0.20` 的残差太弱。

## 5. 失败原因

### 5.1 残差尺度与 H2 错误排序不匹配

H2 为满足 FPR 约束，最终 SawFM probability threshold 约为 0.11–0.14，对应 logit 约 `-2.0`。最大 `+0.20` 的救援只能产生有限概率位移。大量弱正的 H2 logit 与 hard negative 已经发生深度交叠，0.20 不能改变其排序位置。

连续 signed residual 的 threshold-role 改善也很小。例如 seed 2041 从 alpha 0.05 增至 0.20，mean-power recall 只从约 0.0714 增至 0.0762。由此可见失败不是 alpha 网格过稀，而是受限残差的有效动态范围不足。

### 5.2 强 Raw 排序被 H2 主导项稀释

Raw frontend 单独的 power-development AUROC 接近 1.0，但融合形式始终以 H2 为主，Raw 只能提供很小修正。因此最终 50 dB AUROC 仅由 H2 的约 0.4102 提高到约 0.4877，仍接近随机，远低于 Raw frontend 独立输出。

这证明“独立专家有效”不等于“低幅度残差可接回”。当前 H2 与 Raw frontend 不是小误差互补关系，而是存在明显的排序冲突。

### 5.3 阈值泛化仍存在域差异

所有候选在 threshold role 满足 FPR 约束后才允许入选，但 early/development 中：

- seed 2042 ordinary SawFM FPR 上升到 0.2324；
- seed 2043 power max FPR 上升到 0.2021。

说明不同身份角色上的最优 rescue 区域并不稳定。继续扩大 gate 网格容易造成开发集过拟合，不能作为下一步解决方案。

### 5.4 其他五类保持成功，但不足以抵消主目标失败

其他五类 probability 最大差异为 0，预测差异数为 0；macro-AUPR 和 macro-F1 也保持非劣。这证明实现的隔离性正确，但接回有效性仍不成立。

## 6. 下一步约束

### 停止

- 不继续扫描 `alpha<=0.20`；
- 不继续扩大 hard-rescue gate 网格；
- 不把 Raw frontend 接入当前 H2 输出；
- 不解冻 F2/H2；
- 不使用 locked test。

### 保留

- H2/Base-DB：正式六类二元决策基线；
- Raw frontend：SawFM 独立识别与相位机制证据；
- Phase-only 与 time-shuffled 消融：论文机制证据。

### 若继续研究决策融合

必须改变融合问题定义，而不是增大残差：

1. 把 Raw frontend 视为独立 SawFM classifier，单独校准和决策；
2. 在普通全分布上构造足够的 SawFM absent negatives，而非只依赖组合匹配 hard negatives；
3. 比较 H2、Raw-only、二者的选择性 mixture-of-experts；
4. 先证明 Raw-only 在普通全分布上的 FPR、AUPR和校准可接受；
5. 只有独立 Raw 决策稳定后，才允许研究显式替换或 convex mixture，不能再假设 H2 必须是主项。

## 7. 产物

- 方案：`docs/plans/route_a_h2_raw_frontend_bounded_reconnection_plan_20260725.md`
- 程序：`experiments/run_route_a_h2_raw_frontend_bounded_reconnection.py`
- 测试：`tests/test_h2_raw_frontend_bounded_reconnection.py`
- 汇总：`outputs/tapc/route_a_h2_raw_frontend_bounded_reconnection/route_a_h2_raw_frontend_bounded_reconnection_summary.json`
- 汇总 SHA256：`8779039ED9CD73B1557DAFEB323EA0E52271451DA17D914DC929D559FF535244`
