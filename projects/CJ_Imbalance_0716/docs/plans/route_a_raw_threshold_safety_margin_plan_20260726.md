# 路线 A：Raw SawFM 阈值安全裕量试验计划（2026-07-26）

## 1. 失败现象

当前Raw+Platt五种子中，seed 2039 ordinary threshold角色FPR为0.1944，满足设计约束0.20，但ordinary early FPR升至0.2806。该seed的ordinary AUPR为0.9558、recall为0.9130、power mean recall和50 dB recall均为1.0，因此不是特征排序失败，而是阈值跨角色泛化失败。

五个seed的当前阈值角色ordinary FPR分别为0.1944、0.1974、0.1986、0.1931、0.1818，几乎都贴近0.20边界。原搜索优先最大化召回，没有为有限样本误差和角色分布漂移保留安全裕量，这是当前最直接的结构性原因。

## 2. 改进原则

- 不修改F2、H2或Raw网络；
- 不重新训练或重新校准；
- 只使用已经冻结的ordinary/power threshold搜索结果选择新阈值；
- ordinary early和power development只用于最终评价，不参与阈值选择；
- locked test保持冻结。

## 3. 对照

| 变体 | 阈值规则 | 定位 |
|---|---|---|
| T0_FPR20_Current | 当前每seed经验FPR≤0.20 | 冻结基线 |
| T1_FPR15_PerSeed | 每seed ordinary及各power threshold经验FPR≤0.15 | 主要候选 |
| T2_FPR10_PerSeed | 每seed经验FPR≤0.10 | 保守上限 |
| T3_FPR15_Shared | 五seed共享一个阈值，且每seedthreshold角色FPR≤0.15 | 阈值方差诊断 |
| T4_FPR10_Shared | 五seed共享一个阈值，且每seedthreshold角色FPR≤0.10 | 极保守诊断 |

所有可行阈值继续按mean power recall、ordinary recall、50 dB recall和较高阈值排序。

## 4. 主要晋级门

T1至少满足：

1. seed 2039 ordinary early FPR≤0.20；
2. 5/5 seeds ordinary early FPR≤0.20；
3. mean ordinary recall相对T0下降不超过0.02；
4. mean 50 dB recall≥0.95；
5. 每seed、每power development FPR≤0.20；
6. 标签路由macro-F1相对T0下降不超过0.005；
7. exact match相对T0下降不超过0.01；
8. 标签路由仍至少4/5 seeds通过“macro-AUPR非劣、macro-F1改善、SawFM recall增益≥0.20、其他五类FPR不变、exact match非劣”门；
9. identity与locked-test审计通过。

## 5. 停止规则

- T1未解决seed 2039 FPR或平均recall下降超过0.02：不晋级安全裕量；
- T2/T3/T4只作机制诊断，不能因为early结果较好而直接取代预注册T1；
- 若T1失败，保持当前冻结候选，下一阶段转向Raw训练协议统一复现，而不是继续搜索更多FPR常数；
- 若T1通过，生成新候选冻结包，但仍不得自动执行locked test。

## 6. 产物

- 程序：`experiments/run_route_a_raw_threshold_safety_margin.py`
- 测试：`tests/test_raw_threshold_safety_margin.py`
- 输出：`outputs/tapc/route_a_raw_threshold_safety_margin/`
- 结果：`docs/results/route_a_raw_threshold_safety_margin_results_20260726.md`

