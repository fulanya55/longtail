# Route A ISRJ Copy-Consistency 专家结果

## 判定

五种子组件门禁通过 5/9，整体失败，不得接回 H1 融合头。

增强后的 A1 相对原 A0 310D aligned HGB，early matched AUROC 平均下降 0.003668，95% bootstrap CI 为 `[-0.009033,0.002044]`；AUPR 平均下降 0.002596。只有 seed 2042 的 AUROC/AUPR 同时改善，未达到四种子一致性要求。

A1 相对同结构 block-shuffled S1 的 AUROC 仍高 0.079352，95% CI 为 `[0.063384,0.100623]`，五种子全部为正。这再次确认长周期对齐信息有效，但本轮 20 维 copy-consistency 统计没有提供稳定的额外判别信息。

## 逐种子结果

| seed | A0 AUROC | A1 AUROC | A1-A0 | A0 AUPR | A1 AUPR | A1-A0 | S1 AUROC | A1-S1 | val-early gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2039 | 0.6150 | 0.6145 | -0.0005 | 0.2443 | 0.2440 | -0.0003 | 0.5355 | +0.0790 | 0.0917 |
| 2040 | 0.6011 | 0.5936 | -0.0075 | 0.2247 | 0.2218 | -0.0029 | 0.5320 | +0.0616 | 0.2562 |
| 2041 | 0.6285 | 0.6156 | -0.0128 | 0.2692 | 0.2599 | -0.0093 | 0.5403 | +0.0754 | 0.1928 |
| 2042 | 0.6624 | 0.6683 | +0.0059 | 0.2716 | 0.2749 | +0.0033 | 0.5467 | +0.1217 | 0.0081 |
| 2043 | 0.6065 | 0.6031 | -0.0034 | 0.2509 | 0.2471 | -0.0038 | 0.5439 | +0.0592 | 0.1442 |

## 失败解释与停止规则

单一候选 ACF 峰并不一定对应真实 ISRJ 重复时延；在复合干扰中它可能被 FMB/VFT 或包络叠加主导。基于该候选时延计算的分窗复相关、相位一致性和谐波相关因此会把伪周期也编码进去。seed 2040/2041 较大的 validation-early gap进一步表明这些新增统计存在身份敏感性。

更重要的是，连续四类整改已经给出一致停止证据：全局安全阈值、单 FMB 上下文、多标签风险阈值、冻结分数难负加权、专家 OOF 难负重训和 copy-consistency 扩展均未通过预注册门禁。继续在 development-v3 搜索特征或超参数会显著增加后验选择偏差。

因此最终候选暂时保持原 `H1 + aligned periodicity B2`。它已经在 development-v3 和身份隔离的 ISRJ 参数 OOD-v1 上获得一致的 AUPR、弱召回和弱 AUROC 增益；子群 FPR 风险作为明确局限保留，不再用本开发集修补。下一阶段转入工程开销、统计效应量、复现包和最终冻结审计。

## 证据边界与产物

- seed 2039 smoke 只验证代码路径，未用于方法筛选；五种子结果独立报告。
- test-v3 未访问、未分析、未使用。
- 预注册：`docs/plans/route_a_isrj_copy_consistency_expert_plan_20260801.md`。
- 实现：`experiments/run_route_a_isrj_copy_consistency_expert.py`。
- 测试：`tests/test_route_a_isrj_copy_consistency_expert.py`，1 passed。
- 正式输出：`outputs/tapc/route_a_isrj_copy_consistency_expert`。
- smoke 输出：`outputs/tapc/route_a_isrj_copy_consistency_expert_smoke`。
- 正式汇总 SHA256：`1BA1A8CBD15632469C95EBCC8A52547864F79FEAFB785513838DBBFA00620147`。

