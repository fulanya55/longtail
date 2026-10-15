# 路线 A：Selective Unknown 决策试验结果（2026-07-25）

## 1. 结论

本轮完成冻结 A0/C2 上的 `positive / negative / unknown` 双阈值验证。U2统一选择性决策在3/3种子上均没有满足预注册约束的阈值对；U3分功率oracle在threshold role可行，但在独立early-development上也未通过FPR、正样本误判负和selective risk门槛。

因此本轮结论为：**停止纯校准/双阈值拒判路线，不解冻主干，也不自动进入stage2可靠性双头。保留C2作为SawFM跨功率排序结构，H2/Base-DB作为当前二元决策基线。**

## 2. 协议

- F2、H2、稳定S2、C2 `alpha=0.80`、A0/V2全部冻结；
- 重放上一轮相同expert-scale、visibility-fit和power-threshold身份；
- A0 checkpoint、feature mean/std和source hash直接加载，不重新训练；
- threshold role选择双阈值，ordinary early和40/45/50 dB early-development只评价；
- 网格步长0.01；FPR≤0.20，positive false-negative rate≤0.10；
- ordinary decision coverage≥0.70，三功率平均coverage≥0.40；
- seeds 2041/2042/2043；locked test未使用；identity overlap为0。

重放审计：正式A0旧预测与本轮source replay逐数组最大绝对误差为0，排除了checkpoint或协议漂移。

## 3. 变体

- U0：A0单阈值，不设unknown；
- U1：统一Platt后的C2双阈值；
- U2：A0/V2双阈值，唯一晋级候选；
- U3：ordinary、40、45、50分别选择A0双阈值的oracle上限。

## 4. 主要结果

| 方法 | 可行种子 | 普通recall | 普通FPR | 普通coverage | 三功率平均coverage | 三功率平均确认召回 | 50 dB确认召回 | 50 dB误判负 | 50 dB coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U0 A0单阈值 | 3/3 | 0.7230 | 0.2048 | 1.0000 | 1.0000 | 0.4724 | 0.1668 | 0.8332 | 1.0000 |
| U1 C2统一双阈值 | 0/3 | — | — | — | — | — | — | — | — |
| U2 A0统一双阈值 | 0/3 | — | — | — | — | — | — | — | — |
| U3 分功率oracle | 3/3 | 0.6193 | 0.1181 | 0.9856 | 0.8847 | 0.7879 | 0.6987 | 0.1125 | 0.8029 |

U3展示了较高的潜在召回和coverage，但它未通过独立early风险门，不能视为oracle selective成功：

- 45 dB confirmed-positive FPR均值0.2400，高于0.20；
- 50 dB FPR均值0.1925，但种子2043为0.245；
- 45/50 dB正样本误判负均值为0.1101/0.1125，略高于0.10；
- 最大selective risk均值0.2549，高于0.15；
- U3单种子完整通过数为0/3。

## 5. U2为什么没有可行统一阈值

U2并非完全没有满足FPR和正样本误判负约束的候选。冲突来自普通域coverage门：

| seed | 同时满足FPR/FN时最佳`t_neg/t_pos` | 普通coverage | 三功率平均coverage | 最差功率coverage |
|---:|---|---:|---:|---:|
| 2041 | 0.22 / 0.63 | 0.613 | 0.526 | 0.466 |
| 2042 | 0.21 / 0.73 | 0.570 | 0.457 | 0.393 |
| 2043 | 0.21 / 0.63 | 0.630 | 0.570 | 0.463 |

为了保证所有功率下真实正样本被确定判负的比例不超过0.10，必须把大量普通域样本送入unknown，普通coverage只能达到0.57～0.63，低于预注册0.70。这不是网格过粗或简单阈值数值错误，而是统一分数在普通域和弱功率域中的风险排序不一致。

U1冲突更严重：满足FPR/FN时普通coverage约0.44～0.52，平均功率coverage约0.20～0.26，说明A0的10维条件校准确实比标量C2更接近可用，但仍不充分。

## 6. U3为什么也不能触发下一结构

U3在threshold role中为40/45/50选择的阈值随功率明显下移，例如：

- seed2041：`(0.49,0.55)`、`(0.40,0.45)`、`(0.22,0.32)`；
- seed2042：`(0.47,0.64)`、`(0.44,0.50)`、`(0.21,0.32)`；
- seed2043：`(0.51,0.57)`、`(0.45,0.47)`、`(0.21,0.37)`。

这确认功率分布平移存在。但阈值迁移到独立early后，40/45 dB FPR在多个种子超过0.20，ordinary selective risk也达到0.238～0.276。分域阈值能提高召回，却没有形成稳定的风险控制。

因此，不能依据“U3平均coverage>0.60”直接训练新网络；预注册要求独立early完整通过，实际为0/3。

## 7. 根因更新

1. **不是简单二元阈值问题。** 单阈值和双阈值都受同一分数风险排序限制。
2. **unknown思想在描述层面合理，但现有分数不能稳定决定谁应unknown。** 要保护弱正，普通域拒判率过高；要保持普通coverage，则弱正仍被确定判负。
3. **功率oracle仍发生threshold→early漂移。** 说明即使知道功率，有限threshold身份上的经验分位阈值也无法稳定控制未见身份风险。
4. **现有A0统计更适合排序/局部校准，而不是可靠性估计。** 若未来研究exist/reliability双头，必须先证明其oracle可靠性标签在独立身份上可辨识，不能直接进入大模型训练。
5. **当前不应解冻主干。** 表征排序已通过，而新增失败发生在决策可靠性层；解冻会混淆归因。

## 8. 决策

本轮总门失败，下一决策为：

`stop_selective_calibration_keep_c2_ranking_h2_decision`

具体含义：

- C2继续作为SawFM跨功率排序候选和SCI机制结果；
- H2/Base-DB继续作为当前二元识别决策基线；
- A0保留为40/45 dB可见度校准负/部分正结果；
- 停止继续调单阈值、双阈值、锚点和小型仿射校准器；
- 不自动实施stage2 exist/reliability双头；下一轮若继续，先做只读oracle可辨识性验证，证明“错误风险/可靠性”在stage2时序图上可预测后才允许训练。

## 9. 产物

- 预注册方案：`docs/plans/route_a_selective_unknown_decision_plan_20260725.md`
- 实验程序：`experiments/run_route_a_selective_unknown_decision.py`
- 正式汇总：`outputs/tapc/route_a_selective_unknown_decision/route_a_selective_unknown_decision_summary.json`
- 汇总SHA256：`14BBFBEC477B4DB6F68802D320DE0E218839724929A68F2819C7BBDC474C7E6A`
- 每种子保存完整threshold search、source/identity audit、预测、metrics和receipt；
- PY10全量测试：176 passed；仅有既有AMP接口弃用警告。
