# P3o-I：耐久有限掩码组合 CVaR 的三种子开发止损记录

## 结论

P3o-I 完成了可执行、可复核的三种子开发验证，但未通过全部预注册门，结论为 `stop_no_confirmation_data`。因此不能主张“compound-CVaR 本身比等容量均值残差更能降低参数最坏组合风险”，不进入 1350M 确认，也不生成或访问测试集。

## 已验证的执行与安全契约

- 新身份池为 train seed 1330M、development seed 1340M；身份审计为零交集，且与既有 P3o-A--H 池无重叠。
- 2081、2082、2083 分别在独立 Python 进程和独立输出目录中完成，三者均有成功回执、无异常回执。
- 有限掩码参数损失只在 `label=1 AND parameter_mask=1 AND finite(target)` 的布尔索引位置计算；inactive NaN 不进入 Smooth-L1 算子。
- I0/I1 保持 F-base state 和 logits 逐位一致；分类阈值仅由 F-base calibration 拟合并原样复用。
- 所有三个 F-base 均通过有限样本安全校准与 ISRJ recall 门。
- `test_generated=false`、`test_opened=false`、`locked_test_used=false`。

## 冻结的单变量比较

- **F-base**：选定的局部时频 ParameterMatchedMultiView，BCE + `0.1 × masked Smooth-L1`。
- **I0**：有限掩码、状态保持、幅值受限的 delay residual，均值参数损失。
- **I1**：与 I0 完全相同的容量、初始化、数据顺序、预算和 checkpoint 规则，仅把侧支参数目标加入预注册的 compound-label-group 上尾 CVaR。

三臂均固定：20 epoch、2 epoch warmup、batch 63、AdamW `1e-4`/`1e-4`、残差界 0.25、CVaR fraction 0.25；I1 CVaR 权重为 1，I0 为 0。

## 三种子结果（I1 相对基线）

| 指标 | 2081 | 2082 | 2083 | 均值 | 预注册要求 | 判定 |
|---|---:|---:|---:|---:|---|---|
| oracle NMAE / F-base | 0.98970 | 0.97718 | 0.99095 | 0.98594 | 每种子 <= 0.98 | 失败 |
| predicted-gate NMAE / F-base | 0.98943 | 0.97584 | 0.99053 | 0.98527 | 每种子 <= 0.98 | 失败 |
| miss penalty 差值 | -0.00446 | -0.01038 | -0.00410 | -0.00632 | 每种子 <= 0 | 通过 |
| 最坏组合 predicted-gate NMAE / F-base | 0.96688 | 0.96773 | 0.95905 | 0.96455 | 每种子 <= 0.98 | 通过 |
| 最坏组合 predicted-gate NMAE / I0 | 1.01496 | 1.01607 | 1.01462 | 1.01522 | 每种子 <= 0.99 | 失败 |

## 失败归因

I1 相对 F-base 的最坏组合误差稳定下降约 3.3--4.1%，并且 miss penalty 未恶化。这说明“带有限掩码的有界侧支”本身可以改善部分风险指标。

但 I1 只有 seed 2082 达到总体 oracle 与 predicted-gate 至少 2% 的阈值；2081 与 2083 仅约 1% 改善。更关键的是，I1 在三个种子中相对等容量的 I0 都使最坏组合误差上升约 1.46--1.61%。所以可归因的结论是 **bounded residual 对 F-base 有效的迹象**，而不是 **CVaR 项优于均值残差**。后者没有通过直接、容量匹配的因果对照。

## 停止规则

- 不调整 CVaR 权重、fraction、残差界、checkpoint、阈值或身份以追逐门槛；
- 不扩展到五种子、不生成 1350M/1360M 确认、不访问任何测试或 locked test；
- 若未来要继续研究最坏组合参数风险，必须采用不同的、预注册的机制和新的身份池，并仍保留 F-base 与等容量均值残差对照。

## 可复核产物

- 预注册：`configs/p3oi_durable_finite_combo_cvar_preregister_20260802.json`
- 方法冻结：`configs/p3oi_durable_finite_combo_cvar_method_freeze_20260802.json`
- 合并三种子回执：`outputs/p3oi_durable_finite_combo_cvar_development_20260802/summary.json`
- 每种子成功回执：`outputs/p3oi_durable_finite_combo_cvar_development_20260802/seed_2081/summary.json`、`seed_2082/summary.json`、`seed_2083/summary.json`
