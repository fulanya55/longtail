# 路线 A：SawFM 尾类选模与 A6-D2 结果（2026-07-23）

## 1. 最终结论

本轮按预注册顺序完成两个阶段：

1. A6-PU 保持训练不变，改用逐 epoch SawFM AUPR 选择检查点，3 seeds；
2. 尾类选模仍失败后，执行 A6-D2，解冻最后两个时序块并使用分层学习率，3 seeds。

两个阶段均未通过门槛。SawFM 的失败不是单纯选错检查点，也不是只少解冻一个时序块，而是 **从 ASL 检查点切换为 DB 微调时，优化轨迹系统性损害 SawFM 排序能力**。

locked test 始终未使用。

## 2. 阶段 S1：尾类检查点选择

### 2.1 设置

- 与上一轮 A6-PU 使用完全相同的数据、初始化、DB 损失和局部解冻范围；
- 每个 epoch 记录 optimization-role SawFM AUPR；
- 20 epochs 全部执行；
- 使用 SawFM AUPR 而不是 macro-AUPR 保存检查点。

### 2.2 结果

| 指标 | 均值变化 | bootstrap 95% CI |
|---|---:|---:|
| macro-AUPR vs Base-ASL | +0.004709 | [-0.000313,+0.011752] |
| SawFM AUPR vs Base-ASL | -0.015148 | [-0.025029,-0.008676] |
| SawFM AUPR vs macro选择的A6-PU | +0.013479 | [+0.005046,+0.018044] |

逐 seed：

| seed | 尾类最优epoch | Δ macro-AUPR vs ASL | Δ SawFM AUPR vs ASL | 相对macro选模的SawFM增益 |
|---:|---:|---:|---:|---:|
| 2041 | 5 | +0.011752 | -0.011739 | +0.005046 |
| 2042 | 1 | -0.000313 | -0.025029 | +0.018044 |
| 2043 | 1 | +0.002690 | -0.008676 | +0.017348 |

结论：模型选择冲突确实存在，tail选择能明显优于macro选择；但三个种子的tail最优检查点仍全部低于Base-ASL，因此选模不是根本原因。

seed2042和2043在第1轮即达到SawFM最优，之后总体持续下降，说明DB微调从训练开始就沿着不利于SawFM的方向移动。

## 3. 阶段 S2：A6-D2

### 3.1 设置

| 模块 | 状态/学习率 |
|---|---:|
| 分类头 | 解冻，1e-3 |
| `temporal_backbone.5` | 解冻，1e-4 |
| `temporal_backbone.3` | 解冻，3e-5 |
| 更早层 | 冻结 |

继续使用完整DB和SawFM AUPR选模，不使用L2-SP或TailMix。每组可训练参数为19078。

### 3.2 结果

| 指标 | 均值变化 | bootstrap 95% CI |
|---|---:|---:|
| macro-AUPR vs Base-ASL | +0.004393 | [-0.001940,+0.013018] |
| SawFM AUPR vs Base-ASL | -0.018957 | [-0.030003,-0.012348] |
| SawFM AUPR vs A6-PU-tail选择 | -0.003809 | [-0.004974,-0.002782] |
| SawFM AUPR vs Base-DB | -0.042645 | [-0.053461,-0.028531] |

逐 seed：

| seed | 尾类最优epoch | Δ macro-AUPR vs ASL | Δ SawFM AUPR vs ASL | Δ SawFM AUPR vs D1 |
|---:|---:|---:|---:|---:|
| 2041 | 13 | +0.013018 | -0.014520 | -0.002782 |
| 2042 | 1 | -0.001940 | -0.030003 | -0.004974 |
| 2043 | 1 | +0.002100 | -0.012348 | -0.003672 |

增加一层适配没有修复SawFM，反而在三个种子上均低于单层tail选择版本。

## 4. 失败原因重新判定

### 4.1 选模冲突是次要原因

tail选择相对macro选择恢复了 `0.013479`，但最终仍低于Base-ASL `0.015148`。因此选模会放大失败，但不是失败起点。

### 4.2 不是适配深度简单不足

若问题只是最后一层表达能力不足，D2应优于D1；实际D2又下降 `0.003809`。继续机械增加解冻层数没有证据支持。

### 4.3 ASL到DB的微调路径进入错误方向

Base-DB从头训练时SawFM比Base-ASL高约 `0.023688`，但所有从Base-ASL出发的DB微调都下降：

| 路径 | Δ SawFM AUPR vs Base-ASL |
|---|---:|
| Head-DB20 | -0.011873 |
| A6-PU macro选择 | -0.028627 |
| A6-PU tail选择 | -0.015148 |
| A6-D2 tail选择 | -0.018957 |
| Base-DB从头训练 | +0.023688 |

这说明Base-ASL与Base-DB位于不同的优化区域。DB从随机初始化逐步形成的SawFM特征，不能通过在已收敛ASL模型上进行小范围DB微调获得。

### 4.4 SawFM对早期优化路径敏感

seed2042和2043在D1、D2中均于第1轮达到tail最优，随后下降。这不是训练轮数不足，而是目标切换后的梯度立即破坏原有SawFM排序结构。

## 5. 下一步策略

不建议继续执行D3、全解冻微调或搜索更多L2-SP权重。下一阶段应停止“ASL表示后接DB微调”的路径，改为从训练开始构建DB友好的表示。

推荐两条候选：

### 5.1 从头训练的全标签ASL-DB混合

- 从随机初始化训练，不加载Base-ASL；
- 使用现有 `asl_db_mix`；
- 第一轮比较DB权重0.75和0.90；
- SawFM × 3 seeds；
- 与Base-ASL、Base-DB和A6-PU-tail选择比较。

目的：验证能否保留Base-DB的SawFM表征，同时获得ASL的整体稳定性。

### 5.2 反向解耦：DB表示 + ASL分类头

- Stage 1使用Base-DB端到端表示；
- Stage 2冻结表示，仅使用ASL重训分类头；
- 与当前失败的“ASL表示 + DB头”构成严格反向对照。

该方案直接利用现有证据中SawFM更强的DB表示，也是下一轮更有价值的机制试验。

## 6. 产物与审计

- 技术方案：`docs/implementation/route_a_sawfm_selection_and_d2_plan_20260723.md`
- S1 runner：`experiments/run_route_a_sawfm_tail_selection.py`
- S1汇总：`outputs/tapc/route_a_sawfm_tail_selection/route_a_sawfm_tail_selection_summary.json`
- S2 runner：`experiments/run_route_a_sawfm_d2.py`
- S2汇总：`outputs/tapc/route_a_sawfm_d2/route_a_sawfm_d2_summary.json`

审计结果：S1和S2各3组正式结果齐全；样本数均为2400，标签计数误差为0；S1只有最后时序块和分类头变化，S2只有最后两个时序块和分类头变化；locked test使用次数为0。

S1汇总SHA256：`3b836603ac6c9d04369095b79bd2e69f06ca5962f932957811e023bdd0f2173f`。

S2汇总SHA256：`3c4f0e9ee8dce2f009ca67624ed652ef24d0c95902fc340edd602eeb0fcd8955`。

全量测试结果：`123 passed`。
