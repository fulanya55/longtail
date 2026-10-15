# 路线 A：IF=4 表征—分类头解耦实验结果（2026-07-23）

## 1. 核心结论

本阶段完成 27 次正式分类头训练，并复用 Base-ASL、Base-DB、端到端 Mix-0.25 共 27 个对应基线。每个新模型从对应 ASL 最优检查点初始化，冻结共享表征与 BatchNorm 状态，只训练 260 个分类头参数；全部运行使用相同 split/manifest，locked test 未使用。

预注册主候选 Head-Mix25 未通过，但机制诊断分支 Head-DB 得到明确成功结果：

> ASL 负责学习共享表征，随后冻结表征并仅用 DB 训练分类头，可以保留尾类性能，同时恢复 DB 的 macro-AUPR 收益。

该结果支持原始端到端 DB 在 IF=4 的主要失败原因是重平衡梯度破坏共享表征，而不是 DB 分类边界本身无效。

## 2. Head-DB 相对 Base-ASL

| 尾类 | Δ macro-AUPR（bootstrap 95% CI） | Δ tail-AUPR（bootstrap 95% CI） | Δ macro-F1 | Δ Hamming |
|---|---:|---:|---:|---:|
| FMB | +0.02407 [+0.01632, +0.02812] | +0.00362 [+0.00227, +0.00598] | +0.00639 | -0.01583 |
| MFT | +0.00518 [-0.00128, +0.01476] | +0.00090 [-0.00100, +0.00311] | +0.00067 | -0.01000 |
| ISRJ | +0.00617 [+0.00160, +0.01358] | +0.00026 [-0.00227, +0.00257] | -0.00121 | -0.00056 |

三个尾类总体平均：

- macro-AUPR：+0.01181；
- tail-AUPR：+0.00159。

在主要失败标签 MFT、ISRJ 上：

- macro-AUPR：+0.00567；
- tail-AUPR：+0.00058。

MFT、ISRJ 的 tail-AUPR 增益区间仍跨越零，因此当前更准确的表述是“尾类非劣，同时获得 macro 排序提升”，而不是已经证明尾类显著提高。

## 3. 相对端到端 DB 的修复

Head-DB 相对 Base-DB 的 tail-AUPR：

| 尾类 | Δ tail-AUPR | 获胜 seeds |
|---|---:|---:|
| FMB | +0.00289 | 2/3 |
| MFT | +0.03998 | 2/3 |
| ISRJ | +0.02367 | 3/3 |

MFT、ISRJ 的尾类损失被大幅恢复，而分类头仍然使用完全相同的 DB 公式。二者唯一的核心区别是 DB 是否能够反向更新共享表征，因此该对照为“端到端 DB 表征损伤”提供了直接证据。

## 4. 相对端到端 Mix-0.25

Head-DB 相对端到端 Mix-0.25：

| 尾类 | Δ macro-AUPR | Δ tail-AUPR |
|---|---:|---:|
| FMB | +0.01698 | +0.00288 |
| MFT | +0.00687 | -0.00523 |
| ISRJ | +0.00467 | -0.00037 |

Head-DB 在三个标签上均获得更高 macro-AUPR；MFT tail-AUPR 比端到端 Mix-0.25 低约 0.00523，位于预设非劣边界附近。这说明端到端 Mix-0.25 更偏向尾类保护，而解耦 Head-DB 更好地恢复整体排序。

## 5. Head-Mix25 为什么失败

Head-Mix25 的 tail-AUPR 均高于 Head-ASL，但其相对 Base-ASL 的 macro-AUPR 为：

- MFT：-0.00854；
- ISRJ：-0.00568。

二者低于 -0.005 非劣门槛。Head-ASL 继续训练分类头本身也使 MFT、ISRJ macro-AUPR 分别下降 -0.00777 和 -0.00573，说明在已经收敛的 ASL 检查点上继续做 10 epochs 普通头训练容易牺牲全局排序。Mix-0.25 只改善了尾类边界，没有抵消该趋势。

Head-DB 的重平衡强度更大，反而使分类头跳出这一局部边界，同时由于 backbone 冻结，不再破坏尾类表征。

## 6. 成功与失败的最终归因

成功部分：

1. 受控 manifest 消除了样本总量混杂；
2. Mix-0.25 证明降低 DB 梯度可以保护尾类；
3. Head-DB 进一步证明 DB 更适合作为第二阶段分类边界优化器，而不是端到端表征损失；
4. 仅训练 260 个参数，训练成本低，所有比较均可从检查点和哈希复现。

失败部分：

1. 原始端到端 DB 在 IF=4 把收益转移给非尾类并损害 MFT/ISRJ；
2. 全局 Mix-0.50 对标签敏感，ISRJ 明显退化；
3. Head-Mix25 的尾类恢复未转化为 macro-AUPR 非劣；
4. 当前只有 3 seeds，MFT/ISRJ 的 tail-AUPR 区间仍跨零，不能直接进入 locked test。

## 7. 下一阶段技术方案

建议把路线 A 的正式结构更新为：

1. Stage 1：ASL 端到端训练共享表征；
2. Stage 2：冻结表征，使用 DB 微调分类头；
3. 推理时结构不增加，仍为原始 backbone + linear classifier；
4. 下一轮使用新 seeds 2039、2040，对 FMB、SinFM、MFT、ISRJ 做确认；
5. 新种子必须同时训练 Base-ASL、Base-DB 和 Head-DB，不能只挑选成功方法；
6. 5-seed 证据通过后，再冻结六类扩展方案；locked test 继续保持未使用。

该两阶段方法与经典解耦长尾训练思想相近，当前可作为强基线和路线 A 的稳定实现基础；若作为 SCI 创新点，还需要加入复合干扰共现结构专用机制，而不能只依赖“冻结 backbone + 重训分类头”。

## 8. 产物

- 技术方案：`docs/implementation/route_a_if4_decoupled_head_plan_20260723.md`
- 正式汇总：`outputs/tapc/route_a_if4_decoupled_head/route_a_if4_decoupled_head_summary.json`
- runner：`experiments/run_route_a_if4_decoupled_head.py`
- classifier-only 实现：`experiments/run_tapc.py`、`cjlt/models/student_a.py`
