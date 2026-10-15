# 路线 A：DFMS-IQ 标签条件池化、局部证据与早期 SawFM 专家结果（2026-07-24）

## 1. 执行结论

本轮完成：

- 1 seed GPU smoke：C0/Q1/Q2/Q3完整链路；
- 3 seeds 正式结构筛选；
- 9个新增模块训练检查点；
- 标签条件注意力熵、top-k覆盖、残差尺度和专家输出诊断；
- 逐级残差 logit 后验分析；
- 全程 `locked_test_used=false`。

三个新增模块均未通过预注册门槛，完整结构决策为 `stop_before_unfreeze`，因此没有解冻 F2 的 stage4 或 temporal backbone。

核心结论是：**当前问题不是简单缺少更复杂的读取模块。Q1 标签条件残差在 label-LT 训练清单上学习了很强的逐类别 logit/prior 偏移，而不是稳定的弱局部证据；Q2 使用同一有偏查询选择 top-k，无法恢复排序；Q3 虽然真正从 stage2 独立分叉，但专家 logit 缺乏幅度约束并主要学习强 SawFM 模式，最终破坏弱正—负排序。**

## 2. 冻结结构与消融

所有变体共享并冻结：

- F2_DFMS_Full主干；
- H2 pos-weight BCE线性头；
- F2全局128维特征标准化参数；
- 对应 seed 的 split、manifest 和检查点。

增量变体：

| 变体 | 新增模块 | 新增总参数 | 当前阶段训练参数 |
|---|---|---:|---:|
| C0 | 无，复用H2 | 0 | 0 |
| Q1 | 六标签条件weighted mean/std | 25,548 | 25,548 |
| Q2 | Q1 + 每标签top8 mean/max | 42,520 | 16,972，仅local模块 |
| Q3 | Q2 + stage2 SawFM envelope/IF专家 | 62,010 | 19,490，仅Saw专家 |

Q2初始化自Q1并冻结Q1；Q3初始化自Q2并冻结Q1/Q2，因此逐级差值可以直接归因于新加入模块。

## 3. 三随机种子正式结果

以下均使用逐标签 `negative FPR≤0.20` 的 threshold-selection 规则。

| 变体 | Macro-AUPR | Hamming↓ | 弱召回 | 弱正/负 AUROC | Negative FPR | SawFM-AUPR | SawFM弱AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 H2 | **0.8670** | **0.2356** | 0.3774 | **0.6524** | 0.1875 | **0.7709** | **0.5203** |
| Q1 LabelQuery | 0.8549 | 0.2430 | 0.3676 | 0.6119 | **0.1750** | 0.7339 | 0.4527 |
| Q2 LocalTopK | 0.8537 | 0.2424 | 0.3784 | 0.6073 | 0.1789 | 0.7301 | 0.4505 |
| Q3 EarlySawExpert | 0.8532 | 0.2456 | **0.3844** | 0.5973 | 0.1865 | 0.7272 | 0.4446 |

Q3 的弱召回略高于 C0，但弱正—负 AUROC降低0.0550、Macro-AUPR降低0.0138、Hamming增加0.0100。这表示阈值附近预测数量发生变化，不是弱证据可分性改善。

## 4. Q1 标签条件池化：主要失败源

Q1相对C0：

- Macro-AUPR：`-0.01210`；
- Hamming：`+0.00741`；
- 弱分量召回：`-0.00980`；
- 弱正—负 AUROC：`-0.04050`，0/3 seeds为正；
- SawFM-AUPR：`-0.03706`；
- SawFM弱AUROC：`-0.06758`；
- negative FPR：`-0.01256`。

虽然误报略降，但排序和主要性能同步下降，Q1未通过任何性能晋级项。

### 4.1 残差没有保持“小修正”

query scale 从初始约0.05增长到：

- seed2041：约0.28～0.40；
- seed2042：约0.31～0.52；
- seed2043：约0.17～0.29。

early角色上的 Q1-C0 原始 logit 均值偏移呈稳定的类别方向：

| seed | FMB | SawFM | SinFM | VFT | ISRJ |
|---|---:|---:|---:|---:|---:|
| 2041 | +1.83 | -3.30 | +1.50 | -2.35 | -3.13 |
| 2042 | +3.19 | -5.33 | +1.31 | -2.90 | -4.37 |
| 2043 | +1.46 | -2.73 | +1.07 | -0.69 | -2.02 |

这与 label-LT 中 FMB/SinFM较多、SawFM/ISRJ较少的方向一致。Q1实际上学到了逐标签先验偏移；一个标量温度无法消除这种逐标签偏置。

### 4.2 注意力不等于弱证据定位

归一化注意力熵约0.52～0.99：

- 部分标签几乎保持均匀，未形成明确局部定位；
- 部分标签出现集中，但训练目标只要求标签分类，没有要求集中区域对应弱分量；
- 强正样本仍是主要正样本，查询自然优先学习强模式或组合共现。

逐标签看，Q1对SinFM、MFT弱AUROC有小幅提高，但SawFM和ISRJ分别下降约0.0676和0.0437，说明共享训练目标不能保证所有标签查询都寻找弱证据。

## 5. Q2 top-k局部旁路：增加召回但未改善排序

Q2相对Q1：

- Macro-AUPR：`-0.00125`；
- 弱召回：`+0.01080`，2/3 seeds为正；
- 弱正—负 AUROC：`-0.00458`，0/3 seeds为正；
- negative FPR：`+0.00396`；
- SawFM弱AUROC：`-0.00223`。

每个样本六标签top-k位置的去重覆盖约占时间轴8.5%～9.8%，说明该旁路确实选中了稀疏区域；但这些区域由已经产生先验偏移的Q1 scores决定，并没有弱证据监督。结果表现为少量召回变化，而不是弱正与负样本排序改善。

因此当前top-k机制不是无效，因为它没有运行，而是“选区依据错误”：标签分类查询分数不能直接当作弱证据热图。

## 6. Q3 stage2 SawFM专家：独立分叉成立，但输出失控

Q3相对Q2：

- Macro-AUPR：`-0.00048`；
- Hamming：`+0.00315`；
- 弱召回：`+0.00595`；
- 弱正—负 AUROC：`-0.00996`；
- SawFM弱召回：`+0.02484`；
- SawFM-AUPR：`-0.00289`；
- SawFM弱AUROC：`-0.00593`。

专家确实从stage2独立提取幅度包络差分、相位增量和周期复位描述，不再共享晚期pool/projection。但诊断发现：

| seed | tanh(saw scale) | 专家原始logit std | 最终Saw残差均值 | 最终残差std |
|---|---:|---:|---:|---:|
| 2041 | 0.073 | 6.30 | -0.87 | 0.46 |
| 2042 | 0.075 | 8.24 | -1.15 | 0.62 |
| 2043 | 0.123 | 18.02 | -4.61 | 2.22 |

虽然scale数值不大，但专家logit无归一化、无幅度上界，尤其seed2043已经足以覆盖H2原分类边界。专家主要受普通SawFM正样本训练，仍由强SawFM模式主导，没有专门保证弱SawFM排序。

因此本轮只能证明“早期独立专家工程路径可训练”，不能证明其识别机制有效。

## 7. 晋级与解冻决策

- Q1门槛：未通过；
- Q2门槛：未通过；
- Q3门槛：未通过；
- 完整Q3相对C0门槛：未通过；
- 解冻决策：`stop_before_unfreeze`。

没有执行stage4或temporal backbone解冻。当前任何解冻都可能让主干配合错误的先验偏移和专家大logit，而不是提高弱证据提取。

## 8. 对“特征是否充分”的新结论

本轮进一步支持：

1. H2全局特征仍是当前最稳定表示；
2. 性能不足不能简单归因于“没有注意力、没有top-k、没有专家”；
3. 标签条件模块若只接受普通多标签监督，会学习label-LT先验和强正模式；
4. 新结构必须显式约束“残差幅度”和“弱正定位”，否则结构越复杂越容易破坏已有表示；
5. SawFM早期特征可能包含信息，但在重新接入主模型前必须先做oracle-positive/弱正专门可辨识性验证。

## 9. 下一步建议

当前应保留 `F2 + H2`，停止Q1/Q2/Q3当前版本。若继续该方向，应先执行更小的机制验证：

1. 查询残差改为逐标签零均值并使用 `0.20 × tanh(raw_residual)` 硬幅度上界，禁止学习大类先验偏移；
2. 查询训练加入弱正定位或强/弱配对约束，不能只使用普通pos-weight BCE；
3. top-k依据改为经过弱正验证的局部热图，不能直接复用分类query score；
4. Saw专家输出先做LayerNorm/标准化和有界残差，再进行oracle-positive弱SawFM与负样本可辨识性测试；
5. 只有专家独立AUROC和AUPR超过冻结H2的SawFM弱证据门槛，才允许重新接入主分类器。

## 10. 产物

- 技术方案：`docs/implementation/route_a_dfms_iq_label_local_saw_expert_plan_20260724.md`
- 新模块：`cjlt/models/evidence_readout.py`
- 执行器：`experiments/run_route_a_dfms_iq_label_local_saw.py`
- 残差分析器：`experiments/analyze_route_a_dfms_iq_label_local_saw.py`
- 正式汇总：`outputs/tapc/route_a_dfms_iq_label_local_saw/route_a_dfms_iq_label_local_saw_summary.json`
- 残差诊断：`outputs/tapc/route_a_dfms_iq_label_local_saw/residual_logit_diagnostics.json`
- GPU smoke：`outputs/tapc/route_a_dfms_iq_label_local_saw_smoke/`

以上均为development证据，不是锁定测试集结果。
