# Route A H1j：复合源 VFT 去啁啾统计量验证结果

## 结论

H1j **不授权**进入六类网络融合。正确 LFM 去啁啾相邻相位增量在本次新复合源的固定评估行上呈正向点估计，但其样本级、配对、分层 bootstrap 置信区间跨越零；最初以五个优化种子计算的门槛不具备独立重复含义，因为一维单调分类头不改变排序。

这不是对 H0i 的否定：H0i 仍表明该物理量在受控 VFT 单标签高/低多普勒干预中可识别。H1j 说明在正常六类复合、弱 VFT 场景中，现有的一维读出尚不足以给出可靠的增量证据。

## 固定协议和数据边界

- 新生成、仅训练来源：`5040` 个样本，`63 × 80` 六标签组合；SHA-256 为
  `33de4102ccd871a9d45dec97974a81fc6890597b9da3fc28cabce5d3ad5d3070`。
- `receiver_noise_seed=998677743--998682782`，全部唯一；fit/validation/evaluation
  为 `3024/1008/1008`，每个标签在三个角色中均平衡。
- 审计显示它与所有历史 live train 来源的身份交集均为零。
- 候选为已知 0716 LFM 参考的去啁啾相邻相位增量；控制为同长度、同模、固定相位置乱的参考。两臂均是以 fit 集均值/标准差标准化的一维、零初始化 `Linear(1,1)` BCE 读出，AdamW，学习率 `1e-2`，24 epoch，batch 64；不早停、不校准、不选阈值、不融合。
- 评估只比较 VFT 弱正与其他五标签上下文完全相同的 VFT 缺失负例。固定评估集合含 `107` 个弱正、`480` 个匹配负、`30` 个匹配上下文。
- 未打开或使用 historical locked-test、test-v2、test-v3 或 H1j 外的测试角色；test-v3 状态仍为 `access_authorized=false`、`analysis_allowed=false`、`used=false`。

## 原始预注册运行

五个固定优化种子 `2039--2043` 的点估计一致：

| 指标（候选 − 相位置乱控制） | 点估计 |
| --- | ---: |
| 弱 VFT AUROC | `+0.048715` |
| 弱 VFT AUPR | `+0.017996` |
| 强 VFT AUROC | `+0.219400` |

原运行器曾将五种子均为正、并基于种子值计算的区间作为 H1 gate 通过条件。该原始 summary 与每种子回执被完整保留，不能删除或改写。

## 推断复核（取代“种子 bootstrap”解释）

一维读出在五个种子下虽有略不同的权重和偏置，但它们都保持同一分数排序；AUROC/AUPR 因而几乎完全相同。种子级“bootstrap”退化为近零方差，不能量化固定评估样本的抽样不确定性。

在**不重新训练、不重选模型、不重算任何阈值**的前提下，固定 seed-2039 两臂分数进行了 10,000 次类别分层、成对重采样。每一抽样行同时保留候选与控制分数：

| 指标（候选 − 控制） | 点估计 | 样本级配对 bootstrap 95% CI |
| --- | ---: | --- |
| 弱 VFT AUROC | `+0.048715` | `[-0.032188, +0.132208]` |
| 弱 VFT AUPR | `+0.017996` | `[-0.028179, +0.069921]` |

两个区间均跨零。因此，H1j 只能记为**方向性、未确认的复合源 VFT 物理线索**，而不是可靠弱分量改进，更不能作为论文中“已证明”的 VFT 可见度模块。

## 机制解释与后续约束

1. H0i 使用人为受控的单标签多普勒差异；H1j 的 VFT 信号在复合干扰、弱可见度和上下文配对条件下，统计量的方差显著扩大。
2. 该标量只保留全脉冲的一个相位增量平均值，不能选择局部有效区间，也未显式解耦其余五种干扰的相位结构。
3. 不能在 H1j 身份上继续调节参考相位、窗口、频带、标准化、学习率、epoch、阈值或融合权重；这些身份已提供了一次方向性评估。
4. 若重新开展 VFT，需要是不同的、先验固定的物理可观测量（例如局部时频轨迹一致性，而非当前全局标量），并在新的身份隔离来源上先做 H0/H1 诊断；候选必须预注册样本级配对置信区间作为通过条件。

## 产物

- 计划：`docs/plans/route_a_h1j_vft_dechirped_composite_source_20260802.md`
- 生成审计：`outputs/tapc/route_a_h1j_vft_dechirped_composite_generation/identity_audit.json`
- 原始五种子 summary：`outputs/tapc/route_a_h1j_vft_dechirped_composite/route_a_h1j_vft_dechirped_composite_summary.json`
- 固定分数推断回执：`outputs/tapc/route_a_h1j_vft_dechirped_composite/h1j_fixed_score_stratified_bootstrap.json`
- 执行/推断脚本：`experiments/run_route_a_h1j_vft_dechirped_composite.py`、`experiments/analyze_route_a_h1j_vft_dechirped_composite_bootstrap.py`
