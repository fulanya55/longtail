# Route A development-v5 作用层与 alpha 结构化消融结果

日期：2026-08-01

## 1. 本轮结论

预注册选择仍选择原始 `FullAlpha1p5`。它在 fresh development-v5 上得到：

- macro-AUPR delta `+0.00632`；
- macro-F1 delta `+0.01464`；
- MFT/VFT/ISRJ mean weak recall delta `+0.02813`；
- target FPR `0.19253`；
- 三个目标标签 mean weak recall delta 全部为正；
- 5/5 种子的 target mean weak recall delta 为正。

没有新的 layer/alpha 方案把最差目标标签再改善 `+0.005`，所以改进晋级门槛
失败。当前 alpha 网格停止，不允许访问测试集。

## 2. 最近三轮实验

| 轮次 | macro-AUPR delta | macro-F1 delta | target weak recall delta | 结果 |
|---|---:|---:|---:|---|
| test-v2 once | +0.00648 | +0.01390 | +0.03048 | C 级冻结 |
| development-v4 | +0.00646 | +0.01163 | +0.01691 | 总体稳定，弱 Recall 未过幅度门槛 |
| development-v5 | +0.00632 | +0.01464 | +0.02813 | 原 Full1p5 最优，无更好 layer/alpha |

## 3. 本轮科学问题

C6→M5 方向的收益来自最后共享块、分类器还是两者组合；能否通过拆分恢复
MFT/VFT/ISRJ 的一致弱分量增益？

## 4. 可证伪假设

如果某个拆分层或自适应 alpha 比完整方向更合理，它应在保持 macro/FPR 的同时，
把三个目标标签中的最差 weak recall 至少提高 `+0.005`。正式结果否定了该假设。

## 5. 数据角色与身份隔离

- development-v5 train-only，25,200 样本；
- sample seed `[44000001,44025200]`；
- receiver_noise_seed `[242677743,242702942]`，25,200 unique；
- SHA256 `cb9649ef0cb8e7d8654affc6c8eec9b8807d597cf3e06b2de8c3847912a4c198`；
- 与 v2/v3/v4 和所有冻结/密封角色的声明范围不重叠；
- `test_files_opened=false`。

## 6. 基线和候选

固定网格包括：full alpha 0.5/1.5/2.0，shared-only 0.5/1.0/1.5/2.0，
classifier-only 0.5/1.0/1.5/2.0，shared=1.5 与 classifier=0.5/1.0 的
split 组合，以及内部 validation 自适应选择。

另设 `OrthogonalTaskSubspace1p5`：由五个 other-label directions 构造、与 SawFM
方向正交且全局等范数的结构化控制。

## 7. 实现改动

- 将非零方向严格拆成 `temporal_backbone.0.*` 和 `classifier.*`；
- 其余 state 固定为 C6；
- 自适应规则只使用内部 validation role；
- 结构化正交控制替代 v4 失稳的高维高斯随机方向；
- 保存每种子的方向范数、选择分数、checkpoint、校准、阈值和预测。

## 8. 正式定量结果

### FullAlpha1p5 绝对结果

| 指标 | C6 | FullAlpha1p5 | Delta |
|---|---:|---:|---:|
| macro-AUPR | 0.86367 | 0.86999 | **+0.00632** |
| macro-F1 | 0.74935 | 0.76400 | **+0.01464** |
| target weak recall | 0.29456 | 0.32269 | **+0.02813** |
| target FPR | 0.19145 | 0.19253 | +0.00108 |
| SawFM-present macro-F1 | -- | -- | **+0.02000** |
| SawFM-absent macro-F1 | -- | -- | **+0.00995** |

每标签 AUPR delta：FMB `+0.00548`、SinFM `+0.00069`、MFT `+0.00709`、
VFT `+0.01227`、ISRJ `+0.00609`。

## 9. 置信区间与种子一致性

- target weak recall delta 95% CI：`[+0.02288,+0.03285]`；
- macro-F1 delta 95% CI：`[+0.00938,+0.02090]`；
- macro-AUPR delta 95% CI：`[+0.00536,+0.00729]`；
- positive target weak recall：5/5 seeds。

逐标签 weak recall delta：

| 标签 | Mean | 95% CI |
|---|---:|---:|
| MFT | +0.05495 | [+0.03590,+0.07547] |
| VFT | +0.01879 | [+0.01335,+0.02423] |
| ISRJ | +0.01065 | [+0.00327,+0.02074] |

## 10. 机制分析

| 变体 | macro-F1 delta | target weak delta | 最差目标 weak delta | 结论 |
|---|---:|---:|---:|---|
| FullAlpha1p5 | +0.01464 | +0.02813 | +0.01065 | 最优稳定组合 |
| M5 endpoint | +0.01086 | +0.02199 | +0.00994 | 屏蔽 SawFM 本身已有较强效果 |
| ClassifierOnly2p0 | +0.01617 | +0.02029 | +0.00811 | 合格但弱分量均衡性较差 |
| SharedOnly1p5 | -0.00799 | -0.01019 | -0.03151 | 共享方向单独使用有害 |
| Split 1.5/1.0 | +0.00947 | +0.01689 | +0.00903 | 分类器幅度不足时整体门槛不足 |
| Orthogonal control | +0.00586 | -0.00290 | -0.03107 | 不能解释候选增益 |

关键机制判断：

1. 单独的 shared task direction 并不能解释成功，反而显著损害 MFT；
2. classifier direction 是主要正贡献来源之一；
3. 完整 shared+classifier 方向存在协同，尤其显著提高 MFT；
4. 因此当前证据不支持把方法简单称为“共享表示梯度冲突抑制”；
5. 更准确的机制是“严格配对的 SawFM label-deletion task vector 在共享块与分类器
   中形成协同参数外推”。

## 11. 失败原因和替代解释

- FullAlpha2p0 在仅 3/5 seeds 有限，说明继续增大 alpha 会进入不稳定区；
- validation 自适应在 4/5 seeds 选择 SharedOnly0p5，但 early 正式表现为负，说明
  validation weak-vs-negative AUROC 不能可靠选择最终阈值化 weak recall；
- classifier-only 能获得多数收益，审稿时必须与普通 logit/head 调整区分；
- M5 已达到 weak `+0.02199`，Full1p5 相对 M5 的额外优势有限，必须通过通用
  baseline 阶段证明方法价值。

## 12. 证据等级

本轮为 `development`。它增强了 FullAlpha1p5 的跨身份复现证据，但没有产生新的
独立测试结论，也没有授权访问任何测试集。

## 13. 预注册门槛

eligible candidates：

1. `FullAlpha1p5`；
2. `ClassifierOnlyAlpha2p0`。

固定选择规则选择 `FullAlpha1p5`。晋级门槛 8/9 通过，唯一失败项为：

`worst_target_improves_full1p5_by_0p005=false`。

这表示没有新 layer/alpha 候选优于既有方法，而不是既有 Full1p5 本身失效。

## 14. 产物路径与哈希

- plan：`docs/plans/route_a_development_v5_layer_alpha_preregistration_20260801.md`；
- identity audit：`outputs/tapc/route_a_development_v5_generation/identity_audit.json`；
- runner：`experiments/run_route_a_development_v5_layer_alpha.py`；
- summary：`outputs/tapc/route_a_development_v5_layer_alpha/route_a_development_v5_layer_alpha_summary.json`；
- receipt：`outputs/tapc/route_a_development_v5_layer_alpha/run_receipt.json`；
- artifact manifest：`outputs/tapc/route_a_development_v5_layer_alpha/artifact_manifest.json`。

## 15. 下一轮唯一最高优先级实验

停止当前 alpha/layer 搜索。使用 fresh development-v6，冻结原始
`FullAlpha1p5`，与公平配置的 `M5`、SawFM loss-weight、PCGrad、MGDA、CAGrad、
GradNorm、普通 midpoint 做通用基线比较。

本轮目标不再是寻找更高 alpha，而是回答：

> FullAlpha1p5 是否显著优于最强的简单屏蔽和通用多任务优化基线；若不能，则创新
> 应降级为特定协议下有效的 task-vector 技巧，而不能声称通用优化创新。
