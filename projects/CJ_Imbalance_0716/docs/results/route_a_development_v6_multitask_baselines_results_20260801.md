# Route A development-v6 通用多任务优化公平基线结果

日期：2026-08-01

## 1. 本轮结论

`FullAlpha1p5` 显著优于最强基线 `M5_GradientIsolated` 的 target weak recall：
`+0.00853`，95% CI `[+0.00384,+0.01408]`。方法优势门槛通过。

但候选相对 C6 的 target weak recall 为 `+0.02454`，低于预注册的 `+0.025`
效果门槛 `0.00046`，候选效果门槛失败。最终决定：`stop_innovation_route`。

## 2. 最近三轮实验

| 轮次 | macro-AUPR delta | macro-F1 delta | target weak delta | 结果 |
|---|---:|---:|---:|---|
| development-v4 | +0.00646 | +0.01163 | +0.01691 | 未过幅度门槛 |
| development-v5 | +0.00632 | +0.01464 | +0.02813 | 原 Full1p5 最优 |
| development-v6 | +0.00586 | +0.01266 | +0.02454 | 统计为正，但差 0.00046 未过门槛 |

## 3. 本轮科学问题

冻结 FullAlpha1p5 后，它是否优于 M5、loss weight、midpoint、PCGrad、MGDA、
CAGrad 和 GradNorm？

## 4. 可证伪假设

候选必须先稳定达到自身效果门槛，再以至少一个主要指标 `+0.005` 且 paired CI
下界大于 0 的方式超过 strongest baseline。第二部分通过，第一部分失败。

## 5. 数据角色与身份隔离

- development-v6 train-only，25,200 样本；
- sample seed `[54000001,54025200]`；
- receiver_noise_seed `[252677743,252702942]`，25,200 unique；
- SHA256 `1564d8ddea66ca746dc39e3076ef01f5c767fcc4b5c8d66d1a31e7181bffb8b0`；
- 与 v2--v5 及全部冻结/密封角色不重叠；
- `test_files_opened=false`。

## 6. 基线和候选

所有方法共享相同初始化、数据、batch 顺序、预算、trainable scope、early-stop、
calibration 和 threshold。两任务固定为 common-five BCE 与 `0.2*SawFM BCE`。

方法：C6、M5、LossWeight0p1、midpoint、PCGrad、MGDA、CAGrad0p4、GradNorm、
FullAlpha1p5。

## 7. 实现改动

- PCGrad 对两任务做对称冲突投影；
- MGDA 使用两任务解析最小范数权重；
- CAGrad 在固定 101 点 simplex 上求解；
- GradNorm alpha=1.5、weight lr=0.025；
- 记录冲突比例、权重、训练时间和 clipping；
- FullAlpha1p5 成本按 C6+M5 两个端点计。

## 8. 正式定量结果

| 方法 | macro-AUPR Δ | macro-F1 Δ | target weak Δ | absolute FPR |
|---|---:|---:|---:|---:|
| M5 | +0.00345 | +0.00802 | +0.01601 | 0.19629 |
| LossWeight0p1 | +0.00112 | +0.00296 | +0.00418 | 0.19317 |
| Midpoint0p5 | +0.00137 | +0.00368 | +0.00826 | 0.19570 |
| PCGrad | +0.00002 | -0.00007 | +0.00357 | 0.19672 |
| MGDA | +0.00182 | +0.00362 | +0.00328 | 0.19516 |
| CAGrad0p4 | +0.00102 | +0.00248 | +0.00216 | 0.19382 |
| GradNorm | +0.00020 | +0.00056 | +0.00001 | 0.19312 |
| **FullAlpha1p5** | **+0.00586** | **+0.01266** | **+0.02454** | **0.19473** |

候选目标标签 weak recall delta：MFT `+0.03878`、VFT `+0.01726`、ISRJ
`+0.01758`，全部为正。

## 9. 置信区间与种子一致性

FullAlpha1p5 相对 C6：

- macro-AUPR 95% CI `[+0.00514,+0.00672]`；
- macro-F1 95% CI `[+0.01000,+0.01597]`；
- target weak recall 95% CI `[+0.01257,+0.03651]`；
- positive weak seeds：5/5。

FullAlpha1p5 相对 M5：

- macro-AUPR `+0.00242`，CI `[+0.00203,+0.00278]`；
- macro-F1 `+0.00465`，CI `[+0.00300,+0.00639]`；
- target weak recall `+0.00853`，CI `[+0.00384,+0.01408]`；
- target FPR `-0.00156`。

## 10. 机制分析

1. 最强基线是简单 M5，不是 PCGrad/CAGrad 等梯度方法；
2. common 与 SawFM 梯度的 batch conflict fraction 约 `0.26--0.49`，但局部冲突
   投影只带来很小下游收益；
3. 这与早期 LSCP 负结果一致：瞬时梯度余弦不是长期表示效果的充分条件；
4. FullAlpha1p5 超过 M5，说明 alpha>1 的配对参数外推提供了额外价值；
5. 但不同 fresh development 上 weak effect 为 `0.01691--0.02813`，效应量不稳定。

## 11. 失败原因和替代解释

- 候选并非统计无效：所有主要 CI 均为正；失败来自预注册效果量下限；
- v6 距 `+0.025` 仅差 `0.00046`，但不能事后放宽；
- thresholded weak recall 对身份构成更敏感，而 macro-AUPR/F1 更稳定；
- 继续生成 fresh 数据直到通过会形成顺序试验偏差，因此停止；
- 两任务通用优化实现是当前网络上的公平 development baseline，不等于覆盖所有
  文献实现、架构和超参数。

## 12. 证据等级

本轮为 `development`。既有 test-v2 C 级结论保持冻结，但不能由本轮升级为更广泛
的通用创新或 external claim。

## 13. 预注册门槛

- candidate effect gate：9/10 通过；仅 target weak `0.02454 < 0.025` 失败；
- strongest baseline：M5；
- method advantage gate：8/8 通过；
- final decision：`stop_innovation_route`。

## 14. 产物路径与哈希

- plan：`docs/plans/route_a_development_v6_multitask_baselines_preregistration_20260801.md`；
- runner：`experiments/run_route_a_development_v6_multitask_baselines.py`；
- identity audit：`outputs/tapc/route_a_development_v6_generation/identity_audit.json`；
- summary：`outputs/tapc/route_a_development_v6_multitask_baselines/route_a_development_v6_multitask_baselines_summary.json`；
- receipt：`outputs/tapc/route_a_development_v6_multitask_baselines/run_receipt.json`；
- manifest：`outputs/tapc/route_a_development_v6_multitask_baselines/artifact_manifest.json`。

## 15. 下一步

停止当前创新路线，不进入分布/架构鲁棒性和外部一次性测试。可用于论文的结论是：

> 配对 SawFM label-deletion task vector 在当前协议中取得统计稳定的宏观增益，并
> 优于所比较的两任务优化基线；但弱分量效应量跨 fresh development 不足以稳定
> 达到预注册下限，因此尚不能作为可靠通用创新点。

