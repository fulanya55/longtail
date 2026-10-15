# 路线 A：IF=4 ASL–DB 混合损失实验结果（2026-07-23）

## 1. 执行结论

本阶段完成 18 次正式新训练：`2 mix weights × 3 tail labels × 3 seeds`，并复用上一轮相同 manifest 下的 18 次 ASL/DB 基线。36 个比较运行均未使用 locked test。

严格预注册门槛未通过，不能把混合损失直接升级为已确认方法。但 Mix-0.25 明显修复了原始 DB 在 MFT、ISRJ 上的尾类伤害，是当前最有价值的候选：

- 相对 DB，MFT tail-AUPR：+0.04520；
- 相对 DB，ISRJ tail-AUPR：+0.02404；
- 相对 ASL，MFT tail-AUPR：+0.00612，3/3 seeds 为正；
- 相对 ASL，ISRJ tail-AUPR：+0.00063；
- 唯一未通过项是 MFT macro-AUPR 相对 ASL 为 -0.00170，而门槛要求严格大于 0。

因此本轮结论是：`λ=0.25` 成功完成尾类保护，但尚未证明能对每个失败尾类同时保留 macro-AUPR 正增益。

## 2. Mix-0.25 相对 ASL

| 尾类 | Δ macro-AUPR | Δ tail-AUPR | Δ macro-F1 | Δ Hamming | 判定 |
|---|---:|---:|---:|---:|---|
| FMB | +0.00709 | +0.00074 | -0.00241 | +0.00083 | sanity 通过 |
| MFT | -0.00170 | +0.00612 | +0.00539 | -0.01083 | 仅 macro-AUPR 未通过 |
| ISRJ | +0.00150 | +0.00063 | +0.00899 | -0.00750 | 通过 |

MFT 的 macro-AUPR bootstrap 95% CI 为 `[-0.00541, +0.00401]`，当前证据更接近“与 ASL 持平”，而不是确定性退化。MFT tail-AUPR 的区间为 `[+0.00213, +0.00981]`，尾类恢复方向较稳定。

在 MFT、ISRJ 两个失败标签上，Mix-0.25 相对 ASL 的平均结果为：

- macro-AUPR：-0.000096；
- tail-AUPR：+0.00338；
- 相对 DB 的 tail-AUPR：+0.03462。

在 FMB、MFT、ISRJ 三组上总体平均：

- macro-AUPR：+0.00230；
- tail-AUPR：+0.00250；
- 相对 DB 的 tail-AUPR：+0.02308。

## 3. Mix-0.50 的问题

| 尾类 | Δ macro-AUPR vs ASL | Δ tail-AUPR vs ASL | Δ macro-F1 | Δ Hamming |
|---|---:|---:|---:|---:|
| FMB | +0.01802 | -0.00537 | -0.00129 | -0.00250 |
| MFT | -0.00866 | +0.00811 | +0.00687 | -0.01139 |
| ISRJ | +0.00008 | -0.02893 | -0.02363 | +0.02556 |

Mix-0.50 在 ISRJ 上重新出现明显尾类伤害，并且 FMB tail-AUPR 也低于非劣界限。说明 DB 系数从 0.25 增至 0.50 后，类别间梯度转移再次占主导，单一全局较大权重不适合 IF=4。

## 4. 机制判断

1. 原始 DB 并非完全无效，其非尾类排序收益可以通过低权重保留；
2. `λ=0.25` 足以大幅恢复 MFT、ISRJ 尾类，但对 MFT 的全局 macro 排序没有稳定正增益；
3. `λ=0.50` 对标签语义敏感，尤其损害 ISRJ，说明最优 DB 强度不是全标签共享的常数；
4. 简单继续搜索更多 λ 会在当前 3 seeds 上形成超参数过拟合，不能作为有力创新证据。

## 5. 下一步建议

按预注册规则，本轮不继续搜索 `λ=0.1/0.2/0.3/...`。下一阶段应从全局损失插值转向解耦或标签自适应方案：

1. 以 ASL 训练共享表征，冻结或低学习率更新 backbone；
2. 仅对分类头实施 DB/Mix 重平衡，避免 DB 梯度破坏尾类表征；
3. 对每个标签依据阳性率、共现率和负样本数设置有界 DB 系数，而不是共享 λ；
4. 首轮仍使用 MFT、ISRJ，FMB 为 sanity check，复用完全相同的 manifest；
5. 主指标保持 tail-AUPR，macro-AUPR 采用非劣界限而不是追求小幅正增益。

若研究目标允许把 MFT macro-AUPR 的门槛从“严格正增益”改为“非劣于 -0.005”，则 Mix-0.25 已满足该工程标准；但该标准变更必须作为新的预注册确认实验，不能回溯修改本轮结论。

## 6. 产物

- 技术方案：`docs/implementation/route_a_if4_asl_db_mix_plan_20260723.md`
- 正式汇总：`outputs/tapc/route_a_if4_mix/route_a_if4_mix_summary.json`
- 实现：`cjlt/losses/classification.py`、`experiments/run_tapc.py`
- runner：`experiments/run_route_a_if4_mix.py`
