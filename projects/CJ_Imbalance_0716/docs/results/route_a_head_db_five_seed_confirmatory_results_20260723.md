# 路线 A：Head-DB 四尾类五随机种子确认结果（2026-07-23）

## 1. 最终判定

本阶段新增27次正式训练，结合既有结果形成 FMB、SinFM、MFT、ISRJ 各5个随机种子的 Base-ASL、Base-DB、Head-DB 完整配对，共20个三元组。

所有预注册门槛均通过，Head-DB 可从“诊断性候选”升级为路线 A 的四类确认方法：

> Stage 1 使用 ASL 端到端学习共享表征；Stage 2 冻结表征和 BatchNorm 状态，仅使用 DB 训练260个分类头参数。

locked test 在全部运行中保持未使用。

## 2. Head-DB 相对 Base-ASL

| 尾类 | Δ macro-AUPR（bootstrap 95% CI） | macro获胜 | Δ tail-AUPR（bootstrap 95% CI） | tail获胜 | Δ macro-F1 | Δ Hamming |
|---|---:|---:|---:|---:|---:|---:|
| FMB | +0.03041 [+0.02090,+0.04390] | 5/5 | +0.00231 [+0.00073,+0.00427] | 4/5 | +0.00682 | -0.01600 |
| SinFM | +0.02743 [+0.01185,+0.04343] | 4/5 | +0.00035 [-0.00017,+0.00081] | 4/5 | +0.00811 | -0.01150 |
| MFT | +0.00390 [-0.00236,+0.01053] | 3/5 | +0.00181 [+0.00014,+0.00336] | 4/5 | +0.00010 | -0.00600 |
| ISRJ | +0.00470 [+0.00044,+0.00949] | 4/5 | +0.00073 [-0.00101,+0.00246] | 3/5 | -0.00000 | -0.00167 |

四尾类20个配对总体结果：

- macro-AUPR：+0.01661，95% CI `[+0.00932,+0.02443]`；
- tail-AUPR：+0.00130，95% CI `[+0.00050,+0.00218]`；
- macro-F1：+0.00376，95% CI `[+0.00049,+0.00720]`；
- Hamming loss：-0.00879，95% CI `[-0.01421,-0.00346]`；
- tail-F1：+0.00319，95% CI `[+0.00029,+0.00590]`；
- macro-AUPR 获胜：16/20；
- tail-AUPR 获胜：15/20。

总体证据已经不再是单纯的“尾类非劣”：macro-AUPR、tail-AUPR、macro-F1、Hamming loss 和 tail-F1 的总体区间均支持 Head-DB。

## 3. 相对端到端 Base-DB 的机制证据

Head-DB 相对 Base-DB 的 tail-AUPR：

| 尾类 | Δ tail-AUPR | bootstrap 95% CI |
|---|---:|---:|
| FMB | +0.00293 | [+0.00028,+0.00607] |
| SinFM | +0.01507 | [-0.00257,+0.03171] |
| MFT | +0.02916 | [+0.00401,+0.05434] |
| ISRJ | +0.01346 | [-0.00070,+0.03109] |

20个配对总体 tail-AUPR 恢复为 +0.01515，95% CI `[+0.00617,+0.02536]`；tail-F1 为 +0.02584，95% CI `[+0.00978,+0.04222]`。

由于 Head-DB 和 Base-DB 使用相同 DB 公式，核心差别仅是 DB 是否能够更新共享表征，因此5-seed结果进一步支持：IF=4 下端到端 DB 的主要问题是表征梯度损伤，而不是 DB 分类边界无效。

## 4. 仍然存在的失败和边界

方法并非每个 seed 都获胜：

- SinFM seed2038 macro-AUPR：-0.00057；
- MFT seed2038/2040 macro-AUPR：-0.00128/-0.00565；
- ISRJ seed2040 macro-AUPR：-0.00217；
- ISRJ seed2038 tail-AUPR：-0.00227。

MFT 的 macro-AUPR 区间仍跨零，SinFM、ISRJ 的单尾类 tail-AUPR 区间也跨零。因此不能宣称对每个类别、每个随机种子都显著提高；准确结论是“四尾类总体稳定改善，各尾类满足预注册的正向/非劣门槛”。

## 5. 路线 A 当前技术结构

建议将路线 A 的四类方法冻结为：

1. 受控、哈希记录的 Label-LT manifest；
2. ASL 端到端训练共享复合干扰表征；
3. 冻结共享表征和 BatchNorm；
4. DB 仅优化最终分类头；
5. 阈值仍在独立 threshold-selection role 上确定；
6. locked test 继续冻结。

该结构没有增加推理参数或推理分支，第二阶段只训练260个参数。

## 6. 下一阶段建议

四类确认门槛已通过，可以进入六类开发扩展，但还不能直接使用 locked test：

1. 在六类 optimization role 上先执行 MILP 可行性搜索，重新冻结总样本数、尾类正例数和实际 IF；
2. 六类必须建立新的哈希 manifest，不能沿用四类样本索引；
3. 比较六类 Base-ASL、Base-DB、Head-DB，并轮换 SawFM、VFT 等新增尾类；
4. 先运行3 seeds筛选，通过后扩展5 seeds；
5. 六类及跨功率开发验证全部通过后，才讨论一次性 locked-test 评估。

Head-DB 与经典解耦长尾分类思想接近，当前证据足以作为强基线和稳定主干。若作为SCI核心创新，还需要在第二阶段引入复合干扰标签共现约束或标签条件化分类头，并证明优于普通解耦训练。

## 7. 产物

- 技术方案：`docs/implementation/route_a_head_db_five_seed_confirmatory_plan_20260723.md`
- 正式汇总：`outputs/tapc/route_a_if4_head_db_confirmatory/route_a_head_db_confirmatory_summary.json`
- runner：`experiments/run_route_a_head_db_confirmatory.py`
