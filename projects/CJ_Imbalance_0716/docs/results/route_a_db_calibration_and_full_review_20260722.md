# 路线 A DB 校准结果与现有全部实验复盘

日期：2026-07-22  
环境：`D:\anaconda3\envs\PY10\python.exe`  
数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`  
边界：只使用 development roles；locked test 未读取

## 1. 本段执行结果

本段完成四个尾类、ASL/DB、五个种子的 40 个正式 run。每个 run 均保存模型、history、温度、五套阈值、五套 early-stop 指标、split/manifest/config hash 和成功 receipt。

预注册主模式为 `shrinkage_per_label_f1`。最终五项门槛通过四项：

| 门槛 | 结果 | 判定 |
|---|---:|---|
| DB 主模式相对自身 fixed-0.5 的 tail recall 提升 ≥0.10 | +0.08122 | 未通过 |
| DB 主模式相对自身 fixed-0.5 的 Hamming 恶化 ≤0.01 | -0.02746 | 通过 |
| DB-ASL 主模式 macro-F1 ≥-0.01 | +0.01326 | 通过 |
| DB-ASL macro-AUPR >0 | +0.03451 | 通过 |
| DB tail-AUPR 至少 3/4 尾类不低于 ASL | 3/4 | 通过 |

严格结论：DB 校准实验未通过全部预注册门槛，因此本轮不能直接宣告路线 A 最终定型。但失败只来自 recall 提升阈值差 `0.01878`；其他排序、F1 和错误率证据均为正。

## 2. 主模式总体指标

以下是 20 个 ASL run 与 20 个 DB run 的均值和标准差：

| loss | macro-AUPR | tail-AUPR | macro-F1 | Hamming | tail recall | tail precision | tail F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASL | 0.87734 ± 0.04904 | 0.94847 ± 0.03866 | 0.82092 ± 0.02145 | 0.20142 ± 0.03261 | 0.80394 ± 0.08514 | 0.93032 | 0.85900 |
| DB | **0.91185 ± 0.03365** | **0.95607 ± 0.02994** | **0.83418 ± 0.02176** | **0.18342 ± 0.02887** | **0.80407 ± 0.07631** | **0.95265** | **0.87043** |
| DB-ASL | **+0.03451** | +0.00760 | **+0.01326** | **-0.01800** | +0.00013 | +0.02233 | +0.01142 |

配对 bootstrap 95% 区间：

- macro-AUPR 差：`[+0.02393, +0.04580]`，稳定为正；
- tail-AUPR 差：`[-0.00090, +0.01605]`，包含 0；
- 主模式 macro-F1 差：`[+0.00415, +0.02134]`，稳定为正；
- 主模式 tail recall 差：`[-0.02565, +0.02583]`，DB 与 ASL 实质持平。

因此当前最可靠的结论是：DB 改善整体排序、macro-F1、Hamming、尾类 precision 和尾类 F1；校准把原先显著落后的 tail recall 拉回到 ASL 水平，但尚不能证明 DB 提高 tail recall。

## 3. 五种阈值模式

| loss / mode | macro-F1 | Hamming | tail recall | tail precision | tail F1 |
|---|---:|---:|---:|---:|---:|
| ASL fixed-0.5 | 0.75957 | 0.33421 | **0.92111** | 0.75827 | 0.82849 |
| DB fixed-0.5 | 0.80775 | 0.21088 | 0.72285 | **0.98453** | 0.82838 |
| ASL global | 0.79487 | 0.22483 | 0.70034 | 0.96866 | 0.80352 |
| DB global | 0.80960 | 0.21096 | 0.72920 | 0.97655 | 0.82936 |
| ASL per-label F1 | 0.82435 | 0.19962 | **0.85935** | 0.88549 | 0.86929 |
| DB per-label F1 | **0.83828** | 0.17867 | 0.84479 | 0.91707 | **0.87732** |
| ASL shrinkage F1 | 0.82092 | 0.20142 | 0.80394 | 0.93032 | 0.85900 |
| DB shrinkage F1 | **0.83418** | 0.18342 | **0.80407** | **0.95265** | **0.87043** |
| ASL Youden | 0.81322 | 0.19413 | 0.80929 | 0.93127 | 0.86411 |
| DB Youden | 0.82767 | **0.17638** | 0.80895 | 0.94392 | 0.86916 |

逐类 F1 阈值使 DB 相对自身 fixed-0.5 的 tail recall 平均提高 `+0.12195`，19/20 个 run 为正，同时 Hamming 改善 `-0.03221`。它通过了 recall 数值门槛，但该模式不是本轮预注册主检验，不能用它事后替换主结果。

收缩模式的 beta=100 对尾类阈值偏保守：它将 DB tail recall 从逐类 F1 的 0.84479 压到 0.80407，同时把 precision 从 0.91707 提高到 0.95265。这说明当前剩余矛盾主要是 precision-recall 工作点选择，而不是 DB 表征能力不足。

## 4. 分尾类分析

| 尾类 | ASL/DB macro-AUPR | ASL/DB tail-AUPR | 主模式 DB-ASL macro-F1 | 主模式 DB-ASL tail recall |
|---|---:|---:|---:|---:|
| FMB | 0.85108 / **0.91594** | 0.97594 / **0.98412** | +0.02223 | +0.02675 |
| SinFM | 0.83258 / **0.86852** | **0.97104** / 0.96619 | +0.00788 | -0.05419 |
| MFT | 0.90026 / **0.92761** | 0.94616 / **0.95643** | +0.01509 | +0.00781 |
| ISRJ | 0.92543 / **0.93531** | 0.90074 / **0.91755** | +0.00783 | +0.02017 |

SinFM 是唯一没有 tail-AUPR 优势的尾类，也是主模式 tail recall 唯一下降明显的尾类。后续实验必须把 SinFM 作为预注册困难尾类单独报告，不能只汇总四类均值。

## 5. 温度与分数分布

ASL 的平均温度为 `1.0775 ± 0.2340`，DB 为 `0.3912 ± 0.0938`。DB 的 calibration BCE 平均下降 `0.06607`，远大于 ASL 的 `0.00470`。

温度小于 1 表明 DB 原始 logits 偏压缩，温度缩放需要把其拉开。温度缩放本身不改变 0.5 阈值对应的 logit 正负号，因此 fixed-0.5 recall 缺口主要仍需逐类阈值移动解决。DB 在 fixed-0.5 下呈现“极高 precision、偏低 recall”，与这一判断一致。

## 6. 现有路线 A 实验总览

| 阶段 | 核心结果 | 当前判断 |
|---|---|---|
| Phase-A Power-LT | T0P 相对 T0 macro-AUPR +0.07634、weak recall +0.11628 | 同身份多功率视图是有效增强，但不能证明标签长尾方法 |
| T2/T2S ordering | 方向 margin 和满足率提高，macro-AUPR、弱成分召回下降；stability 未激活 | 物理目标可学习，但与分类目标不一致 |
| T2O-v2/T2D | 单侧 ordering 修复部分弱召回；T2D 单 seed 小幅提高 | 进入多 seed 验证，不足以定论 |
| Phase-B 三 seed | T2D-T0P macro-AUPR 仅 +0.000097，F1/weak recall 无收益 | T2D 只保留无损消融，不是主创新 |
| ordering 权重/梯度解耦 | Shared 提高方向指标但损害 Hamming；StopGrad/Delayed 未恢复 | 排除“简单反向梯度冲突”，ordering 暂停并入主干 |
| 真实 label-LT 初筛 | DB 三 seed macro-AUPR +0.00582，固定阈值 recall 明显下降 | DB 成为候选，必须校准 |
| 四尾类两 seed | DB 8/8 提高 macro-AUPR、7/8 提高 tail-AUPR | 收益可跨尾类复现，扩大到五 seed |
| 本轮 40-run 校准 | DB 20/20 提高 macro-AUPR；主模式 F1/Hamming/precision/tail-F1 均优于 ASL，tail recall 持平 | 当前路线 A 最强证据，但 tail-AUPR CI 和 SinFM 仍是缺口 |

路线 A 的研究主线已经从复杂 TAPC 辅助项收敛到更清晰的命题：真实 label-LT 下，DB 提高排序与整体决策质量，独立阈值校准修复其保守输出。ordering、stability 和 T2D 不应继续占据论文主创新位置。

## 7. 现有路线 B 实验总览

| 阶段 | 观察 | 原因判断/决策 |
|---|---|---|
| 初始 Mean Teacher | 伪标签 coverage 全开或全关 | 预热不足、固定阈值与动态控制器失配 |
| 公平曝光 V2 | B1/B2/B3 coverage 为 0，仍比 B0 差 | 空 mask 时 strong-view 前向改变 BN/随机轨迹 |
| 预训练 Phase-B | coverage 恢复，但 B1/B2 比 B0_CONT 低约 0.020 macro-AUPR | calibration 与未标注池分布失配，无监督项过强 |
| 协议纠错 | 初始伪标签 precision 通过 0.90；lambda=0.05/0.10 仍下降 0.018/0.021 | 错误不再是伪标签入口，而是 strong augmentation 硬 BCE 与监督目标冲突 |
| 无标签功率 SSL | SameSourcePower 线性探针 0.84720，显著高于随机初始化 0.61548 | 无标签预训练能改善表示 |
| SSL 对照 | SameViewAug 0.87068，高于 SameSourcePower；RandomSource 也达 0.80247 | 正确跨功率配对的独特贡献未建立 |
| 全量微调 | 四种初始化均约 0.904–0.907 macro-AUPR | 20% 标签足以抹平初始化差异 |

路线 B 当前没有形成 SCI 主创新闭环。硬伪标签路线已被多轮证据排除；功率自监督只证明“预训练有用”，没有证明“同源跨功率结构优于普通增强”，也没有形成最终微调收益。

## 8. 当前综合判断

### 已成立的证据

1. 数据原始根集均衡，人工 label-LT manifest 和角色隔离是长尾结论前提。
2. 同身份功率视图作为数据增强有效。
3. DB 在真实 label-LT 下稳定提高 macro-AUPR：本轮 20/20 配对为正，bootstrap 区间不跨 0。
4. 独立逐类阈值能修复 DB 的大部分固定阈值 recall 缺口，并改善 F1、Hamming 与 tail-F1。
5. 普通无标签对比预训练显著改善冻结表示。

### 尚未成立的证据

1. DB 对每一种尾类都提高 tail-AUPR：SinFM 为反例，整体 tail-AUPR 区间包含 0。
2. DB 提高 tail recall：当前只能证明校准后与 ASL 持平。
3. ordering 提高最终识别：所有改法均有分类代价。
4. T2D 提供有意义的识别增益：三 seed 差异近似为零。
5. 同源跨功率 SSL 优于普通增强或提高最终微调性能。
6. 半监督硬伪标签方法有效。

## 9. 下一步建议

路线 A 优先进行外部确认式扩展，而不是继续堆模块：

1. 将本轮诊断得到的 `per_label_f1` 固定为候选阈值算法，在未参与本轮选择的新条件 IF=2、IF=5 和新 seeds 上验证；不得再根据这些结果更换阈值算法。
2. 同时保留预注册 shrinkage 模式，形成“保守高 precision”和“均衡 F1”两个部署工作点。
3. 对 SinFM 做分组合、功率和弱分量误差分析，判断其 tail-AUPR 下降来自共现组合变化还是损失本身。
4. IF=2/5 仍成立后，再扩展六分类和五种子；最后才冻结模型与阈值并申请一次 locked-test 评估。

路线 B 若继续，只做低成本机制筛选：SameViewAug 主目标 + identity-aware multi-positive/false-negative masking + 功率等级辅助头，在 5% 标签和 held-out power 上比较。若不能超过 SameViewAug，则终止路线 B 主创新定位。

## 10. 产物

- 冻结方案：`docs/implementation/route_a_db_calibration_experiment_plan_20260722.md`
- 本轮原始汇总：`outputs/tapc/route_a_db_calibration/route_a_db_calibration_summary.json`
- 本轮 runner：`experiments/run_route_a_db_calibration.py`
- 各 run：`outputs/tapc/route_a_db_calibration/{tail}/{asl,db}/seed_*`
- 路线 B SSL 汇总：`outputs/student_b/route_b_power_ssl_multiseed_summary.json`

所有正式汇总均记录 `locked_test_used=false`。128 样本不可行的 label-LT 冒烟不计入实验结论；1024 样本 1-epoch 冒烟只用于验证代码链路，也不计入统计。
