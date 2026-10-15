# 两条路线 Phase-B 实验结果与后续判断

日期：2026-07-22  
环境：`D:\anaconda3\envs\PY10\python.exe`  
数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`  
配置依据：`docs/implementation/two_route_phase_b_execution_plan_20260722.md`  
locked test：未读取

## 1. 结论

- 路线 A：T2D 的三种子 macro-AUPR 与 T0P 实质持平，但固定阈值 F1、弱分量召回和 Hamming 均未形成收益。T2D 可作为“跨功率表征蒸馏不显著伤害排序能力”的辅助消融保留，但当前证据不足以替换 T0P 主基线。
- 路线 A ordering：权重越大，方向 margin 和满足率越高，说明物理约束确实被学到；但固定阈值识别质量同步下降。0.10 通过原先三个显式门槛，却造成 macro-F1@0.5 明显下降和 Hamming 明显升高，因此不进入主网络，下一轮应隔离 ordering 梯度对共享 backbone 的干扰。
- 路线 B：预训练、Teacher 初始化、逐标签阈值和空掩码跳过均已生效，伪标签从零覆盖恢复为连续 12 个 epoch 非零覆盖。B1/B2 仍比同起点纯监督续训低约 0.020 macro-AUPR，未通过识别门槛。当前失败已从“链路未激活”转变为“阈值跨分布失配、正负伪标签逐步失衡和无监督损失过强”。

## 2. 路线 A：T0P 与 T2D 三种子结果

协议：Power-LT，训练/开发评估样本 1024/300，20 epochs，seeds 2026/2027/2028，功率 25/30/35 dB，参考功率 30 dB。

| 变体 | macro-AUPR | macro-F1@0.5 | Hamming@0.5 | weak recall@0.5 | cardinality MAE |
|---|---:|---:|---:|---:|---:|
| T0P | 0.974956 ± 0.003693 | 0.885534 ± 0.009975 | 0.133056 ± 0.010552 | 0.653107 ± 0.090601 | 0.494444 ± 0.041141 |
| T2D | **0.975053 ± 0.003649** | 0.883257 ± 0.007311 | 0.136111 ± 0.007281 | 0.646537 ± 0.084973 | 0.500000 ± 0.032830 |
| T2D - T0P | +0.000097 | -0.002276 | +0.003056 | -0.006570 | +0.005556 |

逐 seed 的 T2D macro-AUPR 均略高于 T0P，但差值只有 0.00004～0.00014；弱分量召回差值为 -0.0100、-0.0097、0。说明蒸馏分支具有稳定的“近似无损”表现，但尚无实际识别增益。

判定：

- macro-AUPR 门槛通过；
- 弱分量召回没有严重退化，但两种子轻微下降，不能宣称改善；
- T2D 作为辅助消融保留，T0P 继续作为路线 A 主基线。

## 3. 路线 A：ordering 权重筛选

固定 seed 2026，与同 seed T0P 比较。

| ordering weight | macro-AUPR | macro-F1@0.5 | Hamming@0.5 | weak recall@0.5 | margin mean | satisfaction |
|---:|---:|---:|---:|---:|---:|---:|
| T0P / 0 | 0.979171 | 0.890971 | 0.129167 | 0.720000 | 0.428674 | 0.635499 |
| 0.02 | 0.978216 | 0.885147 | 0.135833 | 0.680000 | 0.508637 | 0.678024 |
| 0.05 | 0.976812 | 0.880311 | 0.142500 | 0.710000 | 0.610959 | 0.726624 |
| 0.10 | 0.975216 | 0.862057 | 0.170833 | **0.790000** | **0.643066** | **0.756207** |

结果呈现清楚的权衡：ordering weight 从 0 增加到 0.10 时，margin mean 增加 0.214392，满足率增加 0.120708；与此同时 macro-F1@0.5 下降 0.028914，Hamming 增加 0.041667。

原冻结门槛只检查 AUPR、方向指标和 weak recall，因此 0.10 形式上通过；但固定阈值整体识别明显恶化，不能据此并入主模型。该结果支持下一轮采用独立 residual/order head、stop-gradient 或延迟启用 ordering，使物理约束不直接重塑共享分类表征。

## 4. 路线 B：预训练和阈值构造

B0_PRETRAIN 使用完整 20% nested labeled subset，共 840 个标注样本，训练 20 epochs。其 best macro-AUPR 为 0.967388。随后从同一 best checkpoint 分别启动 B0_CONT、B1 和 B2，保证 Student、Teacher 和纯监督基线具有相同初始状态。

校准集逐标签阈值如下：

| 标签索引 | tau_pos (p90) | tau_neg (p10) | 校准候选覆盖 | 正伪标签精度 | 负伪标签精度 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.899810 | 0.427291 | 0.20 | 1.00 | 1.00 |
| 1 | 0.897050 | 0.408586 | 0.20 | 1.00 | 1.00 |
| 2 | 0.889972 | 0.464624 | 0.20 | 1.00 | 1.00 |
| 3 | 0.862575 | 0.534290 | 0.20 | 1.00 | 1.00 |

这些精度只描述 300 个 calibration 样本，不代表未标注池上的真实伪标签精度。

## 5. 路线 B：续训结果

| 变体 | best epoch | macro-AUPR | macro-F1 | Hamming | mean coverage | max coverage | active epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0_CONT | 12 | **0.979444** | **0.822144** | **0.240833** | 0 | 0 | 0 |
| B1 | 2 | 0.959048 | 0.774189 | 0.326667 | 0.239886 | 0.537109 | 12 |
| B2 | 2 | 0.959453 | 0.774189 | 0.326667 | 0.237607 | 0.536272 | 12 |

相对 B0_CONT：

- B1 macro-AUPR 下降 0.020395；
- B2 macro-AUPR 下降 0.019991；
- B1/B2 均在第 2 epoch 达到最好值，第 3 epoch 后总体持续下降；
- B2 的平均 feature consistency loss 为 0.094695，确实被激活；其 AUPR 比 B1 高 0.000405，未进一步恶化，但也没有抵消伪标签分类造成的损失。

冻结门槛判定：连续覆盖和平均覆盖通过；B2 consistency 激活通过；相对 B0_CONT 的 AUPR 门槛失败。因此 B1/B2 均不进入扩大实验。

## 6. 路线 B 失败原因

### 6.1 校准分位数没有迁移到未标注池

虽然 calibration 上每标签候选覆盖都是 20%，B1 首轮未标注覆盖却分别约为 46.3%、98.5%、61.3%、8.7%，总覆盖达到 53.7%。这说明阈值在校准角色和未标注池之间发生明显分布失配，初始无监督信号已远强于计划值。

### 6.2 固定阈值无法跟随 EMA Teacher 概率漂移

以 B1 为例，标签 1 的 Teacher p10 从 0.2715 上升到 0.4305，逐步越过固定 tau_neg=0.4086，负伪标签由首轮 883 个降到末轮 4 个；标签 3 的 p90 从 0.8579 上升到 0.8938，越过 tau_pos=0.8626 后正伪标签不断增加。末轮共有 558 个正伪标签和 4 个负伪标签，训练信号接近单侧。

### 6.3 稳定性门控不是当前瓶颈

大多数 confidence candidate 均通过稳定性门控，后期每轮被 stability 拒绝的数量接近 0。继续放宽 stability ceiling 不会解决识别退化，反而可能扩大噪声。

### 6.4 无监督分类权重过强

当前 `lambda_u=1.0`，仅使用 20% continuation warmup。B1 平均无监督分类损失为 0.279866，B2 为 0.285108；在首轮超过 53% 且分布不均衡的覆盖下，无监督项迅速改变了预训练模型，导致 best epoch 提前到 2，之后 AUPR 持续下降。

## 7. 下一阶段建议

### 路线 A

1. T0P 继续作为主基线，T2D 仅作为辅助消融，不再单独扩大算力投入。
2. 新建与共享 backbone 解耦的 ordering residual/head；优先测试 stop-gradient 到 backbone，再测试仅最后若干 epoch 启用 ordering。
3. 保留 0.02/0.05/0.10 权重，但增加 macro-F1@0.5 和 Hamming 不显著退化的硬门槛，避免只满足方向指标。

### 路线 B

1. 在实际未标注池的 Teacher 概率上按每标签、正负两侧分别取分位数，只用 calibration 估计质量，不再用 calibration 分位数直接控制未标注覆盖。
2. 每个 epoch 重估分位数阈值或使用阈值 EMA，使每标签正、负覆盖分别保持在目标区间；设置每标签/每侧伪标签数量上限，禁止单侧伪标签占满。
3. 下一轮先筛 `lambda_u=0.10/0.25`，无监督 ramp 延长到至少 5 epochs；B2 只在 B1 不低于 B0_CONT 超过 0.01 后再启用。
4. 继续保留相同 B0 checkpoint、相同 labeled exposure 和空 accepted mask 跳过机制；prototype、distribution alignment、参数头仍冻结。

## 8. 结果文件

- 路线 A 汇总：`outputs/tapc/route_a_phase_b/route_a_phase_b_summary.json`
- 路线 A 明细：`outputs/tapc/route_a_phase_b/route_a_phase_b_runs.csv`
- 路线 B 汇总：`outputs/student_b/route_b_phase_b/seed_2026/route_b_phase_b_summary.json`
- 路线 B 阈值：`outputs/student_b/route_b_phase_b/seed_2026/quantile_thresholds.json`
- B1/B2 每轮过程：各自目录下的 `history.csv` 与 `pseudo_label_statistics.json`

所有结果均为 development 阶段证据，`locked_test_used=false`。
