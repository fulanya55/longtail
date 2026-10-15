# 两条路线下一阶段改进与筛选实验方案

日期：2026-07-22  
状态：已执行（路线 A：2 seeds；路线 B：3 seeds）  
环境：`D:\anaconda3\envs\PY10\python.exe`  
数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`  
边界：仅使用 development roles，不读取 locked test

## 1. 本阶段目标

本阶段只回答三个可证伪问题：

1. DB Loss 在真实 label-LT 上的收益是否能跨尾类复现，而不是 ISRJ 特例？
2. 功率 ordering 的分类退化是否与共享主干梯度冲突有关，停止其主干梯度后能否改善？
3. 不产生类别伪标签的同源跨功率自监督，能否学习出优于随机初始化和错误配对的表示？

不在本阶段扩展六分类、复杂 prototype、distribution alignment、参数头或 locked test。

## 2. 路线 A1：尾类轮换

共同配置：T0、label-LT、IF=3、标签计数容差 20%、20 epochs、development eval=300、optimization pool=1024。20% 是四种尾类均可由整数共现组合构造的最小共同容差。

| 因素 | 设置 |
|---|---|
| 尾类 | FMB、SinFM、MFT、ISRJ 轮换 |
| head-to-tail 顺序 | 其余标签保持确定顺序，指定标签放在最后 |
| loss | ASL、DB Loss |
| seeds | 2026、2027 |
| selection | macro-AUPR |

筛选门槛：DB Loss 至少在 3/4 个尾类上提高尾类 AUPR，且跨全部 run 的平均 macro-AUPR 不低于 ASL。

## 3. 路线 A2：功率 ordering 梯度解耦

共同配置：Power-LT、T0P 系列、seed=2026、20 epochs、25/30/35 dB 成对功率视图。

| 变体 | ordering | 主干梯度 | 启用时间 |
|---|---|---|---|
| T0P | 无 | 无 | 无 |
| Shared | strong-only | 允许 | 全程 |
| StopGrad | strong-only | 阻断 | 全程 |
| Delayed-StopGrad | strong-only | 阻断 | 后 50% 训练 |

每个 ordering 变体记录分类与 ordering 对共享 backbone 的梯度范数和余弦相似度。

筛选门槛：相对 T0P，macro-AUPR 下降不超过 0.005、Hamming 恶化不超过 0.01，同时 directional satisfaction 或 weak recall 有明确提高。仅通过门槛的变体才进入多 seed。

## 4. 路线 B：同源跨功率自监督最小原型

先只进行表示筛选，不直接全量微调。

### 4.1 预训练变体

| 变体 | 正样本对 |
|---|---|
| RandomInit | 不预训练 |
| SameViewAug | 同一 30 dB 信号的两个轻增强视图 |
| RandomSource | 不同源样本的跨功率错误配对 |
| SameSourcePower | 同一 `receiver_noise_seed` 的跨功率正确配对 |

使用相同复数 backbone 和相同投影头，采用归一化温度对比损失。预训练数据只来自 optimization role 的未标注功率视图。

### 4.2 表示评价

- 冻结 backbone；
- 使用相同 20% nested labeled subset 训练线性分类头；
- 使用 early-stop role 按 macro-AUPR 选择 checkpoint；
- 每种预训练使用相同的 probe 初始化 seed、epoch 和标注暴露量。

筛选门槛：SameSourcePower 的 linear-probe macro-AUPR 至少比 RandomInit 高 0.01，并且高于 RandomSource。未通过则停止该自监督目标，不进入全量微调。

执行时补充了全量微调诊断，但不改变上述表示筛选门槛。原因是首个种子的线性探针相对随机初始化提升较大，需要判断预训练优势能否在监督微调后保留。该扩展使用与线性探针完全相同的 20% labeled subset，并在三个种子上同时执行。

预训练池最终严格绑定为 `semi_plan["unlabeled_indices"]`；后续监督使用的 labeled subset 不参与自监督预训练。三个正式种子为 2026、2027、2028。

## 5. 输出与审计

- 每个 run 保存配置哈希、split/pair 哈希、history、best checkpoint 和指标；
- 路线 A 保存实际标签计数、尾类身份、校准后与固定阈值指标；
- 路线 B 保存预训练 loss、linear-probe 指标和表示正负对相似度；
- 所有汇总文件显式记录 `locked_test_used=false`。

## 6. 执行产物

- 路线 A：`outputs/tapc/route_a_next_stage/route_a_next_stage_summary.json`
- 路线 B 多种子：`outputs/student_b/route_b_power_ssl_multiseed_summary.json`
- 分析报告：`docs/results/two_route_next_stage_results_20260722.md`
