# 路线 A：六类局部表征适配筛选结果（2026-07-23）

## 1. 最终判定

本阶段完成了 SawFM、VFT、ISRJ 三个困难尾类、3 个随机种子、3 种局部适配方法，共 27 个正式运行。

没有方法通过预注册的完整晋级门槛，因此本轮不扩展到六尾类或 5 seeds，也不使用 locked test。

最接近晋级的是 A6-PU：

- 相对 Base-ASL，macro-AUPR `+0.016521`，95% CI `[+0.011854,+0.021282]`，9/9 获胜；
- macro-F1 `+0.009310`，Hamming loss `-0.012901`；
- 相对 Head-DB20，tail-AUPR `+0.006738`；
- VFT、ISRJ 的 tail-AUPR 在 3/3 seeds 上均提高；
- 但 SawFM 的 tail-AUPR 在 3/3 seeds 上均下降，平均 `-0.028627`；
- 三尾类总体 tail-AUPR 相对 Base-ASL 为 `-0.000125`，未达到正向门槛。

结论是：只解冻最后时序块已经能修复 VFT、ISRJ，但对 SawFM 的适配方向错误，尚不能形成跨尾类统一方法。

## 2. 实验设置

| 项目 | 设置 |
|---|---|
| 数据 | `C:\LJQ_code\干扰数据\复合干扰数据0716` |
| 环境 | `D:\anaconda3\envs\PY10\python.exe` |
| 配置 | `configs/student_a/six_class.yaml` |
| 协议 | controlled Label-LT，N=2400，tail=400，IF=4 |
| 尾类 | SawFM、VFT、ISRJ |
| seeds | 2041、2042、2043 |
| 初始化 | 上一阶段对应 Base-ASL 检查点 |
| 解冻范围 | `backbone.temporal_backbone.5` 与分类头 |
| 可训练参数 | 12806 |
| 分类头学习率 | 1e-3 |
| 最后时序块学习率 | 1e-4 |
| epochs | 20 |
| 模型选择 | optimization role macro-AUPR |
| locked test | 未使用 |

三种方法：

- A6-PU：局部解冻 + DB；
- A6-PU-SP：A6-PU + L2-SP，权重 0.1；
- A6-TailMix：A6-PU-SP + 当前尾类 DB、其他标签 ASL。

## 3. 总体结果

### 3.1 相对 Base-ASL

| 方法 | Δ macro-AUPR | 95% CI | macro胜场 | Δ tail-AUPR | 95% CI | tail胜场 |
|---|---:|---:|---:|---:|---:|---:|
| A6-PU | +0.016521 | [+0.011854,+0.021282] | 9/9 | -0.000125 | [-0.015659,+0.013856] | 6/9 |
| A6-PU-SP | +0.009465 | [+0.004953,+0.013701] | 8/9 | -0.006880 | [-0.015043,+0.000718] | 2/9 |
| A6-TailMix | +0.006087 | [+0.002689,+0.009385] | 8/9 | -0.008637 | [-0.016808,-0.000551] | 2/9 |

A6-PU 的整体指标改善最强。L2-SP 和 TailMix 均削弱了 macro 与 tail 收益。

### 3.2 相对 Head-DB20

| 方法 | Δ macro-AUPR | Δ tail-AUPR | tail胜场 |
|---|---:|---:|---:|
| A6-PU | +0.009580 | +0.006738 | 6/9 |
| A6-PU-SP | +0.002524 | -0.000017 | 5/9 |
| A6-TailMix | -0.000854 | -0.001774 | 4/9 |

A6-PU 明确优于仅训练分类头的 Head-DB20，证明六类困难尾类确实需要表征更新，而不是继续增加分类头训练轮数。

### 3.3 相对端到端 Base-DB

| 方法 | Δ macro-AUPR | Δ tail-AUPR |
|---|---:|---:|
| A6-PU | -0.003980 | -0.040504 |
| A6-PU-SP | -0.011036 | -0.047259 |
| A6-TailMix | -0.014415 | -0.049016 |

A6-PU 的 macro-AUPR 已接近 Base-DB，但尾类 AUPR 仍有明显差距。这说明最后一个时序块能够修复整体判别，却没有提供足够深度的尾类表征重构。

## 4. 逐尾类结果

### 4.1 相对 Base-ASL 的平均变化

| 方法 | 尾类 | Δ macro-AUPR | Δ tail-AUPR | macro胜场 | tail胜场 |
|---|---|---:|---:|---:|---:|
| A6-PU | SawFM | +0.014976 | -0.028627 | 3/3 | 0/3 |
| A6-PU | VFT | +0.025892 | +0.020577 | 3/3 | 3/3 |
| A6-PU | ISRJ | +0.008695 | +0.007676 | 3/3 | 3/3 |
| A6-PU-SP | SawFM | +0.012303 | -0.017370 | 3/3 | 0/3 |
| A6-PU-SP | VFT | +0.015275 | +0.003765 | 3/3 | 2/3 |
| A6-PU-SP | ISRJ | +0.000817 | -0.007034 | 2/3 | 0/3 |
| A6-TailMix | SawFM | +0.008335 | -0.019155 | 3/3 | 0/3 |
| A6-TailMix | VFT | +0.010103 | +0.003647 | 3/3 | 2/3 |
| A6-TailMix | ISRJ | -0.000177 | -0.010402 | 2/3 | 0/3 |

### 4.2 目标尾类 AUPR 实际均值

| 尾类 | Base-ASL | Head-DB20 | A6-PU | A6-PU-SP | A6-TailMix | Base-DB |
|---|---:|---:|---:|---:|---:|---:|
| SawFM | 0.712712 | 0.700839 | 0.684085 | 0.695342 | 0.693557 | 0.736400 |
| VFT | 0.621006 | 0.624099 | 0.641583 | 0.624771 | 0.624653 | 0.668724 |
| ISRJ | 0.661321 | 0.649514 | 0.668997 | 0.654287 | 0.650919 | 0.711053 |

## 5. 三个机制假设的判定

### 5.1 局部解冻：部分成立

A6-PU 在 VFT 和 ISRJ 上均为 3/3 seeds 正向，并且宏指标 9/9 获胜。因此“冻结分类头不足，需要局部表征更新”得到支持。

但 SawFM 的结果方向相反：macro-AUPR 提高，tail-AUPR 却稳定下降。这说明最后时序块学到了有利于其他标签和总体平均的变化，但牺牲了 SawFM 的排序能力。

### 5.2 L2-SP：当前全局权重不适用

L2-SP 将 SawFM 的损失从 `-0.028627` 缩小到 `-0.017370`，说明它确实抑制了错误方向的漂移；但同时把 VFT 收益从 `+0.020577` 压到 `+0.003765`，并将 ISRJ 从 `+0.007676` 变为 `-0.007034`。

因此单一全局保持强度不能同时满足三个类别：SawFM 需要更强保持或不同适配方向，而 VFT、ISRJ 需要更充分的表征变化。

### 5.3 TailMix：不成立

TailMix 没有把宏指标收益转化为尾类收益，反而总体 tail-AUPR 降至 `-0.008637`。

原因可能包括：

1. DB 的样本权重由一个样本中所有共现正标签共同决定，只保留目标列会丢失共现标签对共享特征的有用梯度；
2. 尾类输出上的 DB 梯度不能单独决定共享时序特征的更新方向；
3. 检查点仍按 macro-AUPR 选择，目标尾类与宏平均冲突时会优先保存宏平均更高的模型。

## 6. SawFM 为什么成为唯一阻断点

SawFM 的 Base-DB 比 Base-ASL 高约 `0.023688`，说明 DB 信号不是无效；但只更新最后时序块后反而低 `0.028627`。

这表明 SawFM 所需变化不位于最后一个高层时序块，或者需要多个时序尺度协同改变。当前局部适配只改变高层局部模式，可能改善 VFT/ISRJ 和其他标签，却破坏 SawFM 所需的较低层调制结构。

此外，A6-PU 的 SawFM 最优 macro-AUPR 检查点位于第 16–20 轮，不是简单的训练不足。下一步必须同时记录并用于选择目标尾类 AUPR，才能区分“训练轨迹中曾经有效但被宏指标选择丢弃”和“整个适配方向始终错误”。

## 7. 下一阶段建议

下一阶段不再继续调整 L2-SP 或 TailMix 权重，优先处理两个结构性问题。

### 7.1 先增加逐epoch尾类选择证据

- 在 early-stop role 每个 epoch 记录逐标签 AUPR；
- 同时保存 macro-AUPR 最优、目标 tail-AUPR 最优和二者 Pareto/调和分数最优检查点；
- 先复用 SawFM 的 A6-PU 设置做 3 seeds，不改变训练目标；
- 若 tail 最优检查点可转正，问题主要是模型选择；否则确认是表征深度不足。

### 7.2 若表征深度不足，进行分层解冻

- A6-D2：解冻最后两个时序卷积块与分类头；
- 使用分层学习率，例如分类头 `1e-3`、最后块 `1e-4`、倒数第二块 `3e-5`；
- 使用完整 DB，不使用本轮失败的 TailMix；
- 第一轮只做 SawFM × 3 seeds，并以 Base-ASL、A6-PU、Base-DB 为对照；
- 只有 SawFM tail-AUPR 转正且 macro-AUPR 非劣，才回到 VFT/ISRJ 和六尾类扩展。

## 8. 审计与产物

- 技术方案：`docs/implementation/route_a_six_class_partial_adaptation_plan_20260723.md`
- runner：`experiments/run_route_a_six_class_partial_adaptation.py`
- 正式汇总：`outputs/tapc/route_a_six_class_partial_adaptation/route_a_six_class_partial_adaptation_summary.json`
- 27 个正式运行：`outputs/tapc/route_a_six_class_partial_adaptation/runs/`
- 两轮冒烟：`outputs/tapc/route_a_six_class_partial_adaptation_smoke/` 与 `outputs/tapc/route_a_six_class_partial_adaptation_smoke_v2/`

审计结果：27 份 metrics、receipt、manifest 均齐全；样本数均为 2400，标签计数误差为 0；所有运行均为 12806 个可训练参数；状态变化仅发生在最后时序块与分类头；locked test 使用次数为 0。

正式汇总 SHA256：`3387c9fd12c2e7e98904d770eb45213db40dc0dfb5f915a5edad7f7847633fb0`。全量测试结果为 `121 passed`。
