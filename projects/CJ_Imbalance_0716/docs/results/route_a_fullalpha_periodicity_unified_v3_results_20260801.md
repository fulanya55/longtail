# Route A FullAlpha1p5 + ISRJ Periodicity 统一组合结果

## 判定

development-v3 组合门禁 13/13 全部通过。`U2_FullAlphaAlignedPeriodicity` 是目前第一个同时满足原 Goal 统一六标签宏观、三目标弱召回、weak-AUROC、FPR、强正保护和 SawFM 安全门槛的候选。

相对 `U0_C6RawSaw`：

- macro-AUPR `+0.009264`，95% CI `[+0.007816,+0.010715]`；
- macro-F1 `+0.019100`，CI `[+0.015665,+0.022697]`；
- MFT/VFT/ISRJ mean weak recall `+0.061275`，CI `[+0.050094,+0.075642]`；
- 三目标 mean weak-vs-negative AUROC `+0.020786`，CI `[+0.012443,+0.028446]`；
- mean strong recall `+0.048653`；
- target mean FPR 绝对值 `0.190806`，增量 `+0.002527`；
- target weak recall 为 5/5 种子正增益。

## 绝对结果

| 变体 | macro-AUPR | macro-F1 | target weak recall | target weak AUROC | target strong recall | target FPR |
|---|---:|---:|---:|---:|---:|---:|
| U0 C6+RawSaw | 0.877927 | 0.765668 | 0.299151 | 0.589002 | 0.797396 | 0.188280 |
| U1 FullAlpha+RawSaw | 0.883127 | 0.776423 | 0.332016 | 0.602022 | 0.823682 | 0.191613 |
| **U2 FullAlpha+aligned periodicity** | **0.887191** | **0.784768** | **0.360425** | **0.609788** | **0.846049** | 0.190806 |
| U3 FullAlpha+block-shuffled | 0.882003 | 0.775465 | 0.335523 | 0.605072 | 0.818935 | 0.191129 |

## 标签级机制分解

U2 相对 U0：

| 标签 | AUPR | 弱召回 | weak AUROC | FPR |
|---|---:|---:|---:|---:|
| MFT | +0.008033 | +0.050298 | +0.027990 | +0.002581 |
| VFT | +0.011041 | +0.035044 | +0.011478 | +0.005806 |
| ISRJ | +0.030752 | +0.098481 | +0.022890 | -0.000806 |
| SawFM | 0 | 0 | 0 | 0 |

MFT/VFT 的增益完全来自 FullAlpha1p5；ISRJ 的额外增益来自周期专家。U2 相对 U1 的 ISRJ AUPR、弱召回、weak AUROC分别为 `+0.024383/+0.085228/+0.023298`，同时 ISRJ FPR平均下降 `-0.002419`。这表明周期模块没有简单用放宽阈值换召回。

## 机制控制

U2 相对 U3 block-shuffle：

- ISRJ AUPR `+0.031128`；
- ISRJ 弱召回 `+0.074707`；
- ISRJ weak AUROC `+0.014148`，CI `[+0.008358,+0.019217]`；
- macro-AUPR/F1 `+0.005188/+0.009303`。

非 ISRJ logits 最大绝对变化为 0；SawFM 的 AUPR、弱召回和分数完全不变。组合收益因此可以分解为：FullAlpha 的共享任务向量负责 MFT/VFT，aligned periodicity 只负责 ISRJ，RawSaw 只负责 SawFM。

## 证据边界与下一步

本轮仍是 development-v3 组合证据。由于两条组件结果此前均已被查看，不能直接进入 test-v3。按预注册，现将 U0/U1/U2/U3 的 checkpoint、融合头、Platt 和阈值冻结；生成全新 receiver identity 的 development-v7，禁止任何重新训练、校准、阈值选择或候选筛选，只做一次确认。

## 产物

- 预注册：`docs/plans/route_a_fullalpha_periodicity_unified_v3_plan_20260801.md`。
- 实现：`experiments/run_route_a_fullalpha_periodicity_unified_v3.py`。
- 测试：`tests/test_route_a_fullalpha_periodicity_unified_v3.py`，1 passed。
- 输出：`outputs/tapc/route_a_fullalpha_periodicity_unified_v3`。
- 汇总 SHA256：`A5D6825A7F84A01B6F1C68DC47AF400037DB4420A41ECB920D8D3D3E33754328`。
- test-v3 未访问、未分析、未使用。

