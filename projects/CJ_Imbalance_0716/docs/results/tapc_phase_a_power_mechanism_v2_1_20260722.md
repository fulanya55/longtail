# TAPC-CJNet Phase-A 功率机制筛选实验结果与下一步计划

版本：V2.1  
日期：2026-07-22  
运行环境：`D:\anaconda3\envs\PY10\python.exe`  
数据来源：`C:\LJQ_code\干扰数据\复合干扰数据0716`  
有效结果目录：`outputs/tapc/phase_a_screen_v2_1_macro_aupr/seed_2026`  
locked test：未读取

## 1. 结论

本轮应选择 **T0P（共享分类器 + 同身份配对功率视图 + CB-ASL）作为下一步主基线**。

- T0P 相对 T0 的 macro-AUPR 提高 0.07634，固定 0.5 阈值 macro-F1 提高 0.05983，弱成分召回率提高 0.11628，说明“同身份多功率视图”本身是有效的数据增强。
- T2 相对 T0P 的方向 margin 均值提高 0.10353、方向满足率提高 0.14247，说明功率方向约束确实被模型学到。
- 但是 T2 的 macro-AUPR 下降 0.01744，超过预设允许下降 0.01 的门槛；固定阈值 macro-F1 下降 0.01980，弱成分召回率下降 0.04651。因此 T2 **物理机制有效，但识别性能门槛失败**，不能原样进入 Phase-B。
- T2S 与 T2 完全一致，stability loss 始终为 0。当前容差 0.25 明显大于实际 unchanged drift（p90 约 0.07），稳定性约束未被激活，不能宣称其有效。

## 2. 实验协议

| 项目 | 设置 |
|---|---|
| protocol | Power-LT |
| seed | 2026 |
| optimization 样本 | 512 |
| early-stop 评估样本 | 128 |
| epochs / patience | 8 / 8 |
| batch size | 64 |
| 功率等级 | 25、30、35 dB |
| reference level | 30 dB |
| loss | CB-ASL |
| conditioning | independent per-label projection |
| checkpoint selection | macro-AUPR |
| AMP | 开启 |

首次 V2.0 执行仍按 subset accuracy 选 checkpoint，与冻结协议不一致。该结果保留在 `outputs/tapc/phase_a_screen_v2/seed_2026` 作为失败审计，不用于机制选择；本报告只使用 V2.1 重跑结果。

## 3. 主结果

下表均为开发集诊断结果，固定阈值指标使用温度缩放后的 0.5 阈值。

| 变体 | best epoch | macro-AUPR | macro-F1@0.5 | weak recall@0.5 | subset acc@0.5 | Hamming | cardinality MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 | 8 | 0.85995 | 0.74067 | 0.51163 (22/43) | 0.13281 | 0.33008 | 1.00781 |
| **T0P** | **8** | **0.93628** | **0.80049** | **0.62791 (27/43)** | **0.18750** | **0.25977** | **0.86719** |
| T2 | 8 | 0.91885 | 0.78069 | 0.58140 (25/43) | 0.17969 | 0.28125 | 0.90625 |
| T2S | 8 | 0.91885 | 0.78069 | 0.58140 (25/43) | 0.17969 | 0.28125 | 0.90625 |

所有变体的 tail-label recall 均为 1.0（67 个正例），因此该指标在本轮没有区分度。T0P 的主要优势来自更高的排序质量、整体 F1 和弱成分召回，以及更低的 Hamming loss 与标签基数误差。

## 4. 物理方向诊断

| 变体 | margin mean | margin p10 | 满足 margin=0.20 的比例 | unchanged drift p90 | ordering loss | distillation loss | stability loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0P | 0.23660 | -0.13135 | 0.54984 | 0.07192 | 0 | 0 | 0 |
| T2 | 0.34013 | -0.02893 | 0.69231 | 0.06978 | 0.06267 | 0.10663 | 0 |
| T2S | 0.34013 | -0.02893 | 0.69231 | 0.06978 | 0.06267 | 0.10663 | 0 |

T2 的方向约束不是无效项：margin 分布和满足率均明显改善，unchanged drift 也没有恶化。问题在于当前排序损失会同时对强、弱功率 logits 反向传播。满足

\[
s_{strong,c}-s_{weak,c} \ge m
\]

既可以通过提高强功率 logit，也可以通过压低弱功率 logit 实现。后者与弱功率视图的分类目标冲突，是弱成分召回从 27/43 降到 25/43 的一个合理原因。

## 5. 逐标签证据

| 标签 | T0P AUPR | T2 AUPR | T0P recall | T2 recall | 判断 |
|---|---:|---:|---:|---:|---|
| FMB | 0.96933 | 0.95882 | 0.94030 | 0.92537 | 下降 |
| SinFM | 0.95162 | 0.93791 | 0.88000 | 0.82667 | 明显下降 |
| MFT | 0.93336 | 0.91441 | 0.90625 | 0.89063 | 下降 |
| ISRJ | 0.89082 | 0.86424 | 1.00000 | 1.00000 | AUPR 下降 |

T2 的 macro-AUPR 下降不是单一标签偶然波动：四个标签的 AUPR 均下降，其中 SinFM 和 MFT 对当前约束最敏感。

## 6. 下一步网络结构：T2v2 单侧方向约束

下一轮继续使用 T0P 的共享 backbone、共享分类器和双视图分类作为主体，只修改方向损失，不恢复 prototype、参数头或决策接口。

```text
weak/reference I/Q ─┐
                    ├─ shared complex CNN ─ shared classifier ─ weak logits
strong/paired I/Q ──┘                                  └────── strong logits

weak logits ── stop-gradient ────────────────┐
strong logits ───────────────────────────────┴─ one-sided ordering loss

shared feature ─ per-label independent projection ─ low-to-high embedding distillation
```

单侧排序损失定义为：

\[
L_{ord}^{one-sided}
= \frac{1}{|M|}\sum_{(i,c)\in M}
\max\left(0, m-\left[s_{strong,i,c}-\operatorname{sg}(s_{weak,i,c})\right]\right)
\]

其中 `sg` 表示 stop-gradient。该结构允许排序损失提高强功率响应，但禁止它通过压低弱功率响应来满足 margin。两个视图仍各自使用 CB-ASL，确保部署使用的弱功率分类器持续得到直接监督。

## 7. 下一轮实验矩阵

先做机制拆分，再做权重微调，避免同时改变多个因素。

| 变体 | 配对视图分类 | 单侧 ordering | embedding distillation | 用途 |
|---|---:|---:|---:|---|
| T0P | 是 | 否 | 否 | 已确定主基线 |
| T2O-v2 | 是 | 是 | 否 | 单独验证单侧排序 |
| T2D | 是 | 否 | 是 | 单独验证蒸馏 |
| T2OD-v2 | 是 | 是 | 是 | 判断两者是否互补 |

第一轮保持与 V2.1 相同的数据、seed、epoch 和 checkpoint 选择规则。仅当 T2O-v2 或 T2OD-v2 同时满足以下门槛，才进入多 seed 或 Phase-B：

1. 相对 T0P，macro-AUPR 下降不超过 0.01；
2. weak recall 不低于 T0P；
3. directional margin mean 和 satisfaction rate 均高于 T0P；
4. 四个标签中不得出现两个及以上标签 AUPR 下降超过 0.01。

若单侧 ordering 通过上述门槛，再在 `0.05 / 0.10 / 0.20` 三个权重上做小规模筛选。stability tolerance 暂不直接调整；应先确认单侧 ordering 的基本收益，再根据新的 drift 分布将 tolerance 设为开发集 p75 或固定小值进行独立消融。

## 8. 实验边界与结果复现

- 本轮未读取 locked test；所有结论均来自开发角色数据，只用于机制筛选。
- 当前证据来自单 seed、小样本和 8 epochs，不用于报告最终泛化性能。
- 原始汇总：`outputs/tapc/phase_a_screen_v2_1_macro_aupr/seed_2026/phase_a_summary.json`
- 表格汇总：`outputs/tapc/phase_a_screen_v2_1_macro_aupr/seed_2026/phase_a_summary.csv`
- 每个变体目录保留配置哈希、环境、数据清单哈希、训练历史、best epoch 诊断、固定阈值和选择阈值指标、预测以及成功 receipt。
