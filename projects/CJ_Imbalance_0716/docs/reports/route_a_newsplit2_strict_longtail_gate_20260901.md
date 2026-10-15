# Route-A 第二独立划分长尾候选晋级门

## 协议

使用与第一新划分完全相同的协议，仅将数据划分种子改为 `260902`：30 dB 六类开发数据，训练 1000 样本、标签 IF=2，验证集 15 个训练中未出现的组合（367 个样本），训练/验证组合交集为空。五个模型均为冻结的 F2 backbone/head（257,511 参数），8 epochs，唯一变量为损失函数；未读取 locked-test，E5 未修改。

## Positive-Focal 五种子确认

| 指标 | BCE | Positive-Focal | 差值 | Wilcoxon p |
|---|---:|---:|---:|---:|
| Macro-AUPR | 0.975027 | 0.979420 | +0.004393 | 0.4652 |
| Macro-F1 | 0.840770 | 0.864958 | +0.024188 | 0.0679 |
| 组合准确率 | 0.558038 | 0.594005 | +0.035967 | 0.1088 |
| Hamming loss | 0.121072 | 0.104905 | −0.016167 | 0.0679 |

方向与第一新划分一致，但效应较小且未达显著性。标签级平均变化：MFT recall `0.7558→0.8322`、ISRJ recall `0.5216→0.6284`；ISRJ 负类 FPR `0.0471→0.0555`（+0.84 个百分点），SinFM FPR `0→0.23` 个百分点。故 Positive-Focal 在“提升尾类召回”和“误报安全门”之间存在划分依赖，当前只能作为候选，不能冻结为 final。

## PF_HN 有界困难负样本筛查（3 seeds）

PF_HN 为 PFmix25 加入训练集头部困难负样本 hinge 约束。相对 BCE 的均值：Macro-AUPR `0.977686→0.979071`，Macro-F1 `0.870029→0.865744`，组合准确率 `0.625795→0.627611`，Hamming `0.101120→0.101877`。组合仅提升 0.18 个百分点，Macro-F1 与 Hamming 变差；ISRJ recall 由 `0.6023` 降至 `0.5739`。该分支停止，不进入确认或 final。

## 判定

第二独立划分未满足“组合准确率、Macro-F1、Hamming 同时改善且任一标签 FPR 增幅≤0.5 个百分点”的晋级门。Positive-Focal 保留为跨划分的研究候选，PF_HN 判定失败。证据说明：普通 BCE 的瓶颈确实主要出现在未见组合和尾类召回，但单纯改变损失函数的收益受组合划分影响，尚不足以形成可靠创新点。

## 下一步

停止继续堆叠 focal/hinge 权重；转向组合结构或可见性约束（例如组合感知原型/条件残差路由），并在新的 development-v2 身份划分上预注册。任何新方法须先通过相同 BCE 对照和五种子安全门，再考虑 final 或 locked-test。

收据：

- `outputs/route_a_newsplit2_strict_combo_labelIF2_20260901/manifest_summary.json`
- `outputs/route_a_newsplit2_strict_combo_methods_screen_20260901/`
- `outputs/route_a_newsplit2_strict_combo_methods_confirm_20260901/`
- `outputs/route_a_newsplit2_strict_combo_pfhn_screen_20260901/`
