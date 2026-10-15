# EfficientNet-B1 多专家与长尾损失筛选结果（2026-07-25）

## 1. 结论

本轮在上一轮 VFT-D11 困难场景上完成了 4 个真实数据冒烟运行和 12 个正式运行，比较：

- 单专家 EfficientNet-B1 + BCE；
- 单专家 EfficientNet-B1 + ASL；
- 三专家 EfficientNet-B1 + BCE；
- 三专家 EfficientNet-B1 + ASL。

所有正式运行均为 30 epochs、3 seeds，并对 development 身份进行九档功率评估。锁定测试集未使用。

结论如下：

1. **默认 ASL 不适合当前 VFT-D11。** 它把弱召回推到约 0.992，但 VFT 负类 FPR 同时升到约 0.982，VFT-AUPR 没有提高，说明它主要放宽了正类响应，没有改善弱正样本与困难负样本的排序。
2. **轻量三专家结构有有限价值，但不能直接晋级。** 相对单专家 BCE，三专家 BCE 的 VFT-AUPR 平均提高 0.0066，Hamming 改善 0.0052，VFT FPR 降低 0.0605；但弱召回下降 0.0897，没有达到预设门槛。
3. **当前专家没有形成真正专门化。** 三专家 BCE 的专家间总体概率相关系数为 0.993，VFT 概率相关系数为 0.974，专家间 VFT 概率标准差仅约 0.019。三个分支基本在学习同一决策面。
4. **下一步不应继续增加全局类别权重。** 应把多专家改为显式任务分工，并针对 VFT 的高混淆负组合加入困难负样本排序或对比约束。

本轮四种方法均未通过完整晋级门槛，因此当前正式基线仍保留 `efficientnet_b1_1d + BCE`。

## 2. 实验协议

- 目标：VFT；
- 单元：D11，VFT 标签 IF=3，同时使用 10/15/20 dB 弱可见训练视图；
- 每次训练身份数：2400；
- 身份角色、长尾 manifest 和九功率绑定规则与上一轮完全一致；
- seeds：20260724、20260725、20260726；
- 优化器：AdamW，学习率 `3e-4`，weight decay `1e-4`；
- batch size：64；
- epoch：30；
- early-stop、calibration、development 角色保持分离；
- checkpoint 仍按 early-stop macro-AUPR、VFT-AUPR、训练损失排序；
- 评估：九功率 macro/VFT-AUPR、F1、Hamming、弱召回、VFT FPR、逐标签和 63 组合；
- `locked_test_used=false`。

## 3. 多专家结构

三专家模型使用 EfficientNet-B1 的一维 I/Q 接口：

```text
I/Q [B,2,3000]
  -> 共享 Stem + EfficientNet stages 1-6
  -> 三个独立 stage 7 + 1280-channel top
  -> 三个独立 1280 -> 256 -> 6 分类头
  -> 三个专家 logits 等权平均
```

训练损失为集成输出损失，加 `0.3 ×` 三个专家独立损失均值。参数量从 6,643,398 增加到 12,652,498，为单专家的约 1.90 倍。

## 4. 三种子主结果

| 方法 | 参数量 | Macro-AUPR | VFT-AUPR | Macro-F1 | Hamming | 弱召回 | VFT负类FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 + BCE | 6.64M | 0.8095 | 0.5929 | 0.8114 | 0.2317 | 0.7730 | 0.4559 |
| B1 + ASL | 6.64M | 0.8054 | 0.5914 | 0.8392 | 0.2375 | 0.9916 | 0.9820 |
| MoE3 + BCE | 12.65M | 0.8090 | **0.5995** | 0.7994 | **0.2265** | 0.6833 | **0.3954** |
| MoE3 + ASL | 12.65M | 0.8067 | 0.5899 | 0.8369 | 0.2426 | **0.9922** | 0.9833 |

## 5. 相对 BCE 基线的配对变化

| 候选 | ΔMacro-AUPR | ΔVFT-AUPR | ΔHamming | Δ弱召回 | ΔVFT FPR | 晋级 |
|---|---:|---:|---:|---:|---:|---|
| B1 + ASL | -0.0041 | -0.0014 | +0.0057 | +0.2186 | +0.5261 | 否 |
| MoE3 + BCE | -0.0005 | +0.0066 | -0.0052 | -0.0897 | -0.0605 | 否 |
| MoE3 + ASL | -0.0028 | -0.0030 | +0.0108 | +0.2191 | +0.5274 | 否 |

ASL 的高 F1 和高弱召回不能解释为识别能力提高，因为 AUPR 没有改善且负类 FPR 接近 1。三专家 BCE 确实减少了一部分误报，但收益不足且牺牲弱正召回。

## 6. 专家是否形成差异

三专家 BCE 的三种子平均诊断：

- 全部标签概率的专家间相关系数：0.9930；
- VFT 概率的专家间相关系数：0.9738；
- VFT 概率在专家间的平均标准差：0.0191；
- 三个独立专家 VFT-AUPR：0.5990、0.5994、0.5964；
- 集成 VFT-AUPR：0.5991。

集成没有获得明显的互补增益。失败原因不是专家数量不足，而是三个专家使用相同数据、相同任务和相同损失，导致分支收敛到高度相似的表示。

## 7. VFT 困难负组合

单专家 BCE 在 10 dB 下最容易把以下不含 VFT 的组合误报为 VFT：

| 组合 | BCE FPR | MoE3-BCE FPR |
|---|---:|---:|
| SinFM+MFT+ISRJ | 0.975 | 0.908 |
| SinFM+MFT | 0.958 | 0.817 |
| MFT+ISRJ | 0.925 | 0.642 |
| SinFM+ISRJ | 0.917 | 0.825 |
| SinFM | 0.917 | 0.658 |
| MFT | 0.800 | 0.692 |

三专家结构对部分组合降低了误报，尤其是 FMB+MFT+ISRJ、FMB+MFT、FMB+ISRJ 和 MFT+ISRJ；但 SinFM/MFT/ISRJ 家族仍是主要瓶颈。同时，三专家对部分包含 SawFM 的负组合出现了新的误报上升，说明通用等权集成没有建立稳定的困难负样本边界。

## 8. 下一步改进方向

下一步采用“共享骨干 + 显式专门化残差专家”，不再继续测试默认全局重加权损失：

1. **通用专家**：保持六标签 BCE，维持整体分类能力；
2. **VFT 弱正专家**：重点学习 10/15/20 dB 的 VFT 正样本及其 30 dB 配对视图；
3. **VFT 困难负专家**：集中学习不含 VFT 的 SinFM/MFT/ISRJ 高混淆组合，以及本轮 MoE 新增误报的 SawFM 组合；
4. **VFT 残差融合**：通用专家负责六标签主 logits，两个专门专家只修正 VFT logit，避免其他五类被专家路由扰动；
5. **专门化约束**：加入专家差异约束，并使用 VFT 正样本与困难负组合之间的监督排序或对比损失，直接优化表征可分性，而不是只调决策偏置。

建议下一轮先做三个受控单元：

- E0：当前 B1+BCE 基线；
- E1：显式专门化 MoE，仍只使用 BCE；
- E2：E1 + VFT 困难负排序/对比损失。

晋级门槛继续使用三种子配对判断：VFT-AUPR 平均至少 `+0.02`，至少 2/3 seeds 为正；Macro-AUPR 不低于 `-0.005`；Hamming 不恶化超过 `+0.005`；弱召回不低于基线 `-0.05`；VFT FPR 不高于基线。通过后再使用两个全新确认 seeds，仍不开启锁定测试集。

## 9. 结果文件

- `outputs/vft_d11_backbone_loss_20260725/improvement_runs.csv`：12 次正式运行；
- `outputs/vft_d11_backbone_loss_20260725/improvement_aggregates.csv`：方法均值与标准差；
- `outputs/vft_d11_backbone_loss_20260725/improvement_deltas.csv`：相对基线的逐种子变化；
- `outputs/vft_d11_backbone_loss_20260725/decision.json`：晋级门槛判定；
- `outputs/vft_d11_backbone_loss_20260725/improvement_power_curves.csv`：九功率曲线；
- `outputs/vft_d11_backbone_loss_20260725/vft_negative_combo_fpr.csv`：VFT 困难负组合；
- `outputs/vft_d11_backbone_loss_20260725/expert_diversity.csv`：专家相关性与独立 AUPR。
