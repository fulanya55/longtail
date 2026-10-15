# P6b：参数特异时序查询池化停止报告

## 判定

P6b 未通过预注册晋级门，停止“最终 IQ-Grouped 时序 token 的类别特异 softmax 查询池化”路线；不生成确认身份，不调整 query 维度、温度、adapter、global-control、损失、阈值或 batch。

这不是计算失败、分类安全失败或系统性目标风险失败。关键失败是 C1 的局部时间定位相对等容量、等主干计算 C2 全局查询控制没有可证伪的增量价值。

## 结构与强对照

- **C0**：冻结 P4c 风格 MLCNN 分类 + IQ-Grouped 全局参数路由。
- **C1**：仅将参数池化替换为：最终 fused IQ-Grouped map → 256-D token adapter → 六类 learned query 的时间 softmax → 六个两参数头。
- **C2**：C1 的 token adapter、queries、六个参数头和完整 query-score map 计算全部保留；仅用全局 token 均值和 query 调制替换局部 softmax 池化。

C1/C2 均为 12,965,410 参数，严格相等；C0 为 12,890,786 参数。三者均使用相同的 MLCNN 分类路径、参数卷积 stem、训练预算、归一化与分类校准角色。

## 新身份和开发结果

- 生成 seed：`1170000000`；6,300 唯一 `receiver_noise_seed`；63 个组合，每组合 100 条；JNR 10–30 dB。
- 对根数据及 1,911 个历史 MAT 文件的全量审计重叠为零；12 个参数均有限、mask 对齐且有变化。
- 模型 seed：17041、17042、17043；20 epoch，AdamW，`lr=1e-4`、`wd=1e-4`、2 epoch warmup + cosine、batch=63、BCE + `0.1×masked SmoothL1`；checkpoint 只按 early 参数 macro-NMAE 选择。

| 模型 seed | C1 相对 C0 全 12降幅 | C1 相对 C2 全 12降幅 | C1-C0 Pearson | C1-C0 AUPR | C1-C0 Subset |
|---:|---:|---:|---:|---:|---:|
| 17041 | +2.824% | +0.199% | +0.02029 | 0.00000 | 0.00000 |
| 17042 | +3.972% | +0.364% | +0.03586 | +0.00008 | -0.00317 |
| 17043 | +2.153% | +0.249% | +0.00680 | -0.00012 | -0.00635 |

C1 对 C0 的每种子 2% 门通过，但均值为 **2.983%**，严格低于 3% 门。更具决定性的是，C1 对 C2 的每种子 1%、均值 1.5% 门均失败。因而不能把“时序 token adapter + 类特异参数头”的收益归因于局部位置选择。

## 端到端和逐目标诊断

predicted-gate 全参数相对 C0 的降幅为 +3.291%、+3.816%、+1.828%；miss penalty 三种子均降低。C1 对 C0 的 MFT.JNR、SinFM.JNR/频偏与 FMB 带宽均有稳定正方向，但 SawFM 周期与 ISRJ sample width 仍不稳定；VFT Doppler 三种子均恶化但未达到三种子均超过 +0.03 NMAE 的风险门。

这些诊断支持 token adapter 可改善部分参数，但没有证明局部查询是必要机制，也不能将 predicted-gate 局部改善替代 oracle full-12 与 C2 消融失败。

## 成本审计

| 指标 | C0 | C1 | C2 | 判定 |
|---|---:|---:|---:|---|
| batch=64 median | 4.210 ms | 4.479 ms | 4.575 ms | C1/C0=1.064×；C1/C2=0.979×，通过 |
| batch=1 P95 | 1.513 ms | 1.765 ms | 2.480 ms | C1/C0=1.166×，通过 |

成本没有解释失败，也不构成成功证据。

## 机制结论

P6b 排除了如下解释：

> 仅在最终 IQ-Grouped token 上加入类别特异 softmax 时间注意力，就能提供超过匹配全局调制的可靠全 12 参数收益。

后续不得在 P6b 身份池上继续搜索该查询头。若重启参数表征研究，必须选择不同的核心假设并在全新身份上预注册，例如先证明某个目标的**中间层尺度**相对最终 token 具有独立充分性；不能把 C2 这个必要消融在同一池上事后提升为候选。

## 工件

- [预注册](../configs/p6b_parameter_temporal_query_preregister_20260802.json)
- `outputs/p6b_parameter_temporal_query_20260802/data_audit.json`
- `outputs/p6b_parameter_temporal_query_20260802/result.json`
- `outputs/p6b_parameter_temporal_query_20260802/predicted_gate_confirm.json`
- `outputs/p6b_parameter_temporal_query_20260802/eager_latency.json`

全程为新身份 development，`test_generated=false`、`test_opened=false`、`locked_test_used=false`。
