# MFT/VFT Oracle-Positive 参数可辨识性审计与专家试验结果

## 最终结论

数据索引和参数映射审计全部通过，但五个关闭分类损失的专项专家均未达到“MAE 相对常数基线降低至少 10%、相关系数至少 0.30、3/3 seeds 同方向”的回接门槛。因此本轮不把任何新支路接回 C2，也不使用 locked test。

MFT 自相关专家获得了真实但不充分的信号：总体 MAE 降低 6.39%、相关系数 0.250，两个方向均为 3/3 seeds 正向。在单成分、高 JNR 和部分低复合阶数组中已经超过门槛，说明 `num_targets` 并非完全不可辨识，主要瓶颈是复合成分掩蔽和弱 MFT 可见性。

VFT 的 raw、STFT 和瞬时频率专家均接近常数预测。生成配置只有一个 100 us 有效脉冲，VFT Doppler 为 1--7 kHz，而有效脉冲的傅里叶分辨率约 10 kHz；当前单 PRI 复合波形无法由通用时频支路稳定恢复该小频移。

## P0 参数与样本对齐审计

审计对象为 root train 的 25,200 个样本，未读取 locked test。

| 检查 | 结果 |
|---|---|
| `known_label == label_selected` | 全部一致 |
| `label_all` 按注册表投影到六类 | 全部一致 |
| 64 个锚点的 `data_legacy` 与 `iq_complex` | 最大差值 0 |
| 六类 12 个最小参数的 label-mask | 错位 0，有效非有限值 0 |
| 10 个 component metadata cell 长度与活跃成分数 | 25,200 个样本全部一致 |
| MFT component nominal JNR 与参数第 28 列（0 基 27） | 12,800/12,800 一致，最大误差 0 |
| VFT component nominal JNR 与参数第 49 列（0 基 48） | 12,800/12,800 一致，最大误差 0 |

MFT `num_targets` 确认为 0 基第 28 列，取值 3--8，共六个离散值；VFT `Doppler` 确认为 0 基第 49 列，范围 1000.77--6999.32 Hz，共 12,800 个连续值。

生成器也与审计一致：

- MFT 对参考脉冲做多个不同延迟的循环移位并按随机幅度求和；生成器 SHA256 `FA68EBEDF2FFE34ACFE0EB7341BEC7C3E4B6B798E018D74B51DFBF07076515CB`。
- VFT 使用 `reference.train .* exp(1i*2*pi*fd*t)` 后施加到达时延；生成器 SHA256 `4BA5A666F1D0BC808D7B0A0A6C22D68CB0B18D5E9F96F47065A6DEF30C4CC787`。

## 专家协议

- 分类损失：关闭。
- loader：只包含 `目标标签=1 AND 参数mask=1`。
- seeds：2041、2042、2043。
- 每个目标每 seed：训练阳性 2048、验证阳性 256。
- 同目标不同表征复用完全相同的样本哈希。
- MFT 按六个计数等级轮询抽样；VFT 按目标十分位轮询抽样。
- 常数基线：对应 seed 训练阳性的目标中位数。
- 模型选择：验证参数 MAE。

## 三种子结果

| 专家 | MAE | 常数 MAE | 相对改善 | 相关系数 | MAE正向 | 相关正向 | 回接C2 |
|---|---:|---:|---:|---:|---:|---:|---|
| MFT_RAW_ORD | 1.4676 | 1.4922 | 1.65% | 0.135 | 3/3 | 3/3 | 否 |
| MFT_ACF_ORD | 1.3968 | 1.4922 | 6.39% | 0.250 | 3/3 | 3/3 | 否 |
| VFT_RAW | 1456.1 Hz | 1478.9 Hz | 1.54% | 0.107 | 2/3 | 2/3 | 否 |
| VFT_STFT | 1479.2 Hz | 1478.9 Hz | -0.02% | 0.024 | 1/3 | 3/3 | 否 |
| VFT_IFREQ | 1482.6 Hz | 1478.9 Hz | -0.26% | -0.004 | 1/3 | 1/3 | 否 |

MFT 自相关专家逐 seed 的 MAE 改善为 3.76%、8.36%、7.05%，相关系数为 0.153、0.334、0.261；只有 seed 2042 的相关系数超过 0.30，没有 seed 同时达到总体 10% MAE 门槛。

## 分层诊断

### MFT

| 条件 | 样本数 | MAE改善 | 相关系数 |
|---|---:|---:|---:|
| 单成分 | 25 | 30.65% | 0.624 |
| 二成分 | 113 | 10.21% | 0.388 |
| 三成分 | 248 | 5.89% | 0.221 |
| 四成分 | 228 | -1.08% | 0.044 |
| JNR 10--15 dB | 202 | 2.03% | 0.128 |
| JNR 15--20 dB | 212 | 1.28% | 0.195 |
| JNR 20--25 dB | 178 | 5.52% | 0.193 |
| JNR 25--30 dB | 176 | 18.88% | 0.534 |

该结果支持“延迟峰确实携带目标数量，但在弱 MFT 和复合成分叠加时被淹没”。总体失败不是 ordinal 损失本身，而是自相关中目标峰与其他转发/调制成分的峰难以分离。

### VFT

- `fs=10 MHz`，有效脉冲宽度 `100 us`，有效脉冲 1000 点；单 PRI 为 `300 us/3000点`，且 `num_pulses=1`。
- 有效脉冲傅里叶分辨率约 10 kHz，完整 PRI 的名义分辨率约 3.33 kHz，而待估 Doppler 总跨度只有 6 kHz。
- 通用 STFT 和原始瞬时频率同时包含 LFM 本身、未知到达时延及其他复合成分，目标的 1--7 kHz 常量偏移相对过小。
- VFT_RAW 在不同 JNR 分层中的最大 MAE 改善也只有 2.87%，没有出现高 JNR 后明显可辨识的趋势。

因此当前结果只能说明“通用 raw/STFT/IFREQ 专家失败”，不能证明参考辅助估计在物理上绝对不可能。

## 后续方向

1. **MFT**：保留自相关路线，但加入参考脉冲匹配滤波、峰显著性/峰间距监督和对共现成分的条件化；优先在 JNR 25--30 dB 与一至二成分组建立正向 sanity check，再逐步加入遮蔽成分。
2. **VFT**：停止普通 STFT 超参数搜索，改做参考辅助对齐与 dechirp；先估计到达时延，再用已知 LFM 参考去调频并回归残余相位斜率。
3. **数据边界**：若 dechirp 专家仍不能通过门槛，应增加多脉冲观测或扩大 Doppler 范围，而不是继续提高联合损失权重。
4. 上述新专家仍需独立三 seeds 通过相同门槛，之后才能重新接回 C2。

## 产物

- 对齐审计：`outputs/student_b/oracle_parameter_specialists_20260723/alignment_audit.json`。
- 三种子汇总：`outputs/student_b/oracle_parameter_specialists_20260723/oracle_specialist_summary.json`。
- 分层诊断：`outputs/student_b/oracle_parameter_specialists_20260723/stratified_diagnostics.json`。
- 实验方案：`docs/implementation/oracle_parameter_identifiability_plan_20260723.md`。
- 模型与训练：`cjlt/models/oracle_parameter_specialists.py`、`cjlt/engine/oracle_parameter_trainer.py`。
