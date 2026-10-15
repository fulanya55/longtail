# P1k–P1n：复合干扰识别与参数联合感知 Level-3 验证报告

日期：2026-08-01  
证据等级：全新 identity-safe development-confirmation；非 locked test。

## 1. 最终结论

P1n 在方法完全冻结后，使用 seed `1100000000` 生成的全新 20/25 dB 数据完成一次性测试，全部 9 项预注册门槛通过，`level3_validated=true`。当前可以支持的论文主张是：

> 在预测标签门控下，MLCNN 多标签主干、类别条件物理参数专家、ISRJ 任务路由和逐功率有限样本风险校准，能够在全新身份隔离的复合波形上同时改善识别、全 12 参数提取和最坏功率风险。

该结论达到 Level 3，但尚不是 Level 4/5：当前正式联合证据只覆盖 20/25 dB，尚未完成九档功率、参数分布外、弱分量和 locked test 验证。

## 2. 当前冻结系统

### 2.1 主干与参数头

- 分类与基础参数骨干：`mlcnn_iq_grouped_task_routed`。
- 输入：长度 3000 的双通道 I/Q。
- 分类输出：6 类复合干扰多标签 logits。
- 参数输出：6 类 × 2 参数，共 12 个连续参数。
- 参数分支使用类别条件任务路由；主干宽度 24，分类 head 维度 256，参数隐藏维度 64，dropout 0.2，SE attention。

### 2.2 物理专家

- SawFM period：冻结周期序列专家，仅替换 `[SawFM, period]` 参数。
- VFT Doppler：正确 LFM 参考、物理时延搜索、去斜与局部 delay-Doppler ambiguity 峰；无可训练参数，仅替换 `[VFT, Doppler]`。
- ISRJ：条件 U-Net 分离波形、周期特征与冻结统计专家形成 ISRJ 概率；15 个 bootstrap 路由器的 10% lower score 用于选择接受样本。
- MFT count：未通过专家准入门槛，明确不接入。

### 2.3 P1n 最终路由和校准

- ISRJ 接受覆盖率在 fresh selection 身份上选择，候选覆盖率为 5%–30%。
- 路由接受后采用 `alpha=1.0`，即直接使用 ISRJ 专家概率；P1k–P1m 的 `alpha=0.75` 被证实过于保守。
- ISRJ 路由后复用路由前在 fresh calibration 身份上通过逐功率风险约束的阈值，避免二次校准抬高阈值。
- ISRJ 与 VFT 每个功率均要求 calibration observed FPR ≤ 0.06、Clopper–Pearson 95% 上界 ≤ 0.08。
- 其他类别概率、阈值和 10 个未推广参数输出保持不变。

## 3. 基础训练方式与超参数

- 优化器配置：learning rate `1e-4`，weight decay `1e-4`。
- 学习率：2 epoch warmup + cosine schedule。
- epoch：20；early-stop patience：20。
- batch size：63；随机种子：2041、2042、2043。
- 分类损失：`BCEWithLogitsLoss`。
- 参数损失：只对 `label=1 AND parameter_mask=1 AND finite target` 的位置计算 masked Smooth-L1；inactive NaN 不填零。
- 总损失：`L = L_cls + 0.1 * L_param`。
- 参数 checkpoint 用参数目标选择；阈值、early stop、调参和 confirm 角色相互分离。
- 本轮不使用半监督、不使用长尾采样。

## 4. 四轮联合实验的递进

| 轮次 | 唯一改动 | ISRJ Recall 增益 | 3/3 正向 | 最坏 ISRJ FPR | 判定 |
|---|---|---:|---:|---:|---|
| P1k | lower route + 路由后再校准阈值 + alpha 0.75 | +0.04414 | 否 | 0.05887 | 失败 |
| P1l | 复用路由前安全阈值 | +0.04740 | 是 | 0.05161 | 均值门槛差 0.00260 |
| P1m | lower score 改 ensemble mean | +0.04701 | 是 | 0.05565 | mean 排序无增益 |
| P1n | 恢复 lower score，alpha 0.75→1.0 | **+0.08372** | **是** | **0.07097** | **9/9 通过** |

### 4.1 P1k 失败原因

P1k 的参数侧全部成功，但 seed 2042 的路由后 ISRJ 阈值由 0.850 抬到 0.911，使 25 dB Recall 下降 0.01875。失败来自“路由概率变化—二次阈值校准”的交互，而不是物理专家能力不足。

### 4.2 P1l/P1m 失败原因

共享安全阈值恢复了 3/3 正向和显著 bootstrap 区间，但 alpha 0.75 把专家概率改变量缩小 25%，两套排序规则均稳定停在约 +0.047，略低于 +0.05 的冻结门槛。ensemble mean 没有带来额外收益，因此停止该分支。

### 4.3 P1n 成功机制

P1n 保留原有接受集合与风险阈值，只让已接受样本完整采用专家概率。安全裕量足以吸收更强混合：Recall 增益升至 +0.08372，而最坏 FPR 仍仅为 0.07097。

## 5. P1n 定量结果

### 5.1 识别与风险

- ISRJ 平均 Recall 增益：`+0.08372`。
- 分 seed：`+0.06680 / +0.08320 / +0.10117`。
- paired bootstrap：`+0.08372`，95% CI `[+0.07407, +0.09244]`。
- 最坏 ISRJ 功率 FPR：`0.07097`。
- 最坏 VFT 功率 FPR：`0.04839`。
- macro-AUPR 相对原始 IQ-Grouped：`+0.00215`，95% CI `[+0.00170, +0.00260]`。

### 5.2 全 12 参数端到端指标

- predicted-gate NMAE：safe-route `0.72879` → joint `0.67814`，相对改善 `6.95%`。
- 分 seed 改善：`6.82% / 7.20% / 6.82%`。
- joint vs safe-route NMAE delta：`-0.05065`，95% CI `[-0.05322, -0.04806]`。
- joint vs original NMAE delta：`-0.05443`，95% CI `[-0.05717, -0.05179]`。
- 漏检惩罚相对 original delta：`-0.03274`，95% CI `[-0.03498, -0.03059]`。
- worst-combo predicted-gate NMAE 相对 safe-route 平均下降 `0.01016`。

## 6. 计算代价

RTX 5080、FP32、batch 1/64 的冻结流水线基准：

| 指标 | Main only | P1n joint | 代价 |
|---|---:|---:|---:|
| 参数量 | 12.891 M | 12.989 M | +0.76% |
| profiler 支持 FLOPs/样本 | 49.39 M | 252.01 M | 5.10×，为下界 |
| batch=1 延迟 | 1.577 ms | 6.595 ms | 4.18× |
| batch=64 延迟/样本 | 0.0659 ms | 0.1805 ms | 2.74× |
| batch=64 吞吐 | 15181/s | 5541/s | -63.5% |
| batch=64 峰值显存 | 174.7 MB | 335.4 MB | 1.92× |

FLOPs 由 PyTorch profiler 支持算子统计，FFT/STFT 与自定义物理特征算术可能未计入，因此联合 FLOPs 是明确的下界。参数开销很小，但延迟开销显著；后续论文应把“专家仅在预测阳性或高风险样本上惰性调用”作为系统优化方向，当前不能声称计算代价已被完全解决。

## 7. 证据边界与后续路线

已验证：

1. 至少两个关键参数专家达到 Level-2 准入并在联合系统中保持收益。
2. 预测标签门控下，全 12 参数 NMAE、漏检惩罚和最坏组合同时改善。
3. ISRJ Recall、macro-AUPR 和逐功率 FPR 同时通过冻结门槛。
4. 两个强基线与路由消融均有 identity-grouped paired-bootstrap 置信区间。

尚未验证：

1. 九档功率与弱分量泛化。
2. 参数分布外与新的复合结构。
3. MFT count 专家。
4. 低开销惰性专家调用。
5. locked test；本报告 `locked_test_used=false`。

推荐长线顺序：先冻结 P1n，不再修改方法；完成九档功率、弱分量和参数 OOD 的 fresh independent 验证；随后优化惰性调用并在新数据重新确认；所有规则再次冻结后，再向用户申请一次性 locked test。

## 8. 复现实物

- P1n 方法冻结：`outputs/p1n_full_blend_joint_20260801/method_freeze.json`
- P1n 汇总：`outputs/p1n_full_blend_joint_20260801/summary.json`
- P1n 逐 seed/逐样本：`outputs/p1n_full_blend_joint_20260801/seed_*/`
- 效率：`outputs/p1n_full_blend_joint_20260801/efficiency_benchmark.json`
- 联合实现：`cj0724/p1k_joint_level3_system.py`
- 冻结预测统计：`cj0724/p1k_finalize_saved_predictions.py`
- 全量回归测试：`104 passed`。

方法冻结 SHA-256：`f461e6a3b3beeb1c02fff94d47b67320b850b574b260b7961e95805d9e8d3ed7`。  
`test_used_once=true`，`test_fit_or_selection=false`，`confirm_used=false`，`locked_test_used=false`。
