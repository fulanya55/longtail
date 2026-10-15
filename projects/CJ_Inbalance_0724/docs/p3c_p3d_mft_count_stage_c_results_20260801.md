# P3c–P3d MFT count Stage-C 结果

## 结论

MFT `num_targets` 的精确分量物理可辨识性已经得到强支持，但从原始复合 IQ 稳定提取 count 尚未成立，因此当前不得接回 C2/P1n。

- P3c 精确 MFT 分量 + 正确参考反卷积：5/5 区间均为 MAE=0、Pearson=1、精确率=100%，所有预注册门槛通过。
- P3c 原始复合 IQ + 逐样点阈值计数：明显失败；响应被噪声和其他干扰的连续超阈值样点污染，预测常饱和到 8。
- P3d 原始复合 IQ + 参考反卷积延迟序列 + ordinal 头：在常规间隔上获得有限且可重复的增益，但 5 区间、3 种子冻结门槛未通过。
- 全部结果仅属于 oracle-positive、identity-safe development 证据；分类损失关闭，未生成或打开新 test，`locked_test_used=false`。

## 参数与波形对应关系审计

- MFT 类别索引：3（zero-based）。
- `num_targets`：参数列 28。
- `delay_01_s ... delay_08_s`：参数列 31–38。
- `amplitude_01 ... amplitude_08`：参数列 39–46。
- delay 真值是相对目标时延与公共到达时延之和，因此配置的 30–50 us 相对区间在文件中约表现为 40–100 us 可观测时延。
- 不同机制区间的参数值和原始复合波形均发生同步变化；没有发现标签—波形脱节。
- 每个区间 630 个身份、320 个 MFT positive；25/35/50 dB 为相同接收机噪声身份的配对功率视图，不作为独立身份计数。

## P3c：稀疏延迟反卷积 oracle

### 冻结设计

- 新身份种子：450000000。
- 区间：ID control、low count、close spacing、ultra close、weak target。
- 正确参考：10 MHz 采样、100 us LFM。
- 反卷积：`FFT(x) conj(R) / (|R|^2 + 1e-6 max|R|^2)`。
- 对照：错误 chirp rate、确定性时序打乱、常数 5.5。

### 结果

精确分量在所有区间、复合组合子集和功率档均恢复正确 count：MAE=0、Pearson=1、accuracy=1。错误参考和时序打乱对照更差，说明成功依赖正确物理参考与时序结构。

原始复合 IQ 的简单规则失败。例如 ID control 的 MAE=2.523、Pearson=0.086、accuracy=0.184；low count 的 MAE=6.397、Pearson=0.078、accuracy=0.016。失败原因是旧规则统计“超阈值样点”，而不是稀疏局部峰，复合背景使计数饱和。

证据边界：P3c 证明参数信息存在于精确 MFT 分量，不证明分离器、门控器或端到端模型能够保留和提取该信息。

## P3d：复合 IQ 延迟序列 ordinal 专家

### 冻结设计

- 训练身份种子：460000000；开发身份种子：470000000；交集为 0。
- 模型种子：2041、2042、2043。
- 输入：正确参考反卷积后的 35–110 us 延迟区间，包含 log 幅度、局部背景正对比、局部尖锐度和位置四个通道。
- 模型：小型 1D CNN + 7 级 ordinal logits，输出 1–8 count。
- 优化：AdamW，lr=3e-4，weight decay=1e-4，最多 20 epochs，patience=5，batch size=64。
- 损失：ordinal BCE + 0.05 单调性惩罚；分类损失关闭。
- 基线：只由训练身份拟合的常数中位数 5。

### 冻结开发结果

| 区间 | 3 seeds MAE 改善范围 | 3 seeds Pearson 范围 | 判定 |
|---|---:|---:|---|
| ID control | 10.1%–14.5% | 0.396–0.471 | overall 勉强通过；compound 有一个种子低于 10% |
| low count | 42.7%–46.4% | 0.255–0.348 | MAE 强，但相关性非 3/3 通过 |
| close spacing | 5.5%–10.7% | 0.330–0.390 | 相关性存在，MAE 门槛不稳定 |
| ultra close | -16.9%–-12.9% | 0.072–0.108 | 明确失败，劣于常数 |
| weak target | 7.0%–7.9% | 0.417–0.471 | 排序信息存在，但 MAE 改善不足 |

全部 overall 与 compound 的预注册联合门槛均未通过。训练后汇总阶段曾发生 Python 布尔拼写错误；汇总由已完成的三个不可变 seed receipt 重建，没有重训模型或改变预测，并保存独立 recovery receipt。

## 成果与失败原因

### 已取得的成果

1. 从“全局 FFT/全局特征回归”推进到与 MFT 生成机制一致的参考反卷积延迟序列。
2. 精确分量上跨 count、间隔和弱幅度区间达到完美恢复，排除了“count 参数本身没有进入波形”的可能。
3. 原始复合 IQ 上，常规区间出现跨 3 seeds 的正相关和约 10% 以上 MAE 改善，证明物理前端确实提取到部分 count 信息。
4. 失败集中到可解释边界：超近间隔的字典高度相关、弱分量被强分量与复合背景遮蔽、low-count 虽能区分低/高但类内排序不足。

### 尚未形成可靠创新点的原因

1. P3d 未满足预注册的 3/3 seeds、每区间、compound 一致性门槛。
2. ultra-close 不是优化抖动，而是三个种子同方向失败，提示物理分辨率/条件数限制。
3. weak-target 的 Pearson 较高但 MAE 改善不足，说明模型能排序，却没有可靠计数校准。
4. 仍是 oracle-positive 参数评估，尚未包含预测标签门控造成的漏检与误路由。
5. 尚未证明该专家相对共享多任务头的统计显著优势、端到端净收益或计算代价可接受。

## 冻结决策与后续门槛

- 不在 seed470000000 上继续调阈值、结构或超参数。
- 不把 P3d 接回 C2/P1n。
- MFT count 的下一项试验若继续，必须换新身份并提出新的可证伪机制，例如显式稀疏追踪/分辨率感知的不确定性拒识；仅增加 CNN 容量不足以构成新机制。
- 更稳妥的论文路线是把“分辨率感知可靠性路由”作为创新核心：在可辨识区间输出参数，在 ultra-close/弱支持区间拒识或退回常数/区间预测，并证明风险覆盖曲线、校准与端到端净收益。
- 在方法与规则完全冻结前，不申请 locked test。

## 主要产物

- `configs/p3c_mft_sparse_deconvolution_preregister_20260801.json`
- `outputs/p3c_mft_sparse_deconvolution_development_20260801/summary.json`
- `configs/p3d_mft_inverse_ordinal_preregister_20260801.json`
- `outputs/p3d_mft_inverse_ordinal_development_20260801/summary.json`
- `outputs/p3d_mft_inverse_ordinal_development_20260801/summary_recovery_receipt.json`
- `cj0724/p3c_mft_sparse_deconvolution.py`
- `cj0724/p3d_mft_inverse_ordinal.py`
