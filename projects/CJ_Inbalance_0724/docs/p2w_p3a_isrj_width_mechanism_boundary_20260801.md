# ISRJ sample-width 物理机制长线审计：P2w–P3a

日期：2026-08-01  
证据等级：全新 identity-safe development；没有生成或读取 test；`locked_test_used=false`。

## 1. 研究问题

P2a 九功率一次性验证中，分类、逐功率风险、弱分量和组合最坏值均通过，但全 12 参数 predicted-gate NMAE 改善为 `4.7769%`，低于预注册 `5%` 门槛约 `0.2231` 个百分点。ISRJ sample width 在既有多任务模型中有较强可学习证据，因此本阶段只回答三个递进问题：

1. ISRJ sample width 在 exact physical component 上是否可辨识？
2. 已有或专用分离器能否把该物理证据从复合 I/Q 中保留下来？
3. 若 STFT 分离失败，样本级时域网络是否能够修复局部相位重置？

每轮只改变一个机制，失败后更换全新身份角色；没有回读 P1n/P2a 已消费测试的逐样本预测。

## 2. 阶段结果

### P2w：最大显著度 ACF 宽度估计

- ID-control exact component：MAE `0.1769 us`，Pearson `0.9758`。
- near-low/near-high/far-high 的 exact component 误差很低。
- far-low `[1,2] us`：MAE仍仅 `0.2494 us`，但 Pearson 为 `-0.0871`，未通过排序门。
- 失败诊断定位到搜索下界端点不可被 `find_peaks` 选中，以及少量远端谐波抢占。

结论：不能晋级原 ACF 规则，但失败并不等于物理信息不存在。

### P2x：相位重置间隔估计

全新 seed `4100000000`；五个区间、25/35/50 dB、63 组合。

- exact-component 五区间的 MAE 均接近 0，Pearson 均接近 1。
- far-low 从 P2w 的负相关恢复为 MAE `0 us`、Pearson `1.0`。
- time-shuffle 破坏控制显著更差。
- 所有预注册物理充分性门通过。

结论：ISRJ sample width 的 phase-reset 机制在 exact component 上物理充分，允许测试分量隔离；这不是一个可直接部署的复合 I/Q 专家。

### P2y：冻结共享 U-Net 隔离迁移

原 P2y seed `4200000000` 和补充确认 seed `4150000000` 都失败。补充确认的三个冻结 U-Net 种子：

| 模型 seed | 平均 width MAE 改善 vs raw | ID-control 改善 | 平均 waveform cosine |
|---:|---:|---:|---:|
| 2041 | -6.17% | +1.30% | 0.575 |
| 2042 | -5.56% | +1.55% | 0.585 |
| 2043 | -2.01% | +0.84% | 0.602 |

波形余弦相似度通过，但 width Pearson、灾难错误、相对 raw 改善均失败。

结论：通用扫频 U-Net 保留总体波形形状，却没有保留 ISRJ 宽度所需的局部相位事件；不得接入参数路由。

### P2z：专用 STFT-U-Net + phase-curvature consistency

训练 seed `3700000000`，独立开发 seed `3800000000`；三模型种子；S0 waveform-only 与 S1 waveform+reset 等算力。

- S1 相对 S0 的三种子平均 MAE 改善为 `-0.016%`，近似为零且方向略差。
- S1 虽将 selection reset loss 相对 S0 降低约 8%–9%，但五区间 Pearson 仍接近 0。
- S0 与 S1 的 waveform cosine 约 `0.56–0.77`，两者绝对物理门均失败。

结论：单纯增加 reset consistency loss 不能把高波形相似度转化为可排序的 sample width；不再搜索 loss 权重。

### P3a：不降采样时域复数残差分离器

训练角色与 P2z 完全相同；全新开发 seed `4000000000`；唯一机制变化为 STFT-U-Net → 时域 dilated residual separator。

- 参数量：`2,914`，仅为 P2z `83,095` 参数的 `3.51%`。
- 三种子训练时间约 `4.9–5.8 s`。
- 相对冻结 P2z-S1 的平均 width MAE 改善：`-0.231% / -0.093% / -0.218%`。
- 五区间 waveform cosine 仍约 `0.60–0.64`，但 Pearson 仍接近 0。
- ID-control、near-high、far-high 的绝对误差和灾难错误门失败。

结论：失败不只是 STFT hop 或下采样造成。以波形重建为主目标时，模型可以获得较高全局相似度，却不必恢复决定宽度排序的稀疏相位重置事件。按预注册规则，学习式 waveform separator 路线停止。

## 3. 根本原因

当前证据支持以下因果链：

1. **目标可辨识**：exact component 上 phase-reset 专家近乎精确，排除了物理目标本身不可辨识。
2. **原始混合不可直接解析**：同一解析器在复合 I/Q 上 Pearson 接近 0，说明其他干扰的局部相位变化淹没了目标事件。
3. **全局波形相似度不是参数充分统计量**：共享、专用 STFT 和时域分离器都能得到中高 waveform cosine，却不能恢复 width 排序。
4. **辅助损失与架构替换均未改变统计边界**：reset loss 降低、STFT 替换为无下采样时域网络后，width 指标仍不变。

因此，继续加深分离器、扩大通道数或调 reset 权重缺乏可证伪依据，容易形成无效模型搜索。

## 4. 当前可保留的创新点

可以可靠保留：

- P1n 的 bootstrap lower-confidence ISRJ 分类路由与逐功率有限样本校准；
- P2a 的九功率分类/风险/弱分量/组合方向性证据，但不能宣称已通过 5% 全参数 Level-4 门；
- P2x 的“exact-component phase-reset 物理可辨识性”与 P2y–P3a 的“复合混合可达性边界”；
- 一套有明确 Oracle gate、身份隔离、破坏控制和停止规则的负结果链。

不能宣称：

- 已获得可部署的 ISRJ sample-width 物理专家；
- 已解决九功率全 12 参数 5% 门槛；
- waveform cosine 提升等价于参数可用性提升；
- P2x exact oracle 可以直接接入 P1n/P2a 系统。

## 5. 下一轮唯一允许的机制

不再输出或重建 ISRJ 波形。下一轮只筛查：

> 直接从 raw I/Q 构造幅度加权 phase-curvature / reset-periodicity 特征，由一个小型标量专家预测 sample width，并与 IQ-Grouped 原参数头和常数基线比较。

预注册要求：

- 全新 identity seed；不使用 seed `3800000000` 或 `4000000000` 选择特征、阈值或模型；
- 固定一套物理特征和一个小模型，不做特征集合搜索；
- 三模型种子，逐区间、逐功率、compound、最坏组合和 OOD 报告；
- 与 IQ-Grouped 原参数头做相同身份配对比较；
- 若 ID-control Pearson `<0.30` 或三种子相对原参数头不改善，则停止 ISRJ width 系统附件，只保留 Oracle/边界创新；
- 只有通过后才允许构造不确定性有界参数路由并重新计算全 12 参数 NMAE；
- locked test 继续保持未授权、未读取。

## 6. 证据与产物

- `outputs/p2w_isrj_width_oracle_ood_development_20260801/summary.json`
- `outputs/p2x_isrj_phase_reset_ood_development_20260801/summary.json`
- `outputs/p2y_isrj_phase_reset_isolation_development_20260801/summary.json`
- `outputs/p2y_isrj_shared_unet_isolation_development_20260801/summary.json`
- `outputs/p2z_isrj_reset_consistent_separator_development_20260801/summary.json`
- `outputs/p3a_isrj_time_domain_separator_development_20260801/summary.json`
- 对应 preregistration、checkpoint、history、receipt、identity audit 与开发预测均已保存。
