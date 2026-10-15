# Stage C：ISRJ `sample_width` 物理可辨识性与复合波形提取结果

## 1. 研究问题与证据边界

本阶段只研究 ISRJ `sample_width`（参数列 53，零基索引）。有效样本严格满足 `label[ISRJ]=1 AND parameter_mask[53]=1 AND finite(target)`；所有参数专家均关闭干扰分类损失。

五个宽度区间为：ID `5–20 μs`、near-low `2–5 μs`、near-high `20–30 μs`、far-low `1–2 μs`、far-high `30–50 μs`。每个数据角色包含全部 63 种复合方式、每种 10 个 identity，以及 25/35/50 dB 三个配对功率视图。

所有实验均为 development 或 oracle-positive development；`locked_test_used=false`，未生成、未打开 locked test。

## 2. 逐轮结果

### P2w：最大 prominence ACF 的 exact-component 上界

- 新 identity：seed3900000000。
- exact 分量在 ID、near-low、near-high、far-high 上表现很强；far-low MAE 仅 `0.249 μs`，但 Pearson 为 `-0.087`，未过 `>=0.30` 门槛。
- 失败高度集中：93.4% far-low 样本精确命中；`1.0 μs` 峰位于搜索边界而常被选为 `2.0 μs`，少数 `1.1 μs` 被远端周期峰劫持到 `9–10.1 μs`。
- 结论：物理信息存在，但“最大 prominence 单峰”选择规则不可靠。

### P2x：瞬时频率相位重置序列

- 新 identity：seed4100000000；方法在看数据前冻结。
- exact 分量五区间全部通过：Pearson 约 `1.0`，MAE 为 `0–0.001 μs`，compound 子集同样通过。
- far-low 相比冻结 P2w：MAE 从约 `0.320 μs` 降至 `0`，Pearson 从 `-0.119` 提升至约 `1.0`。
- 结论：相位重置间隔是 ISRJ width 的强物理充分表征，但仅构成 oracle 上界。

### P2y：冻结通用 U-Net 是否保留相位重置

- 新 identity：seed4200000000；冻结 separator seeds 2041/2042/2043。
- 波形余弦为 `0.44–0.66`，但 separated width Pearson 仅约 `-0.02–0.06`。
- ID MAE 约 `11.5 μs`，near-high 约 `23.8 μs`，far-high 约 `39.1 μs`；3/3 seed 均失败。
- 结论：总体波形相似度不能保证稀疏相位边界被保留。

### P2z：专用 U-Net + 相位曲率一致性损失

- 训练 seed3700000000，身份隔离开发 seed3800000000；分类损失为 0。
- waveform-only 与 waveform+reset-consistency 均失败；reset loss 相对 waveform-only 的平均变化为 `-0.06%–+0.03%`，无稳定贡献。
- 三个 seed 的 ID/near-high/far-high Pearson 仍接近 0；`S0_ablation_passed=false`。
- 结论：问题不是简单增加一个 reset loss 权重即可解决；STFT U-Net 的平滑重建接口与稀疏相位事件不匹配。

### P3a：多物理周期序列 + flat ordinal

- 训练 seed3400000000，身份隔离开发 seed3500000000。
- ACF-only ordinal 明显优于 ACF+包络+相位曲率：多物理版本在三个 seed 上平均退化 `11%–17%`。
- ACF-only 在 35/50 dB 的 ID、near-high、far-high 已达到 Pearson 约 `0.62–0.81`；但 near-low/far-low 仍约为 `-0.04–0.20`。
- 主要失败为少量跨区间长尾预测：低宽度样本的中位预测接近目标，但部分预测跳到 `20–40 μs`；25 dB 进一步恶化可见性。
- 结论：raw 复合波形中最有用的信息仍集中在 ACF；额外通道构成稳定负迁移。

### P3b：五宽度带门控 + 带内局部回归

- 训练 seed200000000，身份隔离开发 seed250000000。
- 层级模型相对 flat ordinal 在三个 seed 上平均退化约 `27%–40%`。
- ID MAE 约 `4.05–4.53 μs`，near-low/far-low Pearson 仍接近 0；跨带错误率门槛未通过。
- 结论：显式带门控未解决底层特征不可分问题，反而增加错误传播。

## 3. 成果

1. 建立了可重复的 ISRJ width 物理上界：相位重置间隔在 exact 分量上跨 `1–50 μs` 几乎完美。
2. 证明了“波形重建质量”与“参数充分性”并不等价：余弦相似度通过时，width 仍可完全失序。
3. 通过 3/3 seed 消融否定了三条看似合理但无效的路径：通用 separator、reset-consistent 专用 separator、多物理融合与层级带门控。
4. 明确了可见性边界：ACF-only 在 `>=35 dB` 的中高宽度区间有可用排序，但 25 dB 和 `1–5 μs` 尚不可靠。

## 4. 失败根因

- `sample_width` 由少量相位重置边界决定；STFT 波形损失偏向总体能量和频谱，对这些稀疏事件权重不足。
- 复合波形中的其他调制产生大量相位曲率事件，直接 phase-reset 检测失去特异性。
- 包络与相位曲率 ACF 在复合条件下增加混杂，未提供独立于 complex ACF 的有效信息。
- 低宽度区间目标动态范围仅 `1–3 μs`，少量跨带异常即可摧毁 Pearson；当前 ACF 特征不能稳定区分这些异常。
- 25 dB 下所有学习方法均出现更高跨带错误，说明存在独立的弱可见性瓶颈。

## 5. 冻结决策

- 保留 P2x 作为 oracle physical sufficiency 证据，不作为可部署结果。
- 不将 ISRJ width 专家接入 C2/P1n，不进入预测标签门控或 locked test。
- 不在 seed350、seed250 或任何已打开身份上继续调特征、分离器、损失权重或门控。
- ISRJ width 学习线暂时停止；只有提出能够在原理上抑制复合相位事件、且使用全新身份的可证伪机制时才可重启。

## 6. 主要工件

- `configs/p2w_isrj_width_oracle_ood_preregister_20260801.json`
- `outputs/p2w_isrj_width_oracle_ood_development_20260801/summary.json`
- `configs/p2x_isrj_phase_reset_ood_preregister_20260801.json`
- `outputs/p2x_isrj_phase_reset_ood_development_20260801/summary.json`
- `outputs/p2y_isrj_phase_reset_isolation_development_20260801/summary.json`
- `outputs/p2z_isrj_reset_consistent_separator_development_20260801/summary.json`
- `outputs/p3a_isrj_multiphysics_ordinal_development_20260801/summary.json`
- `outputs/p3b_isrj_hierarchical_width_development_20260801/summary.json`

