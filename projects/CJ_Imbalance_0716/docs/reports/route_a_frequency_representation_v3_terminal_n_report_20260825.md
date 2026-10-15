# Route A Frequency Representation V3：Terminal N 失败复盘

日期：2026-08-25  
项目：`E:\LJQ_code\CJ_Imbalance_0716`  
数据：`E:\LJQ_code\干扰数据\dual-axis-frequency-representation-v3`  
环境：`D:\anaconda3\envs\PY10\python.exe`（PyTorch 2.11.0+cu128，CUDA，RTX 5080）

## 1. 本轮问题与证据边界

本轮在全新 receiver identity（3940000001–3940025200）上验证：冻结 B0-DB 主干，只训练 stage4 rank-4 residual adapter、六个标签条件证据投影、全局 128 维安全旁路，以及 feature-nearest hard-negative ranking。VH/VL 是同一 receiver/component seed 的 30/20 dB SawFM 配对；训练使用 VH，VL 只做可见性诊断。没有打开旧 R2、locked、test-v2 或 test-v3 样本级数据，三个测试标志均为 false。

Stage 0、Stage 1 和 Stage 2 均完成：25,200 identity、63 组合×400、五角色不重叠；5 个新 seed×7 个基线/控制 regime 共 35 cells。旧 RBC-cRT 基线先做了独立聚合，随后候选使用每个 cell 冻结的 B0-DB checkpoint；候选 checkpoint 和 evaluation 预测均独立保存。

## 2. B0/RBC 基线结果

旧 RBC-cRT 相对每个 seed 的 best(F0 Focal、F1 DB、F2 ordinary cRT) 在 LabelIF20+ComboIF20 的 VH macro-AUPR 平均 Δ=-0.00668，5,000 次 paired bootstrap 95% CI 为 [-0.00738,-0.00594]；仅 2/5 seed 的 IF20 均值为正。FB safety 通过，但 Label-LT 主效应门、尾标签门、跨 IF 稳定门和 random-pair 对照门失败。因此 V2 式 co-occurrence ranking 不能成为本轮创新证据。

## 3. V3 候选结果

候选固定损失为：

`L = L_label_balanced + 0.20 L_feature_rank + 5e-4 L_anchor + 1e-4 L_adapter_norm`。

候选可训练参数 13,510；B0 总参数 257,511，增量约 5.25%，已超过预注册的 5% 成本门。每个 seed/regime 的候选 checkpoint 早期选择只使用 early role；evaluation 使用独立 evaluation role。

| 条件（VH） | 候选相对该 cell best(F0/F1/F2/F3) | 5,000 次 receiver-paired CI | 判定 |
|---|---:|---:|---|
| LabelIF20 | -0.00254 | [-0.00355, -0.00152] | 失败 |
| ComboIF20 | -0.000378 | [-0.000500, -0.000262] | 失败 |
| FB | -0.00127 | [-0.00164, -0.000901] | safety 失败 |
| LabelIF20+ComboIF20 联合 | -0.00147 | [-0.00198, -0.000963] | 失败 |

所有 bootstrap 条件的正增益比例均为 0。LabelIF20 的五个 seed 中只有 1 个 VH macro-AUPR 点估计非负；ComboIF20 五个 seed 全部相对 best baseline 为负。ComboIF20 的绝对 AUPR 很高，但对应 baseline 同样极高，不能把“高绝对分数”误报为方法增益。

## 4. 硬门结果

- LabelIF20 macro 增益≥+0.010：失败（-0.00254）。
- LabelIF20 三尾标签平均增益≥+0.020：失败（整体负）。
- ComboIF20 增益≥+0.003：失败（-0.000378）。
- paired bootstrap CI 下界>0：失败（联合 CI 全部低于 0）。
- 至少 4/5 seed 为正：失败（LabelIF20 1/5，ComboIF20 0/5）。
- FB 下降≤0.005：点估计通过数值范围，但 bootstrap 为负，且不改变主门失败结论。
- 参数增量≤5%：失败（约 5.25%）。
- RawSaw、Cartesian/phase-only、2×2 joint、fresh confirmation：未授权、未执行。

结论：`frequency_representation_failed`，本轮 Terminal N。不得在同一 evaluation source 上重调 rank、margin、anchor、adapter rank、阈值或损失权重；不得把本轮候选称为双轴长尾创新的有效模型。

## 5. 失败原因与下一步边界

1. **表征残差没有可迁移的频率主效应**：在 LabelIF20 上候选几乎保持 B0 或下降，说明 stage4 幅度统计与全局安全向量并未提供稳定的弱标签可分性。
2. **Combo-LT 的高绝对分数不是增益**：组合清单本身可分性强，baseline 已接近饱和；候选的小幅下降说明 hard-negative ranking 没有带来额外信息。
3. **标签条件头仍受共享表征限制**：六个独立投影只能重新组织已有 128 维证据，不能恢复被主干池化抹掉的弱局部证据。
4. **复杂度门也未通过**：rank-4 adapter+projection 的 13,510 个可训练参数相对 B0 超过 5% 预算，说明当前改进并非低成本增量。

后续只能另起新的 development source/预注册，优先做表征可识别性诊断（冻结 feature 的 label-wise AUROC、stage4 与全局旁路的互补性、组合匹配 hard-negative 的独立可分性），而不是继续在本源上调参。RawSaw/可见性路线和 confirmation 均保持停止状态。

## 6. 关键产物

- Stage 0 audit：`outputs/route_a_frequency_representation_v3/stage0/stage0_audit.json`，SHA256 `564386d83cfbf3e617f3633f3b070dc24868080c968ac886dc6bd66cc494aa98`。
- Stage 1 data audit：`outputs/route_a_frequency_representation_v3/stage1_data_audit/data_audit.json`，SHA256 `09a71015d4185fb9215d3488932af816d4d04c1b42b6e24d72a38e7a9f0991d6`。
- Stage 2 formal receipt：`outputs/route_a_frequency_representation_v3/stage2_formal/terminal_receipt.json`，SHA256 `c9d902fdab65a47209bf57d8da49cf3b6fb4c18045f76dfe7f3c73fe7494a93a`。
- Candidate point summary：`outputs/route_a_frequency_representation_v3/stage3_candidate/candidate_point_summary.json`，SHA256 `1e49f19ad2bb9399877d8421cbed4e8b44dbc5f7ecc107f79e0e84a0c75c26dc`。
- Candidate bootstrap summary：`outputs/route_a_frequency_representation_v3/stage3_candidate/candidate_bootstrap_summary.json`，SHA256 `2e44bb6a4d9c54715c82eeea6a7750955943758561a7a25ec40cef1604632cf9`。
