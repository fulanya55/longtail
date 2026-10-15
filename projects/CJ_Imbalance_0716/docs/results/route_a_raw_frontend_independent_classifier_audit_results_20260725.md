# 路线 A：Raw SawFM 独立分类器全分布审计结果（2026-07-25）

## 1. 结论

Raw-IQ 前端作为独立 SawFM 分类器通过 3/3 seeds 全部门禁；Platt 与 Isotonic 均通过。推荐保留结构更简单、排序不受分段常数映射影响的 `I1_RawPlatt`，进入五种子确认，但暂不替换六类主模型中的 H2，也不解冻 locked test。

本轮同时澄清了上一轮“有界残差重连失败”的原因：Raw 分支不是没有 SawFM 证据，而是其分数几何与 H2 不兼容。把 Raw 压缩成 H2 上的 `<=0.20` 小残差，会破坏其强排序与独立决策能力；直接校准并独立判决后，普通域、弱功率域和非 swept 子群均稳定改善。

## 2. 固定协议

- 环境：`D:\anaconda3\envs\PY10\python.exe`，正式运行设备为 CUDA；
- 数据：0716 六类数据协议；power views 按 `receiver_noise_seed` 隔离；
- seeds：2041、2042、2043；
- 网络：冻结 `RawStage1SawFMFrontend(mode="raw", hidden_channels=16, topk=16, dropout=0.1)`，5,521 参数；
- checkpoint：各 seed 的 `N0_RawOnly/best_frontend.pth`；
- 校准：仅 ordinary calibration；
- 阈值选择：ordinary threshold 与 40/45/50 dB power threshold；网格 `0.01:0.01:0.99`；所有域 FPR 必须 `<=0.20`；
- 独立评价：ordinary early 与 40/45/50 dB power development；
- locked test：未使用。

比较项：

| 变体 | 分数 | 校准 |
|---|---|---|
| I0 | H2 SawFM logit | ordinary Platt |
| I1 | Raw frontend logit | ordinary Platt |
| I2 | Raw frontend logit | ordinary Isotonic |

## 3. 三种子聚合结果

| 指标 | I0 H2+Platt | I1 Raw+Platt | I2 Raw+Isotonic |
|---|---:|---:|---:|
| ordinary AUROC | 0.7074 | **0.9375** | 0.9331 |
| ordinary AUPR | 0.7709 | **0.9548** | 0.9388 |
| ordinary recall | 0.4878 | **0.8687** | 0.8664 |
| ordinary FPR | 0.1680 | 0.1720 | 0.1720 |
| ordinary Brier | 0.2128 | **0.0995** | 0.1005 |
| ordinary ECE | 0.0605 | 0.0457 | **0.0420** |
| with-swept SawFM recall | 0.4285 | **0.8747** | 0.8721 |
| without-swept SawFM recall | **0.9164** | 0.8344 | 0.8344 |
| power mean recall | 0.0346 | **0.9809** | **0.9809** |
| 50 dB recall | 0.0693 | **0.9628** | **0.9628** |
| max power FPR | 0.1683 | **0.0695** | **0.0695** |

相对 H2，Raw+Platt 的主要变化为：ordinary AUPR `+0.1839`、ordinary recall `+0.3809`，而 ordinary FPR 仅增加约 `0.0040`；power mean recall 增加约 `0.9463`，50 dB recall 增加约 `0.8936`。Brier 由 `0.2128` 降至 `0.0995`。

## 4. Raw+Platt 分种子结果

| seed | 阈值 | ordinary AUPR | ordinary recall/FPR | without-swept recall | power mean recall | 40/45/50 dB recall | 40/45/50 dB FPR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2041 | 0.35 | 0.9591 | 0.8523 / 0.1258 | 0.7143 | 0.9762 | 0.9898 / 0.9796 / 0.9592 | 0.0408 / 0.0306 / 0.0102 |
| 2042 | 0.39 | 0.9387 | 0.8481 / 0.1901 | 0.8889 | 0.9700 | 1.0000 / 0.9700 / 0.9400 | 0.0400 / 0.0300 / 0.0200 |
| 2043 | 0.41 | 0.9666 | 0.9056 / 0.2000 | 0.9000 | 0.9965 | 1.0000 / 1.0000 / 0.9894 | 0.1277 / 0.0319 / 0.0213 |

所有种子同时满足：ordinary AUPR 不低于 H2 `-0.02`、ordinary FPR `<=0.20`、recall 相对 H2 提升 `>=0.05`、without-swept recall `>=0.60`、power mean recall `>=0.70`、50 dB recall `>=0.60`、各 power FPR `<=0.20`、Brier `<=0.20`、ECE `<=0.10`。

## 5. 机制解释

1. **Raw 证据具有独立可分性。** 普通域 AUROC 0.9375、power AUROC 接近此前的高水平，说明其能力不局限于组合匹配 hard negatives。
2. **不是 swept-background 捷径。** without-swept SawFM recall 三种子均超过 0.60，均值 0.8344，因此不能把该前端仅解释为“扫频背景专家”。
3. **H2 的主要问题是 with-swept 弱分量。** H2 对 without-swept 的召回很高（0.9164），但 with-swept 仅 0.4285；Raw 将后者提高至 0.8747，同时保留较高的 without-swept recall。
4. **失败来自融合约束。** 上轮残差幅值被限制在 0.20，并受 H2 置信度抑制；这会让 Raw 无法跨越 H2 已形成的错误边界。Raw 应先被视为标签专属判决器，而不是共享头的小修正项。
5. **Platt 优于 Isotonic 作为默认方案。** 两者 recall/FPR 相同，但 Platt 的 AUPR 更高、参数更少、跨种子解释更稳定；Isotonic 仅保留为校准敏感性对照。

## 6. 证据边界

- 本轮只证明了 SawFM 标签的冻结独立决策器在三个既有模型种子上有效；尚未证明六标签整体 macro 指标提高。
- ordinary early 每 seed 最多 300 样本，without-swept 正样本数量较少，仍需扩大确认。
- Raw checkpoint 的训练角色包含 ordinary calibration；校准器也使用该角色，但 threshold 与 early/development 完全独立。下一轮需要固定这一协议并避免重新调参。
- 2039/2040 虽已有 Raw 新 split checkpoint，但缺少同协议 F2/H2/visibility replay 链，不能直接与本轮 2041–2043 聚合。
- locked test 仍冻结，当前结果不能表述为最终测试结果或论文最终效果。

## 7. 下一阶段决策

晋级 `Raw+Platt`，停止继续尝试“小权重 Raw→H2 残差”。下一阶段执行五种子确认与六类决策接入前审计：

1. 为 2039/2040 补齐与 2041–2043 同协议的 F2/H2 replay 产物；
2. 固定 Raw 网络、训练预算、Platt 校准和 FPR 约束，不再使用现有结果调参；
3. 五种子复核 ordinary、with/without-swept、40/45/50 dB 指标；
4. 只有五种子门禁通过，才设计“标签级路由”：SawFM 使用 Raw+Platt，其他五类保持 H2；
5. 路由后先在 development 评估六类 macro-AUPR、每类 FPR、组合一致性和 ECE；通过后再申请一次性 locked test。

## 8. 产物

- 方案：`docs/plans/route_a_raw_frontend_independent_classifier_audit_plan_20260725.md`
- 程序：`experiments/run_route_a_raw_frontend_independent_classifier_audit.py`
- 测试：`tests/test_raw_frontend_independent_classifier_audit.py`
- 输出：`outputs/tapc/route_a_raw_frontend_independent_classifier_audit/`
- 汇总：`outputs/tapc/route_a_raw_frontend_independent_classifier_audit/route_a_raw_frontend_independent_classifier_audit_summary.json`
- 汇总 SHA256：`04921B8DB6E7E0F4C46281AFC5224730849A7EB6517A6B090A29620E3635C77D`

