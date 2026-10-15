# 路线 A：Raw+Platt 五种子确认与六类标签路由结果（2026-07-26）

## 1. 总结结论

本阶段完成了预定的 F1、F2、F3：

1. 为 seed 2039/2040 补齐了与既有实验相同配置的 F2 主干和 H2 冻结分类头；
2. Raw+Platt 独立 SawFM 分类器完成五种子冻结确认，达到预设的 `4/5 seeds` 晋级条件；
3. `SawFM=Raw+Platt，其他五类=H2` 的标签级路由在 `5/5 seeds` 上通过全部 development 门禁；
4. locked test 未使用。当前允许形成 locked-test 申请包，但不能自动执行最终测试。

五种子结果进一步验证：Raw 分支的正确定位是 SawFM 标签专属分类器，而不是 H2 的小幅残差或六类共享特征修正器。

## 2. 固定实验协议

- 环境：`D:\anaconda3\envs\PY10\python.exe`，CUDA；
- 数据：0716 六类训练数据与 40/45/50 dB power views；
- seeds：2039、2040、2041、2042、2043；
- F2：`F2_DFMS_Full`，label-LT 2,400 样本，DB Loss，20 epochs；
- H2：冻结 F2，`H2_PosBCE`，60 epochs；
- Raw：`RawStage1SawFMFrontend(mode="raw", hidden_channels=16, topk=16, dropout=0.1)`；
- Raw 校准：ordinary calibration 上的 Platt；
- Raw 阈值：ordinary threshold 与 power threshold 联合网格，步长 0.01，所有 threshold 域 FPR `<=0.20`；
- 评价：ordinary early 和 power development；
- 路由：只替换 SawFM 概率与阈值，其他五类的 H2 概率、阈值和预测保持不变；
- locked test：未使用。

2039/2040 使用此前冻结的 `R0_FullPhysics` Raw checkpoint；2041–2043 使用此前冻结的 `N0_RawOnly` checkpoint。两组使用相同网络和优化超参数，但 checkpoint 的训练角色抽样属于两个已审计协议，因此汇总中保留了 `frontend_protocol` 来源字段。

## 3. F2/H2 补齐结果

| seed | F2 split SHA256 | H2 split SHA256 | 一致性 |
|---:|---|---|---|
| 2039 | `93643a2579be41f7f04d85a1df49d75fc8b1055cf11d88d91f54cf1a818cb4b3` | 同左 | 通过 |
| 2040 | `8d7620defb6efc0ac7bcaf6ee8b953e111de3c78d7e7f64d43f941bd4e1c3717` | 同左 | 通过 |

F2/H2 receipt 均为 completed，locked-test 标志均为 false。

## 4. Raw+Platt 五种子确认

### 4.1 聚合结果

| 指标 | 五种子均值 | 晋级线 |
|---|---:|---:|
| ordinary AUROC | 0.9384 | 诊断指标 |
| ordinary AUPR | **0.9547** | >=0.90 |
| ordinary recall | **0.8813** | >=0.80 |
| ordinary FPR | 0.1960 | 单 seed 门禁 <=0.20 |
| without-swept recall | **0.8653** | >=0.70 |
| power mean recall | **0.9879** | >=0.90 |
| 50 dB recall | **0.9758** | >=0.85 |
| max power FPR | 0.0760 | <=0.20 |
| Brier | 0.0977 | <=0.20 |
| ECE | 0.0504 | <=0.10 |

预设门禁要求至少 4/5 seeds 通过当前九项检查；实际为 4/5，通过。

### 4.2 分种子结果

| seed | 阈值 | ordinary AUPR | recall/FPR | without-swept recall | power mean recall | 50 dB recall | max power FPR | 单 seed 门禁 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2039 | 0.33 | 0.9558 | 0.9130 / **0.2806** | 1.0000 | 1.0000 | 1.0000 | 0.1340 | 未通过 |
| 2040 | 0.40 | 0.9535 | 0.8873 / 0.1835 | 0.8235 | 0.9969 | 0.9907 | 0.0374 | 通过 |
| 2041 | 0.35 | 0.9591 | 0.8523 / 0.1258 | 0.7143 | 0.9762 | 0.9592 | 0.0408 | 通过 |
| 2042 | 0.39 | 0.9387 | 0.8481 / 0.1901 | 0.8889 | 0.9700 | 0.9400 | 0.0400 | 通过 |
| 2043 | 0.41 | 0.9666 | 0.9056 / 0.2000 | 0.9000 | 0.9965 | 0.9894 | 0.1277 | 通过 |

seed 2039 的唯一失败项是 ordinary early FPR `0.2806`。该 seed 的阈值角色满足 FPR 约束，但 early 角色发生分布波动；其 AUPR、召回、without-swept、power、Brier/ECE 和身份审计均通过。因此这是阈值跨角色稳定性风险，不是特征可分性失败。为避免使用 early 结果二次调参，本阶段没有调整 seed 2039 阈值。

## 5. 六类标签级路由

### 5.1 结构

- `L0_H2All`：六类均使用 H2；
- `L1_RawSaw_H2Others`：SawFM 使用 Raw+Platt，其余 FMB、SinFM、MFT、VFT、ISRJ 完全保持 H2。

没有学习融合权重，没有 oracle power，没有按组合路由，也没有改变其他标签的分数。

### 5.2 五种子聚合结果

| 指标 | H2All | RawSaw+H2Others | 变化 |
|---|---:|---:|---:|
| macro-AUPR | 0.8667 | **0.9001** | **+0.0334** |
| macro-F1 | 0.7573 | **0.8027** | **+0.0454** |
| SawFM recall | 0.4870 | **0.8813** | **+0.3943** |
| exact match | 0.2280 | **0.2747** | **+0.0467** |
| 其他五类最大 FPR 变化 | — | — | **0.0000** |

标签路由 5/5 seeds 同时满足：

- macro-AUPR 不下降；
- macro-F1 提升；
- SawFM recall 提升至少 0.20；
- 其他五类最大 FPR 增量不超过 0.01；
- exact match 不下降超过 0.01。

## 6. 机制与风险分析

### 已验证

1. Raw 前端在普通六类分布和弱功率 hard-negative 分布中均具有稳定 SawFM 排序能力；
2. 能力不依赖 swept 背景，without-swept recall 五种子均值 0.8653；
3. 标签级替换能把单标签改善传递到六类 macro-AUPR、macro-F1 和 exact match；
4. 因为只替换 SawFM，其他五类 FPR 数值上保持不变；
5. 之前有界残差失败来自错误的融合几何，而非 Raw 特征不足。

### 尚未验证

1. locked test 上的最终泛化；
2. seed 2039 暴露的 ordinary FPR 跨角色波动是否会在 locked test 重复；
3. 两组 Raw checkpoint 抽样协议差异是否会影响更大规模复现；
4. 最终论文应使用单模型、五模型独立报告还是预先声明的 ensemble；本轮没有决定 ensemble，也没有使用其结果。

## 7. 决策

冻结 `RawSaw+H2Others` 作为路线 A 当前 development 候选终版，停止以下方向：

- Raw→H2 小权重残差；
- 重新学习六类融合权重；
- 为 seed 2039 在 early 上重新调阈值；
- 在 locked test 前继续增加结构模块。

下一步仅允许：审查并批准 locked-test 申请包；批准后按照冻结配置一次性执行，禁止看到测试结果后再改模型、校准或阈值。

## 8. 产物

- 程序：`experiments/run_route_a_raw_platt_five_seed_label_routing.py`
- 测试：`tests/test_raw_platt_five_seed_label_routing.py`
- 输出：`outputs/tapc/route_a_raw_platt_five_seed_label_routing/`
- 汇总：`outputs/tapc/route_a_raw_platt_five_seed_label_routing/route_a_raw_platt_five_seed_label_routing_summary.json`
- 汇总 SHA256：`1C456FF579F0FA2EFCCBA202152F5F2CA117CA215A7D84131447D4086D491EFC`

