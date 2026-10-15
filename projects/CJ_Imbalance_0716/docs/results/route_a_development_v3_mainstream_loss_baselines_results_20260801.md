# Route A development-v3 主流损失公平基线结果

## 1. 结论

在相同 F2 初始化、partial-scope、训练顺序、RawSaw 路由、Platt 校准和 `FPR<=0.20` 阈值协议下，DB Loss 是本轮 mean macro-AUPR 最强的传统损失基线。冻结 F3 周期融合头相对 DB 的预注册 ISRJ 专项目标全部通过：AUPR、弱召回、强召回和弱 AUROC 均为 5/5 种子正向，且没有明显增加平均 ISRJ FPR。

但是，本轮还不能独立支撑“传统损失已经充分优化”的最终主张。BCE、Focal、ASL、DB、CB-BCE 和 Logit Adjustment 在五个种子上的最佳 epoch 全部落在预算上限 12。该现象不否定当前公平比较，却表明传统基线可能尚未收敛。F3 因而保留为 development-v3 候选，下一步必须先做训练预算收敛审计。

## 2. 公平协议

- 数据：development-v3，SHA256 `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`；
- 种子：2039--2043，复用冻结 role indices；
- 初始化：各 seed 的同一 `F2_DFMS_Full`；
- 可训练范围：`temporal_backbone[-1]` 53,376 参数和 `Linear(128,6)` 774 参数；
- 优化器：AdamW；分类头 LR `5e-4`，共享层 LR `5e-5`，weight decay `1e-4`；
- batch 64，最多 12 epochs，patience 4，gradient clip 5；
- 训练损失：common-five loss + `0.2 * SawFM loss`；
- 推理：所有候选使用同 seed 冻结 RawSaw 替换 SawFM logit；
- 校准与阈值：每个方法独立用 calibration role 做 Platt 校准，threshold role 在逐标签负样本 `FPR<=0.20` 下最大化弱召回；
- test-v3：未访问、未校准、未用于选择。

## 3. 各方法五种子结果

| 方法 | macro-AUPR | macro-F1 | ISRJ AUPR | ISRJ 弱召回 | ISRJ 弱 AUROC | ISRJ FPR |
|---|---:|---:|---:|---:|---:|---:|
| BCE | 0.885069 | 0.773719 | 0.840266 | 0.262832 | 0.563515 | 0.193226 |
| PosBCE | 0.877927 | 0.765668 | 0.833501 | 0.250143 | 0.562852 | 0.190968 |
| Focal | 0.888382 | 0.778240 | 0.846775 | 0.256451 | 0.568765 | 0.193387 |
| ASL | 0.888496 | 0.778909 | 0.846757 | 0.260006 | 0.569608 | 0.194355 |
| **DB** | **0.889218** | **0.779523** | **0.846945** | 0.256997 | 0.568823 | 0.191935 |
| CB-BCE | 0.885024 | 0.773795 | 0.840377 | 0.262832 | 0.563568 | 0.193226 |
| Logit Adjustment | 0.882640 | 0.771633 | 0.833712 | 0.253769 | 0.564115 | 0.191774 |
| C4 同容量参考 | 0.881279 | 0.770427 | 0.842995 | 0.258586 | 0.566456 | 0.193387 |
| **F3 周期融合** | 0.884445 | 0.777731 | **0.861988** | **0.334977** | **0.581412** | 0.193871 |

DB 在整体 macro-AUPR 上最强；F3 的优势集中在目标机制 ISRJ，而不是全局 macro 指标。这与“周期专家补足 ISRJ 长周期弱证据”的局部创新定位一致，也要求论文不能把它描述成所有标签上的统一性能提升。

## 4. F3 相对最强传统基线 DB

| 指标 | 平均差 | bootstrap 95% CI | 正向种子 | 单侧 Wilcoxon p | 单侧符号检验 p |
|---|---:|---:|---:|---:|---:|
| ISRJ AUPR | +0.015043 | [0.010551, 0.020520] | 5/5 | 0.03125 | 0.03125 |
| ISRJ 弱召回 | +0.077980 | [0.051126, 0.109212] | 5/5 | 0.03125 | 0.03125 |
| ISRJ 强召回 | +0.056803 | [0.049324, 0.062836] | 5/5 | 0.03125 | 0.03125 |
| ISRJ 弱 AUROC | +0.012588 | [0.005846, 0.022213] | 5/5 | 0.03125 | 0.03125 |
| ISRJ FPR | +0.001935 | [-0.008871, 0.012581] | 3/5 | 0.50000 | 0.50000 |
| macro-AUPR | -0.004773 | [-0.006617, -0.003096] | 0/5 | 1.00000 | 1.00000 |
| macro-F1 | -0.001792 | [-0.005402, 0.002420] | 1/5 | 0.84375 | 0.96875 |

预注册门槛 7/7 通过。F3 的 ISRJ 改善具有跨种子一致性，平均 FPR 增量的区间跨 0；但 macro-AUPR 明确低于 DB。因此当前被支持的陈述是“对 ISRJ 弱周期证据有额外贡献”，不是“整体最优分类器”。

## 5. 训练稳定性与证据缺口

除复用同协议历史 checkpoint 的 PosBCE 外，六个重新训练损失在所有五个种子上均运行满 12 epochs，且最佳 checkpoint 均为 epoch 12。单个方法五种子总训练时间约 22--25 秒，表明增加预算的验证成本较低。

该结果产生一个明确的反证风险：若 DB、ASL 或 Focal 在 24--36 epochs 后继续提高，F3 相对最强传统损失的优势可能缩小。因此：

1. 本轮结果可作为同预算公平证据；
2. 尚不能作为传统基线已饱和的证据；
3. 下一步应只扩展最强且相邻的 DB、ASL、Focal，不搜索损失超参数；
4. 使用预注册的固定更长预算和相同选择指标，检查曲线是否平台化；
5. 若延长后的传统基线达到 F3，必须提升该基线并收缩创新主张。

## 6. 当前创新证据边界

已支持：

- 同预算、同主干、同路由和同阈值协议下，单纯损失重加权没有复现 F3 的 ISRJ 弱证据收益；
- F3 相对最强 DB 的 ISRJ AUPR、弱召回、强召回和弱 AUROC均为 5/5 正向；
- 该收益没有以显著的平均 ISRJ FPR 增长换取；
- test-v3 未使用。

尚未支持：

- 传统损失已充分收敛；
- F3 在 macro-AUPR 上优于 DB；
- F3 优于两阶段解耦分类头或同容量非周期 MLP；
- F3 在身份隔离外部/OOD 数据上保持收益并控制最坏子组 FPR；
- 最终 frozen candidate 在 test-v3 上成立。

## 7. 后续门槛

先执行 36-epoch 收敛审计，保留 12、18、24、30、36 epoch 轨迹，并对 DB、ASL、Focal 使用相同种子与固定超参数。只有当最强长预算传统基线的平台化得到验证，才进入两阶段解耦与同容量 MLP 基线；此后才是外部/OOD、工程统计和最终封存。

## 8. 产物

- 预注册：`docs/plans/route_a_development_v3_mainstream_loss_baselines_plan_20260801.md`；
- 实现：`experiments/run_route_a_development_v3_mainstream_loss_baselines.py`；
- 测试：`tests/test_route_a_development_v3_mainstream_loss_baselines.py`，4 passed；
- 正式输出：`outputs/tapc/route_a_development_v3_mainstream_loss_baselines`；
- 汇总 SHA256：`340A7FD4BA79B86CC2855119CA9D665F1AEC4CFE7BF14A5F1B379CFEDF8DA1C9`；
- test-v3：`access_authorized=false, analysis_allowed=false, used=false`。
