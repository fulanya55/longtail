# 路线 A：稳定专家选择与可见度条件校准结果（2026-07-25）

## 结论

本轮完成了两件关键工作：

1. 使用完全不接触 early-development 的 validation-only 规则，为种子 2043 冻结选择了初始化 274302 的 S2 checkpoint。
2. 22 参数可见度条件仿射校准器 V2 显著优于统一 Platt 和 isotonic：三档平均召回由 0.044 提高到 0.472，40/45 dB 达到 0.710/0.540，同时功率域 FPR 保持在 0.118 以下。

但 V2 未通过预注册总门槛。失败集中在 50 dB recall 只有 0.167，以及普通域 FPR 相对 V0 增加 0.026，超过允许的 0.02。当前仍不允许解冻最后 stage。

## 1. 稳定 2043 专家选择

选择只使用原 S2 train selection 中固定的 20% validation 身份，并在 40/45/50 dB 上评价。排序规则为最差档 AUROC、三档均值、平均 AUPR。

| 初始化 | 最差档 AUROC | 三档平均 AUROC | 三档平均 AUPR | 选择结果 |
|---:|---:|---:|---:|---|
| 274301 | 0.8522 | **0.9203** | **0.9194** | 未选 |
| 274302 | **0.8549** | 0.9056 | 0.8977 | **选中** |
| 274303 | 0.8441 | 0.9054 | 0.8916 | 未选 |

274302 虽然均值不是最高，但最差功率档最好，因此按预注册规则冻结。early-development 未参与选择，checkpoint hash 已记录。

## 2. 实验协议

- 2041/2042 使用上一阶段已经冻结的原 S2；2043 使用 validation-only 选出的 274302。
- 每种子从 train role 排除专家训练和普通路由身份，再选择 28 组合 × 30 对。
- 按组合拆成三个互斥角色：每组合 10 对 expert-scale、10 对 visibility-calibration、10 对 power-threshold。
- calibration、threshold、early-development 角色保持隔离；多功率视图按 identity 绑定。
- F2、H2、S2、C2 alpha=.80 全冻结。
- V2 仅 22 参数，不输入 JNR、功率档、raw RMS或数据生成元数据。
- locked test 未使用。

## 3. 三种校准器结果

| 方法 | 普通 recall | 普通 FPR | 三档平均 recall | 最差档 recall | 最大功率 FPR | oracle gap |
|---|---:|---:|---:|---:|---:|---:|
| V0 Uniform Platt | 0.6177 | 0.1786 | 0.0436 | 0.0000 | 0.0000 | 0.7336 |
| V1 Isotonic | 0.4385 | 0.1051 | 0.0136 | 0.0000 | 0.0000 | 0.6575 |
| V2 Visibility Affine | **0.7230** | 0.2048 | **0.4724** | **0.1668** | 0.1176 | **0.3123** |

V1 的单调非参数映射没有解决问题，说明只对 C2 标量分数做更灵活的单调变换仍不足。V2 的内部可见度统计确实提供了额外信息，但尚不足以完整恢复极弱可见度。

## 4. V2 逐功率结果

| 功率 | recall | hard-negative FPR | oracle recall | oracle gap |
|---:|---:|---:|---:|---:|
| 40 dB | 0.7101 | 0.1176 | 0.8521 | 0.1420 |
| 45 dB | 0.5403 | 0.0765 | 0.8013 | 0.2610 |
| 50 dB | 0.1668 | 0.0419 | 0.7008 | 0.5340 |

40 dB 已达到门槛，45 dB 平均召回也达到门槛；50 dB 是明确瓶颈。50 dB 的 oracle recall 仍有 0.701，证明排序并未消失，但校准器没有把这部分排序转化为统一阈值下的正判决。

## 5. 逐种子结果

| 种子 | 普通 recall/FPR | 三档平均 recall | 40 dB | 45 dB | 50 dB | oracle gap |
|---:|---:|---:|---:|---:|---:|---:|
| 2041 | 0.732/0.238 | 0.449 | 0.684 | 0.500 | 0.163 | 0.306 |
| 2042 | 0.671/0.176 | 0.387 | 0.670 | 0.440 | 0.050 | 0.450 |
| 2043 | 0.767/0.200 | **0.582** | 0.777 | 0.681 | 0.287 | **0.181** |

2043 在替换稳定专家后，从上一轮几乎不可用变为三种子中最好，验证了 checkpoint 稳定性修复的必要性。当前最差反而是 2042 的 50 dB recall 0.05。

## 6. threshold role 与 early-development 对比

V2 在独立 power-threshold role 上的最差档召回分别为：

- 2041：0.193，early 为 0.163；
- 2042：0.064，early 为 0.050；
- 2043：0.254，early 为 0.287。

两角色结果一致，说明 50 dB 失败不是 checkpoint selection 或阈值选择过拟合造成的开发落差，而是在未见身份上稳定存在的可辨识性不足。

普通域 FPR方面，threshold role 均被约束在 0.20以内；early-development 为 0.238/0.176/0.200。总均值 0.2048，仅比绝对上限高 0.0048，但相对 V0 增加 0.0262，仍超过预注册的 +0.02，不能事后放宽门槛。

## 7. 失败原因

### 7.1 不是专家排序不足

稳定专家下三档 oracle recall 分别为 0.852/0.801/0.701，50 dB 仍保留可用排序。V2 的任务不是重新学习类别，而是识别分数应该被抬高多少。

### 7.2 单视图统计无法充分确定可见度等级

当前十个特征来自单一样本：C2/H2/S2分数、分数差、其他标签logit、evidence map统计和stage2幅度统计。它们对40/45 dB有效，但50 dB时正样本和hard negative在这些统计上的条件分布仍重叠。校准器为了遵守普通域和功率域FPR约束，无法继续抬高50 dB分数。

### 7.3 直接仿射目标缺少可见度顺序监督

V2只由最终SawFM BCE监督。训练虽然包含40/45/50 dB，但没有显式要求同一identity满足“随着扫频分量增强，SawFM可见度递减”的有序关系。因此模型可以改善平均分类损失，却未必学习到可靠的可见度坐标。

### 7.4 继续扩大仿射网络风险较高

V2只有22参数已能显著改善40/45 dB；当前失败具有明确的50 dB定向性。简单增加隐藏层或参数量容易利用组合/种子相关捷径，并不能提供“学到了可见度”的机制证据。

## 8. 预注册门槛

通过：

- 三档功率FPR≤0.20；
- 普通recall未下降；
- V2优于V0/V1；
- checkpoint选择、身份隔离和locked-test审计通过。

未通过：

- 三档平均recall 0.472 < 0.50；
- 50 dB recall 0.167 < 0.35；
- 普通FPR相对V0增加0.026 > 0.02；
- 0/3种子通过全部单种子门槛；
- oracle gap 0.312 > 0.15。

总门槛失败，决策为 `keep_backbone_frozen_and_revise_visibility_decision`。

## 9. 下一步实验方案

下一阶段不扩宽校准器，而是显式学习可见度坐标：

1. 冻结稳定S2和C2。
2. 对同一identity的root/40/45/50 dB视图，训练一个8～16维visibility head。
3. 使用有序损失要求SawFM可见度满足 `root > 40 > 45 > 50`；同identity内容特征用stop-gradient保持接近。
4. visibility head只使用当前内部统计，不输入功率数值；真实功率档只作为训练阶段的顺序监督。
5. 校准器改为三段共享单调偏置：高/中/低可见度三个anchor，经soft assignment产生偏置；参数量控制在50以内。
6. 对照必须包含V0、当前V2、oracle真实功率分段校准和无顺序损失的同结构模型。
7. 先检查visibility排序准确率/同identity Kendall相关达到0.75，再评价统一阈值召回。
8. 最终门槛继续保持三档平均recall≥0.50、每档≥0.35、FPR约束和oracle gap≤0.15，不得降低。

只有显式可见度坐标通过，才解冻最后stage。

## 10. 产物

- 方案：`docs/plans/route_a_stable_expert_visibility_calibration_plan_20260725.md`
- 程序：`experiments/run_route_a_stable_expert_visibility_calibration.py`
- 汇总：`outputs/tapc/route_a_stable_expert_visibility_calibration/route_a_stable_expert_visibility_calibration_summary.json`
- 每种子保存checkpoint selection manifest、校准器checkpoint/history、阈值搜索、三变体预测/指标、身份与源checkpoint审计及receipts。
