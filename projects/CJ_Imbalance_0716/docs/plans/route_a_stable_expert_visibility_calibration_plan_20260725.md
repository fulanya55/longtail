# 路线 A：稳定专家冻结选择与可见度条件校准方案（2026-07-25）

## 1. 上一轮结论

上一轮证明了两个独立事实：

1. 原种子 2043 的低性能来自初始化/优化偶然性。三个新 S2 初始化的四档平均 AUROC 为 0.9082～0.9249，明显高于原 checkpoint 的 0.6080。
2. 冻结旧 C2 后，一维 Platt 和单一统一阈值无法同时满足普通域 FPR 与 40/45/50 dB 弱正召回。2041/2042 的分档 oracle 阈值可以达到较高召回，说明主要缺陷是可见度引起的分数平移，而非排序不足。

本轮先用严格 validation 规则选择稳定 2043 专家，再验证不输入真实 JNR/功率档的可见度条件校准。F2、H2、S2 和 alpha=.80 全部冻结，不解冻主干，不使用 locked test。

## 2. 2043 checkpoint 冻结选择

- 候选：初始化 274301、274302、274303。
- 选择身份：原 S2 train selection 中固定的 20% validation 身份。
- 评价功率：40/45/50 dB。
- 选择顺序：三档最差 AUROC最高 → 三档平均 AUROC最高 → 三档平均 AUPR最高 → 初始化编号较小。
- early-development、calibration、threshold-selection 和 locked test 均不参与选择。
- 2041/2042 保留原已通过专家；2043 使用冻结选择后的新 checkpoint。

## 3. 校准身份角色

每种子从合法 train role 中排除原 S2 训练身份和上一轮普通路由身份，再按 28 个 SawFM+扫频组合抽取每组合 30 对：

- 前 10 对：`expert_scale`，拟合新专家的一维尺度；
- 中 10 对：`visibility_calibration`，训练校准器；
- 后 10 对：`power_threshold`，选择统一阈值。

三个角色按组合分层并以 identity 拆分，身份互斥。每个身份使用 root 普通视图以及 40/45/50 dB 视图，但多功率视图不增加身份样本量。原 base calibration role可加入校准器 fit；原 threshold-selection role只作为普通域 FPR约束；early-development只作最终评价。

## 4. 冻结 C2 与可见度特征

`C2 = 0.20 * H2 SawFM logit + 0.80 * scaled S2 logit`

校准器不读取真实功率档、JNR、raw RMS或任何数据生成元数据，只允许使用模型内部可部署统计：

1. C2 logit；
2. H2 SawFM logit；
3. scaled S2 logit；
4. H2-S2 差值绝对值；
5. H2 三个扫频标签最大 logit；
6. H2 FMB/SinFM 最大 logit；
7. S2 normalized evidence map top-k 均值；
8. evidence map 最大值；
9. stage2 特征绝对值均值；
10. stage2 时间维标准差均值。

## 5. 对照校准器

- `V0_UniformPlatt`：只使用 C2 logit的一维 Platt。
- `V1_Isotonic`：只使用 C2 logit的单调 isotonic regression。
- `V2_VisibilityAffine`：10 维特征标准化后，两个线性层分别输出正尺度与偏置：`z'=softplus(a(x))*z_C2+b(x)`，共 22 个参数。

V2 使用 BCE、AdamW `1e-2`、weight decay `1e-3`、最多 300 epoch。模型选择只看独立 power-threshold/普通 threshold role，在所有域 FPR≤0.20 的条件下最大化 40/45/50 dB 平均召回；并列时优先最差档召回、普通 F1和更小参数范数。

所有方法最终只输出一个统一阈值，不允许按功率档选择阈值。

## 6. 最终评价与门槛

early-development普通域及其40/45/50 dB同身份匹配对只作最终评价。

V2 晋级门槛：

1. 三档平均 recall≥0.50；
2. 每档 recall≥0.35；
3. 每档 hard-negative FPR≤0.20；
4. 普通 SawFM FPR相对V0增量≤0.02；
5. 普通 SawFM recall相对V0下降≤0.05；
6. 至少2/3数据种子满足上述单种子约束；
7. 相对分档 oracle recall的平均差距≤0.15；
8. V2 至少不劣于 V0/V1 中的最佳三档平均 recall；
9. checkpoint选择、校准和阈值身份交叉为0，locked test未使用。

全部通过后，才进入最后 stage 小学习率联合微调；否则继续保持主干冻结，并根据失败归因决定是增加可见度统计还是采用显式拒识/unknown机制。

## 7. 产物

保存 checkpoint selection manifest、三候选 validation预测及指标、选择后hash；三个校准角色身份与hash；校准器参数、训练历史、阈值搜索；普通/三档功率预测、逐种子指标、oracle差距、receipts和总汇总。
