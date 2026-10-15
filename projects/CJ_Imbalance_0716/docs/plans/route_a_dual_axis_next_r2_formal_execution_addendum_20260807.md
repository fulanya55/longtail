# 路线 A 双轴下一阶段 R2：正式执行补充预注册

本文件在 R2 正式五 seed 运行前冻结实现细节。R2 smoke 的任何数值不参与本文件的选择，smoke 只证明代码可执行。

## 1. 正式单元与顺序

四个训练单元固定为 `FB-VH, FB-VL, LT-VH, LT-VL`，正式 seeds 固定为 `260811–260815`。

先运行主臂：

- `M0_B0_Focal`；
- `MF_B0_DB`；
- `MV_B0_RawSaw`；
- `MFV_B0_DB_RawSaw`。

每个 cell/seed 只训练两个共享结构的 B0 core（Focal 与 DB）和一个 RawSaw expert。MV 必须复用对应 M0 的 core checkpoint，MFV 必须复用对应 MF 的 core checkpoint；MV 与 MFV 必须复用同一 RawSaw checkpoint。这样组合臂不会通过额外 backbone 训练预算获得优势。

若 MF 或 MV 的单轴硬门失败，主机制停止，不运行以正主张为前提的 MEC 和信息破坏消融；若两个单轴均通过，则自动运行 MEC、MTS、phase-only、envelope-only 和 time-shuffled 控制。该顺序是预注册的节约计算策略，不允许依据候选是否“接近门槛”改变。

## 2. 初始化、批次和优化

- 同一 seed 的四个 cell 和 Focal/DB core 从同一 B0 初始 state hash 开始；
- 同一 cell 内 Focal/DB 使用相同 identity 顺序、batch size、epoch 上限和更新步数；
- epochs 36，patience 8，batch 64，gradient clip 5；
- AdamW：backbone LR `5e-5`，classifier LR `5e-4`，weight decay `1e-4`；
- B0 core 的 Focal 与 DB 都沿用当前 B0 reduction：五个非 SawFM 标签 elementwise mean，加 `0.2 × SawFM` elementwise mean；
- DB 使用当前 `DistributionBalancedLoss` 默认参数，positive counts 只从相应 optimization manifest 计算；
- core checkpoint 仅以相应训练视图 early-role macro-AUPR 选择，平分时才参考 SawFM 排序指标。

RawSaw 使用 `raw` 模式、hidden 16、kernels 7/31/63、top-k 16、dropout 0.1、5,521 参数。为防止 RawSaw 自身被 FB/LT 正负比例混淆，正式训练每个 epoch 使用只基于 SawFM 标签的 50/50 positive-negative 加权采样，共 2,400 次抽样；该 sampler 在四个 cell 中固定，BCE、LR `5e-4`、weight decay `1e-4`、epochs 36、patience 8。RawSaw checkpoint 只用相应训练视图 early-role SawFM AUPR 选择；VL 平分时参考 weak-vs-negative AUROC。

## 3. 校准和阈值

为了让四个训练 cell 使用同一部署决策协议，每个冻结模型的 calibration role 和 threshold role 都使用该角色内配对的 VH+VL 两视图；同一 receiver identity 的两个视图只存在于同一个角色中，因此不会造成跨角色泄漏。

- calibration：逐标签 Platt，`C=1e6, lbfgs, max_iter=1000`；
- threshold：`0.05:0.01:0.95`；
- objective：负样本 FPR `<=0.20` 下最大化 weak recall；
- 排序指标使用未校准 logits；校准和阈值只用于 recall/FPR/F1/Hamming；
- evaluation 对 VH、VL 分别报告，同时报告 identity-paired combined 统计。

MTS 直接复用 M0 logits，只运行相同的 calibration/threshold 流程；因此 MTS 不允许出现排序增量。

## 4. 正式统计和推进门

- `MF-M0`：每 seed 在 LT-VH/LT-VL 上的 macro-AUPR 差取平均，五 seed 均值 `>=+0.005` 且至少 4/5 为正；FB-VH/FB-VL 平均降幅不超过 0.005；
- `MV-M0`：每 seed 在 FB-VL/LT-VL 上平均，SawFM weak AUROC `>=+0.020`、weak recall `>=+0.050`、候选 FPR `<=0.20`，strong recall 差 `>=-0.01`；
- `MFV-best(MF,MV)`：LT-VL 上 macro-AUPR `>=+0.005` 或 weak recall `>=+0.020`，receiver-identity paired bootstrap 95% CI 下界 `>0`；
- 其余五标签任一 AUPR 差不得低于 `-0.01`。

先报告逐 seed 差和 seed signs，再执行 receiver-level paired bootstrap。不得只报告跨 seed 平均值。

## 5. 证据边界

R2 正式运行仍是 development evidence `[A]`，不使用任何 confirmation/test。只有 R2 硬门通过、全部模型/校准器/阈值冻结并创建全新 `dual-axis-sawfm-confirmation-v1` 后，才允许一次性 confirmation。

历史 locked test、test-v2 和 sealed test-v3 在 R2 必须继续为 `used=false`。
