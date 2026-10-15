# TRB-PTV 文献审计、机制定义与 fresh-development 预注册

日期：2026-08-01  
状态：**预注册冻结稿；尚未生成 development-v14 数据，尚未实现或运行 TRB-PTV**  
内部代号：Target-Risk-Balanced Paired Task Vector（TRB-PTV）

## 1. 前置阶段 A 的资格结论与边界

本预注册的唯一历史依据是只读审计
`docs/reports/route_a_fullalpha_effect_heterogeneity_pretrb_audit_20260801.md`。
它读取 v4/v5/v6/v12 保存的 predictions，不重训、不做 checkpoint 前向，也不接触任何测试角色。

对 FullAlpha1p5 相对 C6 的 target weak recall，定义 SawFM 差异中的差异：

```text
D_t = [Delta(Full,C6) | SawFM-present] - [Delta(Full,C6) | SawFM-absent]
```

采用在写入本机制前冻结的严格判据：四个历史轮次的每一个 5-seed 配对都必须有同一严格符号，且至少两个目标标签满足。MFT 和 VFT 均为 20/20 严格负；ISRJ 不稳定。因此阶段 A 的决定为
`ELIGIBLE_FOR_TRB_PTV_PREREGISTRATION`。

这个结果仅说明“Full 的效应随 SawFM 共现状态而异，且 MFT/VFT 可审计”；**它不说明相对功率、SNR/JSR 或因果可见性机制**，因为保存预测不含这些字段。它也不构成 TRB-PTV 的性能结论。

历史固定事实不得被重新解释或调参：FullAlpha1p5 为
`theta_C6 + 1.5(theta_M5-theta_C6)`；shared-only 有害、classifier-only 有用、两块共同最佳；FC-PTV 已永久停止，禁止改变其半径、系数、激活门或重试；LSCP/Power-SSL 负结果不得重包装。

## 2. 原始文献差异矩阵（2026-08-01）

检索只使用原始论文/正式 DOI 页面。结论是“未发现完全同构实现”，并不等于证明全球不存在先例。

|工作|已覆盖的核心方法|与 TRB-PTV 的重叠|不能主张的内容|TRB-PTV 的窄差异|
|---|---|---|---|---|
|[F] Task Arithmetic, Ilharco et al. (2023), [arXiv:2212.04089](https://arxiv.org/abs/2212.04089)|由共同预训练锚点到任务端点的参数差分，可加、减、缩放|`d=M5-C6` 与端点算术|首次提出 task vector 或 alpha 合并|仅一个严格 loss-deletion 配对向量；q 由目标弱分量风险边界而非性能搜索确定|
|[F] TIES-Merging, Yadav et al. (2023), [arXiv:2306.01708](https://arxiv.org/abs/2306.01708)|多任务向量的 trim/elect-sign/disjoint merge 以减干扰|参数空间的冲突处理|首次解决 merging interference|不做多向量符号筛选；只在两个可审计块上，以 C6 的目标弱损失为安全锚|
|[F] AdaMerging, Yang et al. (2023), [arXiv:2310.02575](https://arxiv.org/abs/2310.02575)|任务/层级合并系数的自适应优化|学习/优化 merge coefficient|首次逐块或自适应系数|禁止 validation/test 目标、熵或正式性能选择；q 是“最接近 Full1.5 且三目标 fit-only 风险不高于 C6”的固定约束解|
|[F] Multi-Task Model Fusion via Adaptive Merging (2025), [DOI](https://doi.org/10.1109/icassp49660.2025.10887718)|近期多任务自适应融合|再次说明 generic adaptive fusion 已拥挤|通用自适应合并创新|本项目只检验 label-deletion paired vector 的 target-risk boundary，不宣传通用融合|
|[A] FC-PTV 已停止机制（本仓库 v12 预注册与结果）|C6 logits 的函数距离 trust region、两块投影|同样以 `(q_shared,q_classifier)` 修改 Full1.5|把“trust region/功能距离”改名成新方法|TRB 使用目标弱分量的实际 fit 风险约束，不复用 FC 的 MAD 半径、函数 Gram、半径或激活逻辑|

证据标记：`[F]` 为已核验的一手论文页面；`[A]` 为本仓库审计事实。文献审计的允许表述仅为：

> 我们检验一个项目特定、可证伪的后验配对模型合并约束：在严格的 SawFM loss-deletion 端点之间，以 C6 fit-only 的 MFT/VFT/ISRJ 弱分量损失作为不可恶化的风险边界，选择与已冻结 Full1.5 最近的 shared/classifier 两块系数。

不得表述为“首次 task-vector 安全约束”“首次 adaptive merging”或“已证明文献新颖性”。

## 3. 三个候选与唯一选择

|候选|定义|选择结论|
|---|---|---|
|A. SawFM 条件推理路由|按预测/已知 SawFM 状态选择不同 q 或阈值|拒绝：容易把共现条件与测试时路由混淆，且需要新增状态估计与阈值选择，不是当前配对向量问题|
|B. TRB-PTV|在 target weak fit loss 不超过 C6 的约束下，求最接近 Full1.5 的两块 q|**唯一选中**：直接检验历史发现的 MFT/VFT 风险异质性，安全条件、终点和数据角色均可冻结|
|C. Layer-wise adaptive q|给每层或每块学习独立系数|拒绝：与 AdaMerging 过近，自由度过多，且会把 alpha 搜索重新引入本项目|

不得在正式数据结果后换用 A/C，或把 q 的界、风险约束、solver、标签权重视为可调超参数。

## 4. TRB-PTV 数学定义（唯一冻结版本）

对每个正式 seed：

```text
theta0 = theta_C6
d      = theta_M5 - theta_C6 = d_shared + d_classifier
theta(q) = theta0 + q_shared*d_shared + q_classifier*d_classifier
q0 = (1.5, 1.5)
```

端点必须共享初始化、long-tail manifest、fit 样本、batch 顺序、训练预算、early-stop、可训练作用域和数值契约。M5 保留全部 SawFM 共现样本，只屏蔽 SawFM 输出的损失。非零差分只允许在冻结的 `temporal_backbone.0.*` 与 `classifier.*`；否则该 seed invalid。

目标集合固定为 `T={MFT,VFT,ISRJ}`。令 `A_fit` 为该 seed 新 development 身份的 fit 角色；`W_t` 为其中标签 t 为正且已有协议给出的 weak mask 为 1 的样本。定义带固定标签权重的弱分量二元交叉熵：

```text
R_t(q) = mean_{x in W_t} BCEWithLogits(z_t(x; theta(q)), 1)
```

所有标签权重固定为 1；不重加权、不按 SawFM 状态分别求 q、不读取 validation/early/calibration/threshold/test。TRB-PTV 仅解：

```text
q* = argmin_q ||q-q0||_2^2
subject to 0 <= q_shared <= q_classifier <= 1.5
           R_t(q) <= R_t(0,0)  for every t in {MFT,VFT,ISRJ}
```

`(0,0)` 即 C6，故理论上是可行锚点。求解器固定为 float64 `scipy.optimize.minimize(method="SLSQP")`，初值 q0，`maxiter=200`、`ftol=1e-12`、无随机重启；仅在 `A_fit` 完成前向。正式评价前必须以独立的确定性 double-precision 前向复核每条约束，容差固定为 `1e-7`。任意非有限量、非成功状态、边界/约束残差超过容差，均为 invalid；不得缩放、回退、改初值或更换 solver。

**激活门**：至少 4/5 seed 必须满足 `||q*-(1.5,1.5)||_2 >= 0.05`，且两系数和所有约束复核均有限；否则停止 TRB-PTV。0.05 是在观测正式指标前固定的最小可辨别二维系数位移，不是搜索网格。

## 5. 实现前冻结与新身份

在实现、生成数据、训练或读取任何新数据前，必须完成并写入 hash：本文件、实现 manifest、solver 参数、标签/weak-mask 定义、label weights、bootstrap 索引规则、变体名单、停止门和数据生成 config。

预留的新 train-only identity 为 `development-v14`（只预留，尚未生成）：

```text
sample_seed:          [144000001, 144025200]
receiver_noise_seed:  [342677743, 342702942]
expected samples:     25200
expected unique receiver_noise_seed: 25200
```

生成前的只读 preflight 必须对所有既有 manifest/receipt 做 receiver_noise_seed 与 sample_seed 交集审计；禁止打开任何 test 文件。生成后必须保存配置、文件 SHA256、大小、split/manifest SHA256、样本/identity 清单和 receipt，并证明 `test_files_opened=false`、所有 `locked_test_used/test_v2_used/test_v3_used/test_v4...=false`。同一 identity 上发生实现、数据或选择角色变更即作废，不能重用。

角色固定：fit 用于训练端点和 q；validation 仅 early-stop；calibration 仅温度；threshold 仅阈值；early 是唯一 development 评价；所有 test 禁止。receiver_noise_seed 是身份，九个 power view 绝不作独立样本。

## 6. 正式变体、公平性与最小实现测试

每个 seed 的正式变体固定为：`C6_Joint`、`M5_GradientIsolated`、`FullAlpha1p5`、`TRB_PTV`、`ParameterNormOnly`、`ClassifierOnlyAlpha2p0`、`ModelAverage0p5`、`CAGrad0p4TwoTask`、`OrthogonalTaskSubspaceRiskMatched`。后者以五个 other-label deletion 端点构造正交子空间，并以 fit-only 的目标风险预算匹配 TRB；无法有限匹配即 invalid，不能替换控制。

所有变体共享初始化、split、batch 顺序、端点训练、epoch/patience、学习率、训练 scope、clip、early-stop、校准、阈值和 seeds `[2039,2040,2041,2042,2043]`。禁止 alpha/q 网格，禁止从 early、calibration、threshold 或 test 选择 q。

代码必须先通过合成 smoke（仅验证 shape、block mask、约束、非有限拒绝、边界、q 激活审计、identity guard），再在新 identity 进行一次正式 5-seed 运行。smoke 不构成方法结果。

## 7. 强制通过门与停止规则

TRB 必须同时通过以下四组，任一失败立即 `stop_trb_ptv`：

1. **相对 C6（10 项）**：macro-AUPR delta >= -0.005；macro-F1 delta >= +0.010；target mean weak recall delta >= +0.025；MFT/VFT/ISRJ mean weak recall 均 >=0；target FPR absolute <=0.20；至少 4/5 target-weak delta 正；SawFM-present macro-F1 >0；SawFM-absent macro-F1 >=-0.005；三标签 AUPR delta 均 >=-0.010；5/5 输出/参数/q 全有限。
2. **相对 FullAlpha1p5（8 项）**：最差标签弱召回改善 >=+0.005；target mean weak recall 不低于 0；macro-F1/AUPR 均不低于 -0.003；target FPR delta <=+0.005；至少一项主要弱指标 paired bootstrap 95% CI 下界 >0；跨 seed target-weak 效应标准差严格更低；所有结果有限。
3. **三标签和机制特异性**：三标签 mean >0、各至少 4/5 非负、固定 10,000 次 paired-seed bootstrap 的 family-wise max-stat 95% simultaneous lower bounds 均 >0；相对 ParameterNorm、ClassifierOnly 与 Orthogonal 风险匹配控制均须满足主要弱指标 >=+0.005 且相应 AUPR/F1 非劣（-0.003）。
4. **TRB 特有机制门**：5/5 q 可行且风险约束复核通过；至少 4/5 激活门通过；TRB 相对 Full、ParameterNorm 和任一同 q 的普通缩放不数值等价；不允许只靠 classifier-only（`q_shared>0` 5/5）。

报告固定包括逐 seed q、每个 `R_t(q)-R_t(0)`、KKT/残差、块范数、macro AUPR/F1、target FPR、每标签 AUPR/weak recall、SawFM 分层、基数/组合、最差标签与最差 seed、cost、所有失败和 `locked_test_used`。Bootstrap 的 10,000 个重采样索引由固定 seed `20260801` 在正式运行前保存。

## 8. 后续阶段与一次性外部验证

仅当本轮全部通过，才能在**另一个**新 identity 上一次性比较完整 canonical baselines（PCGrad、MGDA、CAGrad、GradNorm、loss weighting、M5、task arithmetic、model average、TRB）。仅当该轮也通过，才预注册鲁棒性（SawFM 共现率、相对功率、SNR/JSR、长尾、未见组合、第二骨干）；至少 80% 条件为正且最差条件无实质 AUPR 退化，才冻结代码、模型、校准、阈值和统计方案并执行一次外部测试。

历史 locked test、test-v2、sealed test-v3 均不得访问。外部测试只允许一次；其后禁止调参、选择、再校准或重跑。TRB 是 FC 失败后的机制循环 2；若该机制失败，不得通过微调 q 的界、0.05 激活阈值、solver 或风险约束另起循环。
