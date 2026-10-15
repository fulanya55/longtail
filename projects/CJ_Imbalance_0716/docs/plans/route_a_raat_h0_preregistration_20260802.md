# Route A RAAT-H0 预注册：从局部物理可辨识性到安全集成

日期：2026-08-02  
版本：`RAAT-H0/v1`  
状态：**预注册冻结；尚无数据、模型、训练或指标。**

## 1. 问题、对象与允许输入

在新的 train-only 六标签源上，分别检验 MFT（index 3）、VFT（index 4）和 ISRJ（index 5）的 RAAT 局部证据，目标是 weak-positive vs composition-matched-negative 的无阈值排序。部署时每个专家只可读：

- 接收复 I/Q `x[n]`，长度 `3000`、采样率 `10 MHz`；
- 已知发射 LFM 参考 `s[n]=exp(j*pi*k*(n/fs)^2)`，有效脉冲 `100 us`、带宽 `5 MHz`、PRI `300 us`；
- 固定于本文件的数值网格与算术操作。

严禁输入：真实/预测 component power、relative power、宽度、重复数、Doppler、delay、随机 seed、label query、position token、全局特征、全局 logits、预测标签、阈值、校准值、pair/delta 或任何训练期可见性真值。标签仅用于二元 BCE 的监督和离线的 weak/strong 评估分层，绝不进入特征。

## 2. 唯一候选：三种互异的 RAAT 观测量

三者都从复交叉相关 `a(tau,nu)=sum_n x[n] conj(s[n-tau]) exp(-j*2*pi*nu*n/fs)` 出发，但不共享学习特征或预测头；每个目标都有独立训练、独立控制和独立门。

| 专家 | 固定观测量 | 张量 | 物理预期 | 排除的旧机制 |
|---|---|---|---|---|
| `RAAT-MFT` | **延迟多重性相干图**：在 `tau=400..1000` 的 601 点 ROI 内计算 `nu={-12,-10.5,...,12} kHz` 共 17 个 bin 的交叉模糊幅相；按相邻 25 点无重叠池化为 24 个 delay bin | `2 x 24 x 17`：每 bin 的 fit-only 标准化 `log(1+|a|/median|a|)` 与 `sin(angle(a))` | 多个移位参考副本应造成可区分的多延迟局部结构，而不是单个峰数 | 不是旧的峰计数/全局池化摘要；不使用 task vector 或 logit residual |
| `RAAT-VFT` | **去调频滞后相干图**：先由 `argmax_tau |a(tau,0)|` 作盲参考对齐；取对齐后前 1000 点 `z[n]=x[n+tau_hat]conj(s[n])`，计算 `q(lag,nu)=sum z[n+lag]conj(z[n])exp(-j2*pi*nu*n/fs)`；`lag=0..920` 每 40 点池化为 24 bin，仍用上述 17 个频 bin | `2 x 24 x 17`：`log(1+|q|/median|q|)` 与 `sin(angle(q))` | 若 VFT 的参考对齐残余相位确有稳定局部证据，其 lag-frequency 相干应优于错误参考；若分辨率不足，该 H0 应失败 | 不是 raw/STFT/IFREQ、不是 32x32 STFT concat、不是参数回归或 predicted-width residual |
| `RAAT-ISRJ` | **分段延迟碎片图**：将 3000 点 PRI 划为 12 个相邻 250 点窗口；每段仅在 `nu=0` 计算参考交叉相关，在相同 24 个 delay bin 内求归一化能量与相邻段的复相干 | `2 x 12 x 24`：段内 `log(1+energy/median)` 与相邻段 `real(coherence)` | 间断采样的回放在参考对齐延迟上应表现为局部片段占据/相干断续，而非需真宽度的延迟—尺度峰 | 不使用 sample width、replica、周期/lag 投票、scale-Hough、固定形态残差或旧 310-D 周期性分数 |

边界：`argmax_tau` 是在 I/Q 和已知参考上的确定性部署计算，不是标签查询或真值 delay；其取值及中间张量不得保存为供训练选择的元数据。所有 0 除以均使用 `1e-6` 下限；所有池化边界如上固定。

## 3. 信息破坏控制与精确匹配

每个目标分别训练两臂，数据、样本索引、标签、batch 顺序、初始化种子、优化器、训练步数和分类器完全一致：

- `C_RaatAligned`：使用上述正斜率 `+k` LFM 参考；
- `B_WrongReference`：将参考替换为**反斜率 `-k` 的 LFM**，其他长度、能量、相位初值、delay ROI、Doppler 网格、归一化、张量尺寸、网络、总步数完全相同。

这是信息内容破坏而非纯 permutation：错误参考仍含等长等能 LFM，但不与实际发射 chirp 相位匹配。验证时必须报告：(i) 两臂每张图的非零比例、均值/方差和 SHA-256 不相同；(ii) 相同样本的 map L2 差异严格大于 0；(iii) 参数计数、初始 state_dict、batch-id 列表、optimizer step 数、BCE 标签与 fit-only 标准化角色完全相等。任一项失败，本轮无效，不可解释为物理增益。

不额外提供“同参数 MLP”第三臂：错误参考臂已经保留相同形状、同 CNN、同初始化及同训练预算，同时真实改变物理对齐信息；它是 H0 的主要因果控制。H2 才加入无可见性同容量网络。

## 4. 固定网络、损失和超参数

每个目标/每个臂独立实例化 `RAATLocalMapNet`，不共享 encoder、不进行多任务训练。

```text
input (2,H,W)
Conv2d(2,16,3,padding=1) -> GroupNorm(4,16) -> GELU
Conv2d(16,16,3,stride=2,padding=1) -> GroupNorm(4,16) -> GELU
Conv2d(16,32,3,padding=1) -> GroupNorm(4,32) -> GELU
AdaptiveAvgPool2d(1) -> Flatten -> Linear(32,1)
```

- 参数量：`7,425`（不随目标或臂改变；实现冻结时必须实测核对）。
- 损失：`BCEWithLogitsLoss(reduction="mean")`，无 focal、类别权重、visibility 权重、pair ranking、辅助损失、校准损失。
- 优化器：AdamW，`lr=3e-4`，`weight_decay=1e-4`，`betas=(0.9,0.999)`，无 scheduler、无 AMP。
- batch size `64`，epochs `30`，不 early-stop、不根据 validation 选 checkpoint；报告第 30 epoch state。
- 训练种子固定为 `[2041, 2042, 2043, 2044, 2045]`；Torch deterministic algorithms 开启，cuDNN benchmark 关闭。

## 5. 新数据、身份与角色（尚未生成）

拟生成一次且只一次的新 train-only 源：`route_a_raat_h0_trainonly_v1`。其身份范围永久预留为 `receiver_noise_seed=978677743..978682782`（5040 个唯一身份）；此前最近 H0h 使用范围为 `968677743..968682782`。生成前后必须对所有 live historical train sources、H0e--H0h 和 test-v3 seal 做全量交集审计。

- 固定 63 个合法六标签组合，每组合 80 样本，共 5040 样本；沿用 0716 的物理生成器、标签顺序、JNR 规则和 `receiver_noise_seed` 协议，不能为了某个目标提高功率、改变 VFT Doppler、增加脉冲或修改 ISRJ 宽度/副本范围。
- 只在生成完成后按 `SHA256("RAAT-H0/v1|receiver_noise_seed") mod 10` 固定角色：`0..5=fit`、`6..7=validation`、`8..9=early`。validation 不用于 early stop 或超参选择，只用于输入/对照完整性；formal H0 的所有主指标在 early 角色一次计算。
- 对每一目标和每一角色，仅保留“target weak positive”与“target negative”；用其他五标签组合、总标签数、以及**由 I/Q 计算的**全 PRI 接收能量五分位作确定性分层，按每个 strata 的较小类数下采样，形成 composition-matched hard-negative view。不得用真实 component power/parameter/seed 进行匹配或输入。
- 每个角色若任意目标的 matched view 含少于 160 weak positives 或 160 negatives，则停止；不可扩大范围、补生成或调规则，只报告不可检验。

## 6. H0 指标、统计与晋级门

无 calibration、无 threshold、无 fusion。主端点为每个目标的 `weak-positive vs matched-negative AUROC`，候选减控制。次端点：弱正 AUPR、全正 AUPR、strong-positive vs matched-negative AUROC。强正端点仅作不伤害保护。

对每个目标，使用 10,000 次 paired hierarchical bootstrap：先有放回抽 5 个训练种子，再在每个种子内有放回抽 early 样本；两臂在相同 sample id 上配对。报告均值、95% percentile CI、五个逐种子 delta 和正向种子数。禁止挑选分层、种子、epoch 或指标。

单目标通过当且仅当全部成立：

1. weak-AUROC 平均 delta `>= +0.015`；
2. paired 95% CI 下界 `> 0`；
3. 至少 `4/5` 个种子 delta 为正；
4. 每个种子的 strong-positive AUROC delta `>= -0.010`；
5. 破坏控制和身份/信息审计全部通过。

只有至少两个目标通过 H0，才允许其各自进入 H1；零或一个目标通过时，RAAT 不进入融合、不开 H2、不触及 test-v3。VFT 失败必须作为“在该观测协议下未发现可部署残余相干证据”报告，而不是再搜 STFT/损失超参。

## 7. 后续阶段的预注册分支

### H1：安全专家接回（仅 H0 通过标签）

冻结 RawSaw 和 H0 的 epoch-30 expert；全局 F2/H2 路径为旁路。每个通过专家只允许影响自己的目标 logit，其他五类 logits/predictions 必须 bitwise 一致。训练期只用目标 BCE，禁止重训 RawSaw、禁止注意力/门控/阈值微调。输出固定裁剪残差 `delta=0.15*tanh(expert_logit)`；在 fit 角色上仅以确定性 z-score 标准化专家 score，记录其数值；不拟合 calibration。H1 需在 fresh H1 source 的 5 seeds 通过：目标 AUPR 与弱 AUROC CI 下界 >0、FPR CI 上界 `<=+0.010`、strong recall CI 下界 `>=-0.010`、非目标 bitwise 保护和 F2/H2 映射保护。失败则永久停止该标签 H1。

### H2：六标签集成（仅至少两个 H1 通过）

新鲜 six-label dev-v2 source，RawSaw 原样旁路。比较 `C0`（F2+RawSaw）、`C1`（C0+安全 RAAT）、`C2`（C1 去掉对应物理图的错误参考控制）及同参数量无可见性 baseline，并补 BCE/Focal/ASL/DB。所有 calibration/threshold 只可在各 seed 的 fit/validation 角色一次拟合，early 角色仅评估；五 seeds 后冻结。晋级要求：至少两个目标 AUPR、弱 recall、weak AUROC 的 CI 下界 >0；macro-AUPR 不降；SawFM 绝对保持；每个标签 FPR 上界 `<=+0.010`；非目标保护、对照与同容量控制均不反证；并报告消融和最近多标签长尾基线。

### H3/H4：鲁棒性与一次性 test-v3

H3 仅对 H2 冻结物做可预先列出的 JNR、成分数、未见组合、参数外推和 AR(1) 噪声验证；不得改动实现、权重、校准、阈值或融合。只有 H3 全过，才生成并一次性打开 test-v3，且先比对 manifest、代码、checkpoint、calibration 和 threshold 哈希。任何阶段失败都禁止以调参形式复活本候选。

## 8. 永久禁止与轮次产物

永久禁止：FC/TRB/FullAlpha/task vector，hard/soft visibility reweighting，power invariance/ordinal/oracle calibration，任何 query/token/position 选择，cross-head logits、predicted-width/scale-Hough，固定形态残差/权重、pair ranking/32x32 STFT concat，pair-logit margin、delta-prototype binding 和当前 unified target-specific complex branch。

每轮报告必须含：当前阶段与证据边界、样本/身份、问题与反证、文献矩阵、可执行输入、网络/损失/超参、对照公平性、smoke/formal、种子与 CI、门状态、代码/数据/role/checkpoint 哈希、`locked_test_used`、下一唯一动作。未完成项必须写未完成，不能由历史分数替代。
