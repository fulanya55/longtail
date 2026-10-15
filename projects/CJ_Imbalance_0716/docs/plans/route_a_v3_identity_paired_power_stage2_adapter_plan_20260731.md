# Route A development-v3 身份配对功率视图与 stage2 adapter 计划

## 1. 单一机制假设

final-temporal 已含可见度信息，但它对弱证据的控制能力不足。使用同一 receiver_noise_seed 的
多功率视图，在 stage2 仅训练低秩 adapter，使低功率视图的目标标签证据接近同身份高可见度
视图，同时使用组合相同但目标缺失的 hard negatives 约束，可增强弱分量表征且避免全主干扩容。

## 2. 数据生成与身份边界

- 基础 seed 固定为 development-v3 的 `24000000`；只生成 train 身份视图，禁止生成或访问
  test-v3 身份的功率视图。
- 九挡为 `10:5:50 dB`，FMB/SawFM/SinFM 固定 30 dB，MFT/VFT/ISRJ 同步扫描。
- 输出根目录：`C:\LJQ_code\干扰数据\复合干扰数据0731_v3\power_sweep_9level_v3`。
- 所有挡位必须逐 identity 核对 receiver_noise_seed 和六标签完全一致；角色继承现有
  development-v3 role_indices，禁止重新随机拆分视图。
- 保存 manifest、逐文件 SHA-256、identity 交集审计和生成 receipt。

## 3. 后续模型对照

完成数据审计后再冻结详细模型预注册。首轮仅允许 stage2 rank-8 adapter、归一化 evidence head
和固定三挡训练视图；必须设置同参数量 same-view augmentation 对照以及不使用配对损失的
adapter 对照。F2 其他层、RawSaw、校准与阈值保持冻结。

进入正式模型实验前的硬门槛：九挡各 25200 个唯一 identity；挡位间 identity 和标签逐一匹配；
与 development-v3 train 一一对应；与 test-v3 identity 交集为零；test-v3 `used=false`。

## 4. 首轮冻结模型预注册

审计门槛通过后，首轮固定如下，不做超参数搜索：

- 输入：F2 `stage2_real/stage2_imag` 拼接为 64 通道，时间方向平均池化 4 倍。
- adapter：GroupNorm 后 `Conv1d(64,8,1) -> GELU -> Conv1d(8,64,1)`，up 层零初始化，
  残差尺度上限 0.20。
- evidence：`Conv1d(64,32,1)`、depthwise `Conv1d(k=15)`、三标签 1x1 evidence map；
  每标签时间归一化后使用 top-k=16 均值和局部最大值形成 logit。
- 目标标签固定为 MFT/VFT/ISRJ；F2 主干完全冻结。
- 训练视图固定为 10/30/50 dB，validation 使用未参与训练的 20 dB；early 报告使用
  15/20/25/35/40/45 dB。角色 identity 严格继承 development-v3。
- epoch=30、patience=8、batch=64、AdamW、lr=1e-3、weight_decay=1e-4。
- 分类损失为三个训练挡位的逐标签 pos-weight BCE。
- 配对 score Smooth-L1 权重 0.25，label evidence-map cosine 权重 0.10；只在对应标签正样本上计算。

三个完全同结构同参数量变体：

- P0 SameView：50 dB 视图与固定 2% stage2 特征噪声增强视图做一致性；
- P1 NoPair：只做三挡分类，不使用一致性损失；
- P2 IdentityPair：同 identity 的 10/30 dB student 对齐 50 dB teacher。

模型选择只看 validation-20dB target macro AUPR。阈值只用 threshold-20dB 负样本按
FPR<=0.20 选择，early 不参与模型、损失或阈值选择。主要机制门槛：P2 相对 P0 的低功率
weak-vs-negative AUROC `>=+0.015`、弱召回 `>=+0.030`、macro AUPR `>=+0.010`；相对 P1
AUROC `>=+0.010`；P2 低功率平均 FPR `<=0.20`；至少 4/5 种子主要 AUROC 增益为正。
若失败，停止当前 complete-invariance 配对目标，分析是否发生低/高功率判别信息冲突。

