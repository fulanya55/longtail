# Route A 五类频次长尾 V2 自动验证 Goal Prompt

你是 `E:\LJQ_code\CJ_Imbalance_0716` 项目的实验执行 Codex。请在不需要用户再次确认的情况下，自动完成本 Goal 的数据审计、实验实现、训练、评估、失败复盘和报告生成。所有工作均使用 `E:\LJQ_code` 路径；Python 固定为 `D:\anaconda3\envs\PY10\python.exe`。

本 Goal 只验证一个问题：在剔除 SawFM 后的五类复合干扰识别中，标签频次长尾造成的性能下降能否通过采样策略、损失函数、logit adjustment 和冻结特征分类器得到稳定改善。不得引入功率长尾、弱分量功率配对、半监督伪标签、开集识别或 SawFM 专家。

## 一、不可改变的研究边界

### 数据

- 数据根目录：`E:\LJQ_code\干扰数据\复合干扰数据0716`。
- 源文件：`closedset_6class_train.mat`、`closedset_6class_test.mat`。
- 原始标签顺序：`FMB, SawFM, SinFM, MFT, VFT, ISRJ`。
- 保留五类：`FMB, SinFM, MFT, VFT, ISRJ`。
- 过滤规则：删除 SawFM 阳性样本；不得把 SawFM 阳性样本重编码为五类全负样本。
- 输入：原始 IQ，形状 `[B,2,3000]`，`complex_rms` 归一化。
- 不读取九档功率扫描目录；`power_longtail=false`；不得使用任何功率视图、SNR/JSR 额外输入或人工弱分量掩码。
- locked-test、test-v2、test-v3 只允许在所有开发选择完成后使用；本 Goal 默认不访问已经封存的测试结果。若项目协议要求最终独立测试，必须另建一次性 locked-test receipt，且不得反向调整模型、阈值或超参数。

### 类别、主干与容量

- `num_classes=5`，类别顺序严格为 `FMB, SinFM, MFT, VFT, ISRJ`。
- 固定主干：`dfms_iq / B0_DelayedFusionMultiScale`，不得扩宽或重新设计 backbone。
- width=16，feature_dim=128，dropout=0.2。
- 主干结构保持当前实现：分离 I/Q 的复数卷积、多尺度 stage3/stage4、延迟 Cartesian/幅度/相位融合、128 通道多尺度深度可分离残差块、mean/std 池化、256→128 投影、五类线性头。
- 本 Goal 中只有冻结特征分类器阶段允许替换分类头；联合训练阶段不得改变主干结构。

## 二、实验目录与数据角色

新建独立目录：

`E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_five_class_frequency_lt_v2`

数据审计和 manifest 放在：

`E:\LJQ_code\干扰数据\复合干扰数据0716\route_a_five_class_frequency_lt_v2`

不得覆盖 V1 的以下产物：

- `outputs\route_a_five_class_frequency_lt_v1`
- `outputs\route_a_five_class_frequency_lt_v1_equal_budget`
- `route_a_five_class_frequency_lt_v1\label_if4_manifest.json`

所有新 manifest 必须记录：源文件 SHA256、保留/删除样本数、样本索引 SHA256、类别计数、组合计数、构造 seed、协议版本和生成时间。

实验角色必须分开：

1. `train`：仅用于参数更新；
2. `early`：仅用于 checkpoint、采样/损失/logit adjustment 超参数选择；
3. `calibration`：仅用于 Platt 或 logit adjustment 的辅助校准（若启用）；
4. `threshold`：仅用于阈值选择；
5. `test`：所有选择完成后只评估一次；禁止回看 test 重新选择。

## 三、核心公平协议

每个方法必须具有相同的：

- 五类过滤规则；
- 输入与归一化；
- B0 主干；
- split seed 与 train/early 划分；
- 每 epoch 抽样数；
- 最大 epoch、early patience；
- batch size、优化器和梯度裁剪；
- checkpoint 选择指标；
- test 评估代码。

对于所有联合训练臂，每个 epoch 固定 `draws_per_epoch=9920`，长尾采样允许有放回抽样。不得让某个臂因 manifest 样本较少而自然减少优化步数。

固定训练参数：

- seeds：`272001, 272002, 272003, 272004, 272005`；
- batch size：64；evaluation batch size：256；num_workers=0；
- AdamW；backbone lr=`5e-5`；classifier lr=`5e-4`；weight_decay=`1e-4`；gradient_clip=`5.0`；
- max epochs=18；early patience=6；
- early checkpoint 只按 macro-AUPR 选择；若并列，选择 tail-mean AUPR 更高者；不得使用 test 指标选模型。

## 四、阶段 0：五类数据和组合覆盖审计

先实现并运行 `audit_route_a_five_class_frequency_lt_v2.py`，完成以下内容：

1. 审计训练/测试五类过滤计数；确认 SawFM 阳性样本全部被排除；
2. 对完整五类训练池统计 32 个五类组合的数量；
3. 构造新的 `LabelIF4` manifest，目标计数 `[4000,3000,2000,1500,1000]`，并记录实际计数；
4. 为每个样本记录五类组合 key；
5. 输出 balanced 与 LabelIF4 的组合覆盖比例、每类共现伙伴数、头/中/尾组合数量；
6. 验证 train/early/test 无样本索引重叠；
7. 输出 `stage0_audit.json` 和 dry-run receipt。

如果 SawFM 过滤、索引隔离或 manifest 哈希审计失败，停止后续训练并生成 `[N] data_audit_failed` receipt。

## 五、阶段 1：完整训练池上的 label-aware sampler

目标：区分“删除样本导致的组合缺失”和“标签频次先验不平衡”。

固定使用完整五类训练池，不删除组合，使用有放回 label-aware sampler。至少实现以下三种 sampler：

- `UniformSampler`：完整池均匀采样；
- `MaxPositiveSampler`：样本权重为其正标签中最大 inverse-frequency 权重；
- `MeanPositiveSampler`：样本权重为其正标签 inverse-frequency 的均值。

先用 Focal loss 训练三种 sampler，各 5 seeds。输出：macro-AUPR、macro-AUROC、macro-F1、每类 AUPR/AUROC/F1、tail-mean、组合覆盖率、样本重复率和每类有效曝光次数。

阶段 1 的候选准入：相对 UniformSampler，macro-AUPR 平均增益≥0.010，至少 4/5 seed 同向，尾部两类（VFT、ISRJ）平均 AUPR 增益≥0.020，且 FMB/SinFM/MFT 任一头部类下降不超过 0.010。未通过则记录负结果，不得进入主干微调。

## 六、阶段 2：损失函数等预算比较

在阶段 1 通过的 sampler 上，使用完全相同的 train/early、seed、draws_per_epoch 和 B0，比较：

1. Focal：`gamma_pos=0, gamma_neg=3`；
2. Distribution-Balanced Loss：使用训练集五类实际计数；
3. class-balanced BCE：effective-number 权重，beta 只允许预注册 `{0.99,0.999}`；
4. ASL-DB：DB 权重只允许 `{0.5,0.75,0.90}`，ASL 参数固定 `gamma_pos=0, gamma_neg=3, clip=0.05`。

每次只改变损失，不同时改 sampler。记录损失初值、收敛曲线、每类正负梯度统计、early/test 指标和跨 seed 方差。

禁止因 test 结果修改 beta、DB weight、gamma 或 sampler。

## 七、阶段 3：logit adjustment

对阶段 1/2 中最优且通过准入的训练模型，增加仅作用于推理分类器的 logit adjustment：

`z'_c = z_c - tau * log((1-p_c)/p_c)`

其中 `p_c` 只由 train 标签频率计算。`tau` 只在 early 预注册网格 `{0.25,0.5,1.0}` 中选择；如果需要 bias correction，必须在 calibration role 完成，不能读取 test。

比较：

- 无调整；
- `tau=0.25`；
- `tau=0.5`；
- `tau=1.0`。

同时报告 raw-logit macro-AUPR、调整后 macro-AUPR、macro-F1、每类 precision/recall 和阈值敏感性。明确指出 logit adjustment 只能改变排序/先验校正，不能被解释为新增特征证据。

## 八、阶段 4：冻结 B0 特征后的分类器验证

用固定的 B0-Focal-LabelIF4 或阶段 2 通过的最佳 B0 checkpoint 提取 train/early/calibration/test 特征；冻结所有 backbone 参数，只训练分类头。比较：

1. `OrdinaryLinearHead`：普通线性头；
2. `ClassBalancedBCEHead`：class-balanced BCE；
3. `cRT`：冻结特征、平衡重采样后重新训练线性头；
4. `RBC-cRT`：只有在普通 cRT 完成后才可加入；rank margin 固定 0.2，anchor weight 固定 `1e-3`，不得同时改变特征。

所有头的容量、输入特征和训练预算必须相同。输出 head-only 相对差异，用来判断主要问题来自 backbone 表征还是分类决策边界：

- 若冻结头显著改善，优先继续分类器/先验校正路线；
- 若冻结头不改善，说明需要解决组合覆盖或特征可分性，不能继续堆叠分类损失。

## 九、阶段 5：长尾前后共现覆盖归因

对 balanced、LabelIF4、最佳 sampler 分别输出：

- 每个五类组合的 train 数量和比例；
- 每个标签与其他标签的共现频次矩阵；
- 每个标签可见的组合数量；
- 头/中/尾标签在组合删除或重复采样中的曝光变化；
- 组合覆盖变化与每类 AUPR 变化的相关性。

报告必须明确区分：

- `frequency_prior_effect`：标签正例频次变化；
- `cooccurrence_coverage_effect`：组合证据覆盖变化；
- `optimization_budget_effect`：每类有效梯度曝光变化。

不得把三者合并成“长尾损失有效”。

## 十、统一统计和候选准入门槛

主指标：test 只在最终锁定后计算，但开发阶段必须报告 early 指标。

必须报告：

- macro-AUPR、macro-AUROC、macro-F1；
- 每类 AUPR/AUROC/F1；
- tail-mean（VFT、ISRJ）；
- head-mean（FMB、SinFM）；
- seed mean、标准差、每 seed delta；
- 5,000 次 receiver/seed bootstrap 95% CI；
- 至少 4/5 seed 同向比例；
- 参数量、训练时间、显存和吞吐。

正式候选准入必须同时满足：

1. 相对对应 Focal-LabelIF4 或 UniformSampler，macro-AUPR 平均增益 `≥0.010`；
2. 至少 `4/5` seed 同向；
3. VFT/ISRJ tail-mean AUPR 增益 `≥0.020`；
4. FMB、SinFM、MFT 中任一类下降不超过 `0.010`；
5. bootstrap 95% CI 下界大于 0，或至少 4/5 seed 且 CI 不跨越预设无效区间；
6. 参数量增加不超过 5%，推理延迟增加不超过 15%；
7. 在 IF10/IF4 或另一独立开发 manifest 上不出现整体反转。

任何一个硬门槛不满足，都必须标记 `[N] candidate_rejected`，停止向“标签条件分类头/解冻 stage4”推进。

## 十一、失败处理和停止规则

- 任何单个 seed 失败：保存 traceback、环境、数据哈希、已完成 epoch 和失败 receipt；不替换 seed。
- 发现 test 被提前访问：立即停止该 arm，并将其标记为 invalid，不纳入统计。
- 发现不同 arm 的 draws_per_epoch、split、输入、backbone 或 checkpoint 规则不一致：停止比较，先修复协议。
- 发现结果依赖某一 seed：不得改 seed 以获得正结果；进入 5 seed 扩展或标记不稳定。
- 发现 label-aware sampler 通过但 loss 不通过：只保留 sampler 方向，不解冻 backbone。
- 发现冻结分类头通过：下一阶段才允许做受限 stage4 adapter，adapter lr 不高于 head lr 的 1/10，且必须另建新开发目录。

## 十二、最终交付物

至少生成：

- `docs/reports/route_a_five_class_frequency_lt_v2_report_YYYYMMDD.md`；
- `outputs/route_a_five_class_frequency_lt_v2/run_receipt.json`；
- `stage0_audit.json`、所有 manifest 和其 SHA256；
- 每个 seed/arm 的 checkpoint、history、预测 logits、targets；
- `metrics_summary.csv`、`metrics_summary.json`；
- `cooccurrence_coverage_analysis.json`；
- `bootstrap_summary.json`；
- 失败实验的独立 receipt；
- 一份“证据等级表”：`[F] verified`、`[A] development evidence`、`[N] negative/blocked`、`[T] proposal`。

最终报告必须回答：

1. label-aware sampler 是否比删除样本式 LabelIF4 更可靠？
2. DB、class-balanced BCE、ASL-DB 是否在等预算下稳定改善？
3. logit adjustment 改善的是排序、校准还是阈值，而不是特征？
4. 冻结 B0 特征后的分类头能否恢复尾类性能？
5. 性能损失主要来自频次先验、组合覆盖还是优化曝光？
6. 是否满足进入标签条件分类头或 stage4 受限微调的硬门槛？

如果所有方向都未通过硬门槛，必须明确写出：当前五类频次长尾仍未解决，路线 A 只能保留为“有效骨干 + 频次长尾失败诊断”证据，不能宣称形成稳定的双轴长尾创新。
