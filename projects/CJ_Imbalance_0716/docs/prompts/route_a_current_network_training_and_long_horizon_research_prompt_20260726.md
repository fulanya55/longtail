# 路线 A 当前网络、训练体系与长线实验规划专业 Prompt（2026-07-26）

## 使用说明

本 Prompt 是对 `route_a_research_status_gap_and_future_experiments_prompt_20260725.md` 的更新和替代。旧 Prompt 中的 S2/C2/A0 属于当时的阶段候选；最新五种子证据已经把当前候选推进为：

`F2_DFMS_Full + H2_PosBCE（五个非 SawFM 标签） + RawStage1SawFMFrontend + Platt（SawFM）`

即 `RawSaw+H2Others` 标签级路由。

本文件可直接交给科研智能体、代码智能体或大模型，用于全面复核当前实验状态、规划长线实验或在明确授权后继续实现。使用者不得删除失败结果、seed 2039 的 FPR 风险、两组 Raw checkpoint 的协议差异和 locked-test 边界。

## 当前体系的最短摘要

```text
输入 I/Q [B,2,3000]
│
├─ 六类共享主线：F2_DFMS_Full
│  ├─ 复数实部/虚部延迟融合主干
│  ├─ 多尺度复数 stage1～stage4
│  ├─ Cartesian + magnitude + lag(1,4,16) phase 晚期融合
│  ├─ 全局 mean/std 安全池化
│  └─ 128维共享表示
│       └─ 冻结 H2_PosBCE 线性六标签头
│            └─ 最终保留 FMB、SinFM、MFT、VFT、ISRJ 五列
│
└─ SawFM 专线：RawStage1SawFMFrontend(mode=raw)
   ├─ 原始复数 I/Q 的幅度、幅度变化、相位增量描述量
   ├─ k=7/31/63 深度可分离多尺度时序卷积
   ├─ mean/std/max/top-k 局部证据统计
   ├─ 88→32→1 SawFM logit
   ├─ ordinary calibration 上拟合 Platt
   └─ ordinary + power threshold 角色共同选择统一阈值

最终路由：SawFM 使用 Raw+Platt；其余五类使用 H2；不学习融合权重。
```

## 可直接复制的专业 Prompt

````text
你是一名专注于雷达复合干扰识别、复数时序网络、多标签长尾学习、弱分量检测、概率校准与可信机器学习实验设计的高级科研人员。请以 SCI 论文、可复现工程和可证伪研究的标准，全面审视下述项目。你的任务不是泛泛提出“加注意力、换损失、调参数”，而是基于现有正式实验产物，准确说明当前网络、训练模式、已验证机制、失败机制、证据边界和未来长线实验路径。

一、项目与环境边界

项目根目录：
C:\LJQ_code\CJ_Imbalance_0716

主要 Python 环境：
D:\anaconda3\envs\PY10\python.exe

主要数据源：
C:\LJQ_code\干扰数据\复合干扰数据0716

主要配置：
C:\LJQ_code\CJ_Imbalance_0716\configs\student_a\six_class.yaml

六类固定顺序：
[FMB, SawFM, SinFM, MFT, VFT, ISRJ]

输入：
- 真实复数 I/Q，张量形状 [B,2,3000]；
- 使用 complex RMS normalization；
- 根训练数据包含63个非空标签组合，每组合400个样本，共25200个样本；根数据本身平衡，不得直接称为长尾数据；
- 标签长尾由可复现 label-LT manifest 构造，当前 F2 训练使用2400样本，目标/实际标签计数为1600、1213、919、696、528、400；
- power sweep 的多个功率视图共享 receiver_noise_seed，是同一身份的配对视图，不能当作独立样本；
- train、head validation、ordinary calibration、ordinary threshold、ordinary early、power fit/threshold/development 必须严格隔离；
- locked test 当前未使用，未经用户明确批准不得读取；
- 所有结论必须优先读取 summary JSON、predictions、run receipt、split/manifest/checkpoint SHA256，不得只根据 Markdown 或记忆猜测。

二、首先必须复核的最新证据

1. 当前五种子最终 development 结果：
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_raw_platt_five_seed_label_routing_results_20260726.md

2. 正式汇总 JSON：
C:\LJQ_code\CJ_Imbalance_0716\outputs\tapc\route_a_raw_platt_five_seed_label_routing\route_a_raw_platt_five_seed_label_routing_summary.json

该汇总 SHA256：
1C456FF579F0FA2EFCCBA202152F5F2CA117CA215A7D84131447D4086D491EFC

3. 当前实现：
C:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_raw_platt_five_seed_label_routing.py

4. locked-test 申请包：
C:\LJQ_code\CJ_Imbalance_0716\docs\plans\route_a_raw_saw_label_routing_locked_test_request_20260726.md

5. F2 结构证据：
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_dfms_iq_structure_screen_results_20260724.md

6. H2 冻结头证据：
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_dfms_iq_frozen_heads_results_20260724.md

7. Raw 前端形成过程：
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_trainable_raw_frontend_and_new_split_confirmation_results_20260725.md
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_h2_raw_frontend_bounded_reconnection_results_20260725.md
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_raw_frontend_independent_classifier_audit_results_20260725.md

三、当前冻结候选网络

当前方法不是一个端到端联合训练的单网络，而是一个“共享五标签主线 + SawFM 标签专线 + 冻结标签路由”的解耦系统。

1. 共享主干 F2_DFMS_Full

源码：
C:\LJQ_code\CJ_Imbalance_0716\cjlt\models\phase_multiscale.py

输入与总体参数：
- 输入 [B,2,3000]；
- width=16；feature_dim=128；dropout=0.20；
- 包含原六标签分类器时约257511参数；
- architecture_options：delayed_fusion=true、complex_multiscale=true、include_phase=true、phase_lags=(1,4,16)、attention=false、pooling=global_statistics。

复数编码器：
- stage1：ComplexResidualBlock1d，1→16，kernel=7，保持 real/imag 两条复数流，长度3000；
- AvgPool1d，kernel=2，stride=2；
- stage2：ComplexResidualBlock1d，16→32，kernel=5，长度1500；
- stage3：ComplexMultiScaleResidualStage1d，32→48，多尺度支路 kernels=(3,7,15)、dilations=(1,2,4)，长度约750；
- stage4：ComplexMultiScaleResidualStage1d，48→64，相同多尺度机制，长度约375；
- I/Q 直到 stage4 仍保留独立 real/imag 复数表征，避免原 PAM-IQ 的过早融合。

晚期融合与全局表示：
- Cartesian 投影支路；
- 稳定幅度支路；
- 多滞后相位支路，lags=(1,4,16)，使用归一化复相位的 cos/sin 相互作用；
- 三路融合为128通道，相位支路通过有界残差尺度加入；
- 128通道 MultiScaleDepthwiseResidual1d，kernels=(3,7,15)、dilations=(1,2,4)；
- 不使用注意力池化，使用 GlobalStatisticsPooling1d 的全局 mean/std，得到256维统计；
- Linear + LayerNorm + GELU + dropout 投影为128维共享特征。

F2 已验证作用：
- 延迟融合和深层复数多尺度结构提高总体 macro-AUPR；
- 多滞后相位支路对 MFT、VFT、ISRJ 弱证据有效；
- F2 总体结构优于 PAM-IQ，但 SawFM 是结构例外；
- 冻结 probe 证明大量弱证据仍保留在128维表示中；
- 因此当前不应继续盲目扩宽主干或解冻更多 stage。

2. H2_PosBCE 六标签冻结头

结构：
- F2 全部冻结并保持 eval；
- 输入128维 F2 全局表示；
- 单层 Linear(128,6)，共774个参数；
- 当前最终路由只使用 H2 的 FMB、SinFM、MFT、VFT、ISRJ 五个输出，H2 的 SawFM 输出仅作为对照。

训练：
- F2 label-LT manifest 的2400个样本；
- 80% head fit、20% head validation，即1920/480；
- 每标签 pos_weight=negative_count/positive_count；
- BCEWithLogitsLoss(pos_weight)；
- AdamW，lr=1e-3，weight_decay=1e-4；
- batch=256；epochs=60；patience=12；gradient_clip=5；
- checkpoint 以 validation macro-AUPR 为主、weak AUROC 为次；
- ordinary calibration role 进行温度校准；
- ordinary threshold role 使用逐标签 weak-recall/FPR 阈值，negative FPR 目标≤0.20。

H2 的定位：
- 是当前稳定的六类共享表示/分类器基线；
- 不是 SawFM 最终分类器；
- 不允许因为 SawFM 专线而修改其他五类 H2 分数和阈值。

3. RawStage1SawFMFrontend(mode=raw)

源码：
C:\LJQ_code\CJ_Imbalance_0716\cjlt\models\evidence_readout.py

当前实际只使用 raw branch，不使用 stage1 branch。总参数约5521。

输入描述量：
- 原始 raw_real/raw_imag [B,1,3000]；
- magnitude mean；
- magnitude max；
- magnitude std；
- 相邻时刻 magnitude 绝对差；
- 归一化复数相邻相位增量 cos；
- 归一化复数相邻相位增量 sin；
- 形成6通道显式物理描述序列。

时序编码：
- Conv1d 1×1：6→16；GroupNorm(4)；SiLU；
- 三个 depthwise temporal branches：kernel=7、31、63，通道均为16；
- 拼接为48通道后用1×1 Conv融合回16通道；GroupNorm；SiLU；
- AvgPool1d，kernel=4，stride=4；
- 对编码特征计算 mean/std/max/top-k mean，top-k=16，得到64维；
- 对原始6通道描述量同样计算 mean/std/max/top-k，得到24维；
- 合并为88维；
- MLP：Linear(88,32)→GELU→Dropout(0.1)→Linear(32,1)，输出 SawFM logit。

Raw 训练方式：
- 不与 F2/H2 联合反向传播；
- 使用 SawFM present 与“其他标签组合相同但 SawFM absent”的组合匹配 hard negative；
- 同时使用 root 与40/45/50 dB配对视图；
- BCEWithLogitsLoss；
- AdamW，lr=3e-4，weight_decay=1e-3；
- batch=64；epochs=40；patience=8；gradient_clip=5；
- checkpoint 由独立 power validation mean AUROC 选择，AUPR作为次指标；
- 2039/2040 使用已冻结的 R0_FullPhysics 新 split checkpoint；
- 2041/2042/2043 使用已冻结的 N0_RawOnly checkpoint；
- 两组网络和优化参数相同，但训练角色抽样协议不同，必须在报告中披露，不得把它们描述为完全同协议的五次重复训练。

Raw 机制消融：
- 完整物理描述量开发 AUROC 约0.999；
- phase-only 约0.995；
- envelope-only 约0.749；
- time-shuffled 约0.711；
- 说明主要机制是相位与局部时间结构，而不是简单幅度或静态分布捷径。

4. Platt 校准与统一阈值

- 网络全部冻结后，在 ordinary calibration role 上拟合一维 LogisticRegression(C=1.0, solver=lbfgs)；
- 只输入 Raw logit，不输入真实功率、JNR、receiver_noise_seed或生成参数；
- threshold grid=0.01:0.01:0.99；
- 同一个 SawFM 阈值必须同时满足 ordinary threshold negative FPR≤0.20，以及40/45/50 dB power threshold hard-negative FPR均≤0.20；
- 可行阈值按 mean power recall、ordinary recall、50 dB recall、较高阈值依次选择；
- ordinary early 和 power development 只做评价，不再调参。

5. 最终标签级路由 RawSaw+H2Others

- FMB、SinFM、MFT、VFT、ISRJ：H2 probability + H2 per-label threshold；
- SawFM：Raw+Platt probability + Raw unified threshold；
- 只替换索引1的 SawFM 概率和阈值；
- 不训练 gate，不进行 logit 加权，不进行组合条件路由，不使用 oracle power；
- 因为其他五列保持不变，其他五类 FPR 理论和实测变化均为0。

四、当前训练模式的层级关系

请明确区分五种模式：

模式 M1：端到端共享表示训练
- 对象：F2；
- 目标：学习六类 label-LT 共享复数表征；
- 损失：Distribution-Balanced Loss；
- 只在结构筛选阶段进行，目前已冻结。

模式 M2：冻结表示后的分类器解耦
- 对象：H2；
- F2 不更新；
- 目标：用正样本加权 BCE 重新形成六类边界；
- 结果证明分类边界/工作点是 F2 弱召回问题的重要来源。

模式 M3：标签专项独立专家训练
- 对象：Raw SawFM frontend；
- 不共享 F2 的128维瓶颈；
- 用组合匹配 hard negative 和多功率视图学习 SawFM 局部证据；
- 这是当前最有效的 SawFM 机制。

模式 M4：后训练概率校准与受约束阈值
- 对象：Raw Platt 和 H2 temperature/threshold；
- 不更新神经网络；
- calibration、threshold、evaluation 三个角色严格分离。

模式 M5：无训练标签路由
- 只在输出层替换 SawFM 一列；
- 目标是把专项增益传递到六类系统，同时保证其他五类完全隔离。

五、当前正式 development 结果

五种子：2039、2040、2041、2042、2043。

Raw+Platt 五种子聚合：
- ordinary AUROC=0.9384；
- ordinary AUPR=0.9547；
- ordinary recall=0.8813；
- ordinary FPR=0.1960；
- without-swept SawFM recall=0.8653；
- 40/45/50 dB mean recall=0.9879；
- 50 dB recall=0.9758；
- max power FPR=0.0760；
- Brier=0.0977；ECE=0.0504；
- 预注册要求至少4/5 seeds通过九项门禁，实际4/5通过。

关键风险：
- seed 2039 ordinary early FPR=0.2806，超过0.20；
- 该seed的AUPR、recall、without-swept、power、Brier/ECE均通过；
- 这是threshold→early跨角色稳定性问题，不是排序能力失败；
- 不得事后用 ordinary early 为seed 2039重选阈值。

六类标签路由五种子聚合：
- H2All macro-AUPR=0.8667，RawSaw+H2Others=0.9001，增量+0.0334，bootstrap 95% CI约[+0.0281,+0.0384]；
- macro-F1=0.7573→0.8027，增量+0.0454；
- Hamming loss=0.2307→0.1959；
- SawFM recall=0.4870→0.8813，增量+0.3943；
- exact match=0.2280→0.2747，增量+0.0467；
- 其他五类最大FPR变化=0；
- 标签路由5/5 seeds通过预注册门禁；
- locked_test_used=false。

六、必须保留的历史负结果

1. Base-DB仍是频率长尾主要传统基线；ASL-DB Mix75/Mix90改善SawFM但损害macro-AUPR，存在总体—尾类Pareto冲突。
2. PAM-IQ的早期I/Q融合和共享注意力池化存在结构偏置；F2修复了部分问题，但没有独立解决SawFM。
3. 冻结F2后增加普通MLP没有稳定超过线性probe，盲目增加分类器容量不是答案。
4. Stage2低秩adapter/teacher-student/hard-negative路线曾得到局部排序改善，但跨development稳定性不足；不得把它重新描述成当前主方法。
5. S2与H2固定融合C2改善排序，但统一阈值下50 dB召回失败；排序改善不等于决策改善。
6. VisibilityAffine、ordinal visibility anchor、selective unknown均未通过原晋级门；可见度坐标可预测，但“可见度→共享logit偏置”机制不成立。
7. Raw作为H2的有界残差失败：H2 ordinary recall约0.035→0.055，50 dB约0.069→0.100，0/3门禁；失败原因是Raw与H2分数几何不兼容，小残差无法跨越H2错误边界。
8. Raw作为独立分类器成功，证明之前失败来自融合方式，而不是Raw特征不足。
9. 路线B的半监督/power-SSL是负结果，不得在没有新可证伪机制时复活；监督参数联合训练改善参数NMAE约12.1%，但没有稳定改善分类，应与路线A保持分离。

七、当前研究已经做到什么程度

已完成：
- 根数据、label-LT、power identity和locked-test治理；
- Base-ASL/Base-DB及尾类旋转对照；
- PAM-IQ、容量基线、F1、F2结构筛选；
- F2冻结特征充分性和H2分类器解耦；
- 多轮SawFM专家、teacher-student、hard-negative、calibration、selective和残差实验；
- Raw物理描述前端的机制消融、新split确认、三种子全分布审计；
- 五种子Raw+Platt确认；
- 五种子六类标签级路由验证；
- 全部development产物、预测、checkpoint/split/selection哈希和receipt；
- 当前代码全量PY10测试已通过207项。

尚未完成：
- 一次性locked-test最终确认；
- seed 2039 FPR波动的独立新development复核；
- 在第二个以上标签复制“标签专项物理前端”的有效性；
- 对“双轴长尾”的完整2×2因果证据；
- LOCO/LSCO、未见组合、未见功率、接收机/噪声条件迁移；
- 与最新长尾、多标签、弱信号方法的严格同协议比较；
- 计算量、延迟、显存、参数量和部署成本表；
- 外部数据或新生成批次验证。

八、未来长线实验规划原则

必须遵循：
- 一次只回答一个主要科学问题；
- 每阶段预注册数据角色、对照、超参数、指标、晋级门和停止条件；
- 不同时修改主干、损失、专家和阈值；
- development和locked test分开；
- 不得根据locked-test结果再调参；
- 若当前候选先执行locked test，则其后任何改进必须使用新的独立测试集，不能继续复用该locked test作为调参集。

阶段 L0：冻结审计与最终测试决策

目标：决定当前论文版本是否结束development并一次性使用locked test。

工作：
- 核对五个seed的F2/H2/Raw checkpoint SHA256；
- 固化Platt、threshold、label routing、命令和环境；
- 明确不使用ensemble，默认五个seed独立评价；
- 输出测试前注册文件和只读manifest；
- 未获得用户明确授权时停止，不读取locked test。

成功标准：审计无漂移，用户明确批准后仅执行一次。

阶段 L1：阈值跨角色稳定性研究

仅在选择继续development而暂不解锁测试集时执行。

科学问题：seed 2039 FPR超限来自有限样本方差、校准漂移还是checkpoint协议差异？

对照：
- T0：当前per-seed Platt + per-seed unified threshold；
- T1：K-fold cross-fitted Platt，但阈值规则不变；
- T2：跨seed共享阈值，仅作为稳健性对照；
- T3：conformal/risk-control阈值，仅使用calibration/threshold角色；
- 不允许用ordinary early选方案。

指标：ordinary/power FPR、recall、Brier、ECE、threshold方差、coverage、worst-seed和组合分层。

晋级门：至少4/5 seeds ordinary FPR≤0.20，平均recall不低于当前0.8813-0.02，50 dB recall不低于0.95，macro-AUPR不下降超过0.005。

停止规则：如果FPR波动没有降低或召回下降超过0.02，保持当前Platt，不继续复杂校准。

阶段 L2：Raw专家协议统一复现

科学问题：2039/2040与2041～2043的训练角色差异是否影响结论？

设计：
- 用完全相同的身份抽样、fit/validation角色和训练预算重新训练5个Raw checkpoint；
- 网络、优化器和阈值规则全部冻结；
- 与现有heterogeneous checkpoint组做配对比较；
- 每seed至少2个初始化，只能由原validation选择。

晋级门：五seed ordinary AUPR≥0.93、without-swept recall≥0.80、power mean recall≥0.95，且初始化标准差和threshold方差下降。

停止规则：统一协议若没有降低方差，不将其包装为新方法，只保留为复现性补充。

阶段 L3：从SawFM专项扩展到多标签专项证据框架

科学问题：Raw显式物理描述+组合匹配hard negative是否可迁移到SinFM、MFT、VFT、ISRJ？

必须先做oracle可辨识性筛选：
- 每标签构造“其他标签组合相同、目标标签present/absent”的匹配对；
- 比较Raw descriptor、F2 stage1/stage2、F2全局特征；
- 只在独立development AUROC≥0.70、validation-development gap≤0.15时训练正式专家；
- 每个标签根据物理机理选择descriptor，不能复制SawFM相位描述后直接声称通用。

正式晋级门：至少两个新增标签在5 seeds上AUPR和受约束recall稳定提升，其他标签非劣，才能把论文表述从“SawFM专项方法”扩展为“通用标签条件弱证据框架”。

阶段 L4：双轴长尾因果证据

科学问题：频次偏置和可见度退化是否是可分离且互补的两个机制？

建议2×2因子实验：
- Frequency remedy off/on：BCE或ASL vs DB；
- Visibility remedy off/on：H2All vs RawSaw+H2Others；
- 在balanced-root、label-LT、combo-LT、joint-LT四种manifest上运行；
- power视图仍按identity配对，不称为独立长尾样本。

主要统计：macro/tail AUPR、弱分量recall、逐标签FPR、interaction effect、组合频次分层、功率可见度分层。

晋级门：DB主要改善frequency strata，Raw专家主要改善SawFM visibility strata，二者组合效果不被单一因素解释；至少5 seeds且交互项方向稳定。

阶段 L5：组合和域外泛化

设计：
- LOCO：留一标签组合；
- LSCO：留一相似组合簇；
- 未见功率插值/外推；
- receiver noise、载频、带宽或生成参数变化；
- 新批次或外部数据。

必须报告：known-combo与unseen-combo差距、calibration漂移、worst-group、power-level和receiver-seed分层，不仅报告总体均值。

阶段 L6：效率与部署

- 报告F2、H2、Raw、完整路由的参数量、MACs、显存、batch吞吐和单样本时延；
- 比较只运行H2、始终运行Raw、H2低置信触发Raw三种部署模式；
- 触发式Raw只能用部署时内部置信度，不能读取真实功率；
- 若触发策略降低计算量，必须同时报告漏触发率和SawFM召回损失。

阶段 L7：SCI最终证据链

至少包括：
- Base-ASL、Base-DB、PAM-IQ、F2+H2、Raw-only、RawSaw+H2Others；
- 结构消融：无phase、envelope-only、phase-only、time-shuffled、无top-k、单尺度；
- 训练消融：无hard negative、无power view、随机negative；
- 校准消融：uncalibrated、Platt、Isotonic或risk-control；
- 五seed均值/标准差/bootstrap CI；
- 逐标签、逐组合、逐功率、强/弱、with/without-swept；
- 参数量和推理成本；
- 冻结后一次性locked test；
- 失败结果和适用范围。

九、论文创新表述边界

当前可以支持的表述：
- 延迟复数融合和多滞后相位结构改善六类共享表征；
- SawFM弱证据需要绕过共享128维瓶颈的标签专项局部证据路径；
- 显式幅度/相位增量描述、多尺度局部卷积、top-k统计和组合匹配hard negative形成有效SawFM专家；
- 无训练标签路由能够隔离地提高SawFM，并把增益传递到多标签总体指标；
- 失败的有界残差与成功的独立判决形成清晰机制对照。

当前不能支持的表述：
- 已解决全部六类弱分量问题；
- 根数据天然是三轴长尾；
- power views是独立长尾样本；
- 已在最终测试集验证；
- Raw结构对所有干扰类型通用；
- 任何基于真实功率/JNR元数据的部署能力；
- 路线B半监督方法有效。

十、你必须输出的内容

请输出一份专业中文报告，至少包括：

1. 500字以内执行摘要。
2. 当前网络结构图：标注输入、张量尺寸、模块、冻结/训练状态、参数量和最终路由。
3. F2、H2、Raw、Platt、threshold、label routing完整超参数表。
4. M1～M5训练模式及数据角色流程图。
5. 当前五种子结果表和置信区间。
6. 试验演进表：Base-DB→PAM-IQ→F2→H2→Stage2专家→C2/A0/Selective→Raw residual失败→Raw independent成功→五种子label routing。
7. 已验证事实、尚未验证假设、历史负结果三张独立清单。
8. 当前系统相对论文目标的差距矩阵。
9. 对seed 2039 FPR风险和两组Raw checkpoint协议差异的专项分析。
10. L0～L7长线实验路线图，每阶段写明科学问题、对照、数据角色、训练参数、指标、晋级门、停止条件和产物。
11. SCI创新点和证据链，区分主创新、支撑机制、负结果、工程贡献。
12. 风险清单：数据泄漏、identity泄漏、checkpoint选择混杂、阈值过拟合、功率元数据捷径、seed不稳定、协议异质性、locked-test污染和过度表述。
13. 下一步唯一推荐动作，只能在以下选项中选择：
   A. 冻结当前版本并申请一次性locked test；
   B. 暂不解锁测试集，先执行L1阈值稳定性；
   C. 执行L2 Raw协议统一复现；
   D. 当前路线证据不足，退回F2+H2。

十一、分析和写作规则

- 所有指标必须注明数据角色、seed数和locked_test_used状态；
- 将AUROC/AUPR排序能力与阈值后recall/FPR分开；
- 将development证据与最终测试证据分开；
- 不得把proposal写成validated result；
- 不得根据ordinary early或locked test重新选择阈值；
- 不得把2039/2040和2041～2043描述为完全相同Raw训练协议；
- 不得忽略seed 2039 FPR=0.2806；
- 每个未来建议都必须包含可证伪预测和失败停止规则；
- 若建议修改模型，必须说明它解决哪个已观察失败机制，以及为什么现有模块无法解决；
- 优先使用已有脚本和产物，不重复训练已冻结对照；
- 所有新实验保存config、environment、history、predictions、metrics、threshold search、identity manifest、checkpoint/split/data SHA256和run receipt。

十二、执行权限

如果用户只要求分析和规划：只读取并报告，不修改代码、不运行实验、不读取locked test。

如果用户明确要求“按方案执行”：
- 先写预注册方案；
- 使用PY10运行单seed smoke；
- smoke只验证工程路径，不进入性能结论；
- 再运行正式多seed development；
- 达不到门槛时停止，不自动进入下一阶段；
- locked test仍必须获得单独、明确授权。
````

## 推荐给下一位执行代理的追加语句

如果希望下一位代理在完成分析后继续 development，可在 Prompt 末尾追加：

> 在完成证据复核后，只执行你最终选择的一个 development 阶段。先生成预注册技术方案和门禁，再用 PY10 进行单种子 smoke 与正式多种子实验。不得读取 locked test。若门禁失败，停止并输出失败机制；若门禁通过，只形成下一阶段申请，不自动扩展任务范围。

如果准备申请最终测试，可追加：

> 只审查 locked-test 申请包、checkpoint/配置/阈值哈希和一次性测试规则。在我明确回复批准前，不读取或运行 locked test。

