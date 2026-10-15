# 路线 A 当前研究状态、目标差距与未来实验规划 Prompt

## 使用说明

下面的 Prompt 用于让科研智能体、代码智能体或大模型在不丢失既有证据链的前提下，重新审视路线 A 的六类复合干扰长尾识别研究，并提出可执行、可证伪的下一阶段方案。

它将当前工作区中的结论分成四层：

1. 六类长尾主对比基线 Base-DB；
2. 六类结构主线 F2 + H2；
3. SawFM 弱分量排序主线 S2 + C2；
4. 尚未晋级的决策校准 A0/V2，以及已经否决的有序三锚点 A2。

使用时不要删除路径、指标、失败结果和停止条件。若要让智能体直接修改代码或执行实验，可在 Prompt 最后一段追加“完成设计后按预注册方案实现并运行”。

## 可直接复制的专业 Prompt

```text
你是一名专注于雷达复合干扰识别、多标签长尾学习、弱信号检测、选择性分类与可信实验设计的高级科研人员。请以 SCI 论文方法论和可复现实验的标准，对下面项目进行系统审视，并给出严格区分“已验证事实、尚未验证假设、失败路线、当前候选、下一步实验”的研究报告与实验方案。

一、项目边界与强制协议

项目根目录：
C:\LJQ_code\CJ_Imbalance_0716

主要 Python 环境：
D:\anaconda3\envs\PY10\python.exe

主要数据集：
C:\LJQ_code\干扰数据\复合干扰数据0716

配置文件：
C:\LJQ_code\CJ_Imbalance_0716\configs\student_a\six_class.yaml

六类固定顺序：
[FMB, SawFM, SinFM, MFT, VFT, ISRJ]

输入协议：
- 真实复数 I/Q 输入，[B,2,3000]；
- complex RMS normalization；
- 根数据集本身是平衡数据，不得直接称为长尾数据；
- 标签长尾必须来自可复现的 label-LT manifest；当前审计样本数为2400，目标/实际标签计数约为1600、1213、919、696、528、400；
- 40/45/50 dB 等功率视图是同一 receiver_noise_seed 的配对观测，不能作为相互独立的新样本；
- 训练、校准、阈值选择、early-development 必须按身份隔离；
- locked test 当前继续冻结，除非方法、超参数、阈值和晋级规则已经全部冻结并获得明确授权；
- 所有结论必须优先读取正式 summary JSON、run receipt、prediction 和 checkpoint hash，不得只依据 Markdown 摘要猜测。

研究总体目标是：
在六类复合干扰标签频率长尾与标签内部功率/可见度长尾共同存在时，同时保持总体多标签识别性能、弱分量召回、低误报和跨功率稳定性，并形成具有 SCI 创新价值的完整证据链。

二、必须首先复核的证据文件

1. Base-DB：
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_six_class_db_path_results_20260723.md

2. F2 结构筛选：
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_dfms_iq_structure_screen_results_20260724.md

3. 冻结分类头 H2：
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_dfms_iq_frozen_heads_results_20260724.md

4. F2 特征充分性：
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_dfms_iq_feature_sufficiency_results_20260725.md

5. SawFM stage2 专家：
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_sawfm_stage2_power_teacher_student_results_20260725.md

6. 专家接入和固定融合：
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_sawfm_expert_routing_results_20260725.md
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_sawfm_fixed_fusion_cross_power_results_20260725.md

7. 决策校准和专家稳定性：
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_power_decision_and_s2_stability_results_20260725.md
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_stable_expert_visibility_calibration_results_20260725.md

8. 最新有序可见度失败实验：
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_ordinal_visibility_anchor_results_20260725.md
C:\LJQ_code\CJ_Imbalance_0716\outputs\tapc\route_a_ordinal_visibility_anchor\route_a_ordinal_visibility_anchor_summary.json

三、当前网络和训练体系

请将当前体系理解为多个证据层，而不是一个已经整体晋级的最终网络。

1. 六类论文对比基线：Base-DB

- Base-DB 是六类阶段必须保留的主要对比基线；Base-ASL 只作为传统损失参照。
- Base-DB 三种子 SawFM 实验点的 macro-AUPR 为0.826986，SawFM AUPR为0.736400。
- 对18组尾类旋转配对，Base-DB 相对 Base-ASL 的 macro-AUPR 平均提高0.017870，18/18获胜。
- ASL-DB Mix75/Mix90 虽使 SawFM AUPR 提高约0.068～0.069，但 macro-AUPR下降约0.047～0.049，属于尾类与总体性能的 Pareto 冲突，不能晋级为总体方法。
- DB 表征后仅训练 ASL 分类头也未超过完整 Base-DB，说明表征与分类边界存在目标协同。

2. 当前六类结构主干：F2_DFMS_Full

模型文件：
C:\LJQ_code\CJ_Imbalance_0716\cjlt\models\phase_multiscale.py

结构参数：
- 输入：[B,2,3000]；width=16；feature_dim=128；dropout=0.20；总参数约257,511。
- 复数 block1：ComplexResidualBlock，1→16，kernel=7。
- AvgPool stride=2。
- 复数 block2：16→32，kernel=5，得到 stage2 real/imag [B,32,1500]。
- stage3：三分支复数多尺度卷积，kernels=(3,7,15)，dilations=(1,2,4)，32→48，时间长度约750。
- stage4：相同三分支设计，48→64，时间长度约375。
- I/Q 在 stage4 前始终保持独立复数流，避免原 PAM-IQ 过早融合。
- 晚期融合包含 Cartesian、稳定幅度和多滞后相位三个分支。
- 相位滞后为(1,4,16)，使用归一化相位相关的 cos/sin 形式；相位残差门初始scale=0.1，实验中 tanh(scale)约0.18～0.22。
- 三个融合分支输出128通道并残差相加。
- 后接深度可分离多尺度时序块，kernels=(3,7,15)，dilations=(1,2,4)。
- 使用无注意力的全局 mean/std 安全池化，256维统计投影为128维。
- 最终六标签线性分类器。

F2 端到端训练超参数：
- label-LT 2400样本；seeds=2041/2042/2043；
- Distribution-Balanced Loss；
- AdamW，learning rate=3e-4，weight decay=1e-4；
- batch size=64；epochs=20；warmup=10%；AMP；
- checkpoint 由 development macro-AUPR 选择。

DB Loss 默认参数：
- map_alpha=0.1；map_beta=10.0；map_gamma=0.2；
- negative_scale=2.0；initial_bias=0.05；
- focal_gamma=2.0；balance=2.0。

F2 已验证结果：
- macro-AUPR=0.8562，优于容量匹配 B1 的0.8481；
- Hamming=0.2485，优于B1的0.2724；
- weak-vs-negative AUROC=0.5886；
- 弱分量召回仅0.3586，低于B1的0.5561；
- 冻结线性 probe 达到 macro-AUPR=0.8657、weak AUROC=0.6448，说明大量弱证据仍保留在特征中；
- 多滞后相位支路能改善MFT、VFT、ISRJ，但SawFM弱可分性仍是主要例外。

3. 当前六类冻结分类头：H2_PosBCE

- F2 完全冻结并保持eval状态；
- 输入F2的128维全局特征；
- 线性六标签头；
- 每类 pos_weight=negative_count/positive_count；
- BCEWithLogitsLoss(pos_weight)；
- AdamW，lr=1e-3，weight decay=1e-4；
- batch size=256；epochs=60；patience=12；gradient clip=5；
- checkpoint 首选macro-AUPR，以weak AUROC打破平局；
- 每标签温度校准和受约束阈值：negative FPR≤0.20。

H2 当前结果：
- macro-AUPR=0.8670±0.0057；
- macro-F1=0.7554；Hamming=0.2356；
- 弱分量召回=0.3774±0.0402；
- weak-vs-negative AUROC=0.6524±0.0349；
- negative FPR=0.1875；
- SawFM AUPR=0.7709±0.0549；
- SawFM weak AUROC约0.5203。

H2 是当前总体六类冻结表示基线，但尚未解决SawFM极弱可见度问题。

4. SawFM 专项特征头：S2_TeacherHardNeg

模型文件：
C:\LJQ_code\CJ_Imbalance_0716\cjlt\models\evidence_readout.py

输入与结构：
- 从冻结F2的stage2提取real/imag [B,32,1500]；
- avg-pool 4倍后拼接为[B,64,375]；
- GroupNorm(8, affine=False)；
- rank-8低秩adapter：64→8→64，adapter residual上限为0.20；
- 证据头：1×1 Conv 64→32、GroupNorm、SiLU、depthwise Conv kernel=15、SiLU、1×1 Conv 32→1；
- 对时间证据图做样本内零均值/单位方差归一化；
- pooling=top-k均值(k=16)+0.25×maximum；
- 正尺度softplus和一个bias；
- 新增参数3,651。

训练数据：
- 28种“移除SawFM后的其他标签组合”；
- 每组合40对SawFM正身份与其他标签完全相同、SawFM缺失的hard negative；
- 共1120对；fit/validation=896/224；
- 10 dB高可见度视图作为teacher；40/50 dB为student；45 dB为独立验证/评价功率；
- identity与receiver_noise_seed严格配对。

S2 总损失：
L = L_cls + L_teacher + L_score + L_map + L_rank

其中：
- L_cls：40/50 dB正样本和匹配负样本的BCE；
- L_teacher：10 dB teacher正样本BCE，系数0.5；
- L_score：40/50与teacher logit的Smooth-L1蒸馏，系数0.25；
- L_map：证据图余弦一致性，系数0.1；
- L_rank：同组合正样本高于hard negative的softplus ranking，系数0.25，margin=0.5。

S2 超参数：
- AdamW，lr=3e-4，weight decay=1e-4；
- batch=64；epochs=30；patience=8；gradient clip=5；
- checkpoint按45 dB validation AUROC选择，AUPR打破平局。

S2 已验证结果：
- 45 dB development AUROC=0.7972，AUPR=0.7949；
- validation-development gap=0.0097；
- S2相对teacher+random-negative的AUROC提高0.1739，3/3 seeds为正；
- hard negative是主要有效机制，teacher一致性单独贡献不稳定；
- 旧2043初始化失败已被重初始化实验定位为优化偶然性；三个新初始化四档平均AUROC为0.9082～0.9249。

5. SawFM 排序接入：C2 固定融合

- 先对S2 expert logit做独立Platt尺度校正；
- SawFM logit：z_C2=0.20×z_H2+0.80×z_S2；
- 其他五类保持H2输出；
- alpha=0.80是预注册固定值，不得按最终种子或功率重选。

C2 已验证结果：
- 普通macro-AUPR=0.8749，相对H2提高0.0078；
- SawFM AUPR=0.8179，相对H2提高0.0470；
- SawFM FPR下降约0.0094；
- 35/40/45/50 dB平均AUROC=0.7638，相对H2提高0.2972；
- 最差功率AUROC=0.6955；四个功率均为3/3 seeds正增益；
- 说明SawFM排序结构已被验证，但普通域阈值迁移到45/50 dB时召回几乎为0，因此它还不是完整可部署决策系统。

6. 当前冻结决策候选：A0，即V2 Visibility Affine

- F2、H2、S2和C2全部冻结；
- 输入10个可部署内部统计：C2/H2/S2 logit、H2-S2差、其他标签logit摘要、S2 evidence top-k/max、stage2幅度均值和时间标准差；
- 不输入真实功率、JNR、raw RMS或生成元数据；
- scale=softplus(Linear(10,1))，bias=Linear(10,1)，共22参数；
- z'=scale(x)×z_C2+bias(x)；
- AdamW，lr=1e-2，weight decay=1e-3；max epochs=300；patience=50；
- 域等权BCE；普通域和40/45/50 dB使用一个统一阈值；hard-negative FPR约束≤0.20。

A0/V2结果：
- 普通SawFM recall=0.7230，FPR=0.2048；
- 40/45/50 dB recall=0.7101/0.5403/0.1668；
- 三档平均recall=0.4724；
- 最大功率FPR=0.1176；
- 平均oracle gap=0.3123；
- 40/45 dB有效，但50 dB是主要瓶颈；
- A0是当前冻结决策对照，但没有通过总晋级门，不能称为最终方法。

7. 已否决的最新方案：A2有序可见度三锚点

- A2成功学习root>40>45>50的可见度坐标；early pairwise accuracy=0.9755，Kendall tau=0.9511；
- 但三档平均recall仅0.0721，40/45/50 dB为0.2028/0.0136/0；
- 即使输入真实功率等级的A3 oracle anchor，三档平均recall也只有0.4074，未达到0.60结构上限门；
- 失败原因是可见度偏置同时作用于弱正和同功率hard negative，压缩了正负概率间隔；
- 结论：可见度顺序可预测，但“可见度→共享正偏置→统一阈值”的映射结构不成立；不得继续通过增加锚点或扩大仿射MLP重复该路线。

四、当前总体目标差距

请在报告中定量分析以下差距：

1. 标签频率长尾：
- Base-DB已证明优于Base-ASL，是可靠基线；
- 但ASL-DB混合改善尾类时破坏macro-AUPR，尚未解决尾类与总体性能的梯度冲突；
- 当前结构创新主要集中在SawFM可见度，对六个标签的频率长尾统一机制证据仍不足。

2. 总体六类识别：
- H2 macro-AUPR已达到0.8670，C2达到0.8749；
- 但所有新方法仍应相对Base-DB和H2报告配对增量、置信区间、Hamming、macro-F1和逐标签AUPR；
- 不能因SawFM提升而接受其他标签显著退化。

3. 通用弱分量识别：
- H2弱召回仅0.3774，weak AUROC为0.6524；
- F2对MFT/VFT/ISRJ有帮助，但SawFM需要专项结构；
- 当前有效创新证据主要是SawFM-specific，尚不能宣称已经解决六类通用弱分量问题。

4. SawFM跨功率排序与决策之间的差距：
- C2四档排序AUROC=0.7638，证明表征/排序有效；
- A0三档平均recall=0.4724，距离目标0.50差0.0276；
- 50 dB recall=0.1668，距离每档0.35目标差0.1832；
- 平均oracle gap=0.3123，距离≤0.15目标差0.1623；
- 普通FPR相对V0增加0.0262，超过允许的0.02；
- 0/3 seeds通过全部决策门槛。

5. SCI证据链差距：
- 已有机制消融、三种子、身份隔离、held-out power和开发集证据；
- 尚缺一个真正通过决策门的完整方法；
- 尚缺方法冻结后的locked-test一次性确认；
- 尚需五种子或更强统计置信区间、逐组合/逐标签稳定性、计算量和参数量比较；
- 若主张“双轴长尾”，必须同时证明标签频率轴与功率可见度轴，并证明二者不是同一现象的重复命名；
- 若最终只解决SawFM，应将论文创新准确限定为“组合匹配监督下的SawFM弱可见度专项识别与可靠决策”，不可夸大为六类通用解决方案。

五、优先级明确的未来方向

请按“最低风险验证→必要结构改变→最终联合微调”规划，不要同时大改主干、损失和阈值。

优先级1：Positive/Negative/Unknown选择性决策

目的：验证50 dB低置信样本是否应被视为unknown，而不是被迫在统一阈值下判为负类。

要求：
- 冻结F2/H2/S2/C2，保留A0分数作为输入或对照；
- 在独立power-threshold身份上选择高阈值t_pos和低阈值t_neg；
- score≥t_pos判positive，score≤t_neg判negative，中间为unknown；
- 同时报告确定正召回、确定正FPR、unknown coverage、unknown中的正样本比例、确定负误杀率、selective risk和coverage-risk曲线；
- 对照A0单阈值、统一双阈值、oracle分功率双阈值；真实功率只能用于oracle上限，不能作为部署输入。

建议晋级门：
- 确定正hard-negative FPR≤0.20；
- 50 dB的“确定正召回+unknown正覆盖”≥0.60；
- 确定负对真实正样本的误杀≤0.10；
- 普通域coverage和macro-AUPR不得出现不可接受退化；
- 至少2/3 seeds通过，locked test继续冻结。

优先级2：标签条件的“弱证据/缺失证据”判别头

仅当选择性决策仍不能区分弱正与hard negative时启动。

建议：
- 从F2 stage2或S2 evidence map输入，而不是只使用10维全局统计；
- 使用SawFM标签条件query，在时间轴搜索局部SawFM证据；
- 保留无注意力mean/std和top-k安全旁路；
- 正样本为同身份弱SawFM视图，负样本为其他标签组合完全匹配但SawFM缺失的hard negative；
- 可增加evidence existence和evidence reliability两个输出，而不是直接增加SawFM logit偏置；
- 使用正/负判别损失、matched ranking和证据图一致性；避免再次训练纯功率等级分类器。

优先级3：从校准转向似然比或可靠性建模

- 比较P(Saw present|evidence)与P(evidence reliable|view)，不要用一个共享偏置承担两个任务；
- 可考虑两阶段判决、energy/likelihood ratio、selective calibration或conformal risk control；
- 所有方法必须在相同FPR和coverage条件下比较；
- 不得读取JNR、真实功率或数据生成元数据作为部署特征。

优先级4：S2稳定性和集成

- 每个数据seed固定训练协议，至少3个初始化；
- checkpoint只能由原validation role选择；
- 比较单checkpoint、logit均值ensemble、evidence-map均值ensemble；
- 报告性能方差、推理成本和最差组合；
- 不能根据early-development选择初始化。

优先级5：向其他弱标签扩展

- 先做oracle可辨识性筛选，再决定是否为MFT/VFT/ISRJ建立专项头；
- 每个标签使用物理匹配的正负构造，不要直接复制SawFM损失；
- 只有在至少两个新增标签上出现稳定增益，才能支持“通用多标签弱证据框架”的论文表述。

优先级6：最后stage小学习率联合微调

只有冻结方法通过预注册决策门后才允许：
- 冻结stage1～stage3；
- 最后stage学习率≤1e-5；
- 专家/决策头学习率约1e-4；
- 使用L2-SP或参数漂移约束；
- 保留冻结C2/A0作为并行对照；
- 同时监控普通macro-AUPR、弱AUROC、SawFM跨功率AUROC、50 dB recall、FPR和representation drift；
- 若两个连续评价点未超过冻结方案，立即停止，不得以更长训练掩盖结构失败。

六、明确禁止重复的低价值方向

- 不继续扩宽整个F2主干，除非新的oracle证据证明现有stage2/stage4没有目标信息；
- 不重复ASL-DB 0.75/0.90附近的盲目权重搜索；
- 不使用统一batch内top-k难负排序损失覆盖所有六类；
- 不将共享深层特征的“去相位分支”再次包装为独立SawFM专家；
- 不继续增加三锚点数量或扩大可见度仿射MLP；
- 不通过无约束降低阈值制造高召回；
- 不在early-development上选择checkpoint或超参数；
- 不把配对功率视图计为独立样本；
- 不在冻结方案未通过前解冻主干；
- 不使用locked test反复调参。

七、你需要输出的内容

请输出一份专业中文技术报告，结构必须包括：

1. 执行摘要：用不超过500字说明目前做到什么程度、最强结果、核心失败和下一步首选方向。
2. 试验演进表：按“阶段—方法—关键指标—验证结论—是否晋级”总结Base-DB、F2、H2、S2、C2、A0、A2。
3. 当前网络图：用清晰的层级结构描述[B,2,3000]→F2→H2/S2→C2→决策层，并标明冻结/可训练模块、张量尺寸和参数量。
4. 超参数与损失函数表：分别列出F2、H2、S2、A0的optimizer、学习率、weight decay、batch、epoch、patience、checkpoint选择规则、损失组成和阈值规则。
5. 已验证事实：只列能够被正式JSON、预测或receipt支持的结论。
6. 失败实验与原因：区分表征失败、优化失败、分类边界失败、校准失败和指标/阈值放大。
7. 目标差距矩阵：列出当前值、目标值、绝对差距、主要瓶颈和所需证据。
8. SCI创新性审视：说明当前哪些内容可作为方法创新、哪些只能作为负结果或机制分析、还缺少哪些证据链。
9. 三阶段未来计划：每阶段只能解决一个主要问题，给出对照组、消融、数据角色、训练参数、评价指标、晋级门、失败后停止规则和预计产物。
10. 首选下一实验的预注册方案：优先设计positive/negative/unknown选择性决策实验；给出能够直接实现的伪代码、输出JSON字段、目录结构和命令模板。
11. 风险清单：列出数据泄漏、checkpoint选择混杂、功率元数据捷径、seed不稳定、阈值过拟合、SCI过度表述等风险及审计方法。
12. 最终决策：只能从以下选项中选择并说明证据：
   A. 继续冻结并执行selective unknown；
   B. selective失败后训练stage2标签条件证据头；
   C. 冻结方法通过后进行最后stage微调；
   D. 停止SawFM专项线，回到Base-DB/H2作为最终系统。

写作规则：
- 所有指标注明数据角色、种子数量和locked_test_used状态；
- 将“排序/AUROC改善”和“固定阈值召回改善”分开陈述；
- 将“开发集验证”和“最终测试确认”分开陈述；
- 不得把proposal写成validated result；
- 不得事后降低已注册门槛；
- 对不同协议下的指标不要进行未经说明的直接横向比较；
- 每个建议必须给出可证伪预测：如果机制正确，哪些指标应怎样变化；如果没有变化，应停止什么方向；
- 最终给出最小可执行实验，而不是一次同时修改主干、损失、分类器和阈值。
```

## 推荐使用方式

如果希望下一位智能体不仅分析，而且直接实施，可在上述 Prompt 后追加：

> 在完成证据复核和预注册文档后，使用 PY10 实现第一优先级实验。先运行单种子小规模 smoke，确认身份隔离、阈值逻辑、输出字段和 locked-test receipt；通过后运行 2041/2042/2043 三种子正式实验。保存配置、源码版本、输入和 checkpoint SHA256、identity manifest、训练历史、预测、阈值搜索、coverage-risk 数据、逐功率指标、summary JSON 和中文结果报告。若预注册晋级门失败，不自动进入下一阶段。
