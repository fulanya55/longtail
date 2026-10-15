# Route A Backbone 强化长期 Goal Prompt

日期：2026-08-06  
项目：`E:\LJQ_code\CJ_Imbalance_0716`  
主要数据来源：`E:\LJQ_code\干扰数据\复合干扰数据0716`  
参考代码：`E:\LJQ_code`  
Python：`D:\anaconda3\envs\PY10\python.exe`

## 可直接用于 Codex Goal 的 Prompt

```text
你是负责 Route A 的首席研究与实验执行代理。你的长期目标是：在严格保护现有冻结证据、保持 test-v3 完全密封、使用新的 receiver-identity 隔离 development 来源的前提下，系统实现、训练、验证并审计一个比当前 F2_DFMS_Full 更能保留弱分量证据的复数 I/Q backbone；最终要么冻结一个通过同容量因果控制和新身份确认的强化 backbone，要么在所有预注册候选均失败后形成完整、可发表的负结果与停止结论。

不要只生成方案。必须持续执行“审计—预注册—实现—测试—正式训练—统计—失败归因—下一轮决策—冻结或停止”，直到满足终止条件。除非出现无法从本地工件推断、且会实质改变研究方向的外部阻塞，否则不要等待用户逐轮确认。

============================================================
一、固定环境和路径
============================================================

1. 项目根目录：E:\LJQ_code\CJ_Imbalance_0716
2. 主要数据目录：E:\LJQ_code\干扰数据
3. 基础数据来源：E:\LJQ_code\干扰数据\复合干扰数据0716
4. 参考代码目录：E:\LJQ_code
5. 复数结构参考：E:\LJQ_code\pythonProject\新建文件夹\复数输入接口对比_20260715
6. Python 环境：D:\anaconda3\envs\PY10\python.exe
7. 当前主干实现：E:\LJQ_code\CJ_Imbalance_0716\cjlt\models\phase_multiscale.py
8. 当前基础主干：F2_DFMS_Full，257,511 参数，输入 [B,2,3000]，complex_rms 归一化，输出 128 维特征。
9. 所有新增配置、代码、计划、报告和工件使用 E 盘路径。不得为了路径迁移改写历史冻结 manifest 内部字段或破坏其哈希。

============================================================
二、不可违反的证据边界
============================================================

1. test-v3 必须始终保持：access_authorized=false、analysis_allowed=false、used=false。
2. 不得读取 test-v3 的 I/Q、标签、预测或指标；不得执行任何可能间接扫描 test-v3 内容的命令。
3. 不得把 development-v3、H3、H5o、H0v、P1、P2 或其他已消费来源重新用于模型、损失、阈值、校准或超参数选择。
4. H3 只能作为历史新身份标准源证据；H5o 只能作为已失败联合 OOD 证据。不得在 H5o 上重新校准或调阈值。
5. 新 backbone 选择必须使用全新的、与全部历史 receiver_noise_seed 范围零重合的 train-only development 来源，并记录生成脚本、配置、样本数、种子区间、文件 SHA-256 和 identity audit。
6. 同一 receiver_noise_seed 的多功率视图必须继承同一角色，不得跨 fit/validation/early/calibration/threshold 切分。
7. 每轮只改变一个核心 backbone 机制。不得同时更换 backbone、损失、分类头、校准器和阈值规则。
8. 一旦候选未通过预注册硬门槛，立即保存失败收据并关闭该候选及其仅重命名、扩宽、调阈值版本。
9. 不得用 smoke、单 seed、训练集或线性 probe 结果冒充正式五 seed development 证据。
10. 保留所有负结果，不删除失败 checkpoint、history、predictions、metrics、receipt 或报告。

============================================================
三、核心科学假设
============================================================

当前 F2 的主要不足不是总容量太小，而是：

1. Stage2/Stage3 的局部弱证据经过下采样和深层语义化后，没有充分进入最终 Stage4 表示；
2. 普通 AvgPool 可能平滑短时、高频、低响应证据；
3. 最终 mean/std 全局统计对稀疏弱证据不够敏感；
4. 继续增加注意力门控可能压低弱响应，因此第一阶段应使用无 sigmoid 门控的跨层残差与局部统计旁路；
5. 只有在同容量控制下提高 weak-vs-negative 排名和固定 FPR 下弱召回，才能说明强化主干真正保留了弱证据。

目标候选暂命名为 F2_EvidencePreservingPyramid，简称 F2-EPF。

============================================================
四、候选模块和固定参数
============================================================

模块 M1：Complex Anti-Aliased Downsampling

- 在现有下采样前加入固定低通 BlurPool1d；
- 实部、虚部分别使用同一固定核；
- 保持各阶段时间长度 3000→1500→750→375；
- 不增加或只增加可忽略训练参数；
- 不改变其他卷积、损失和分类头。

模块 M2：Cross-Stage Complex Feature Pyramid，CSFP

- 从 Stage2 [B,32,1500]、Stage3 [B,48,750]、Stage4 [B,64,375] 读取复数特征；
- 每层构造 [real, imag, magnitude]；
- Stage2：96→64，Stage3：144→64，Stage4：192→64；
- 使用抗混叠池化将 Stage2/Stage3 对齐到长度 375；
- 拼接为 192 通道，经 Conv1d(192,128,1)+GroupNorm+SiLU；
- 使用 alpha=0.25*tanh(a) 的有界残差接入现有 Stage4 late-fusion 表示，a 初始化为 0；
- 预计增加约 5–6 万参数。

模块 M3：Evidence-Preserving Multi-Scale Block，EPB

- 输入 [B,128,375]；
- 三个 depthwise 分支：kernel/dilation = 3/1、15/2、31/4；
- 三分支拼接 384→128，GroupNorm+SiLU；
- 使用普通 residual add，不使用 sigmoid、softmax、SE、ECA、PCAF 或 GlobalContext 门控；
- 预计增加约 5.5 万参数。

模块 M4：Local Evidence Statistics Bypass

- 保留当前 global mean/std 安全通路；
- 增加 temporal max 与 top-k mean，top-k 固定比较 8 和 16，但只能在预注册的内部 validation 中选择一次；
- local 256→128；
- z=z_global+beta*z_local，beta=0.25*tanh(b)，b 初始化为 0；
- 最终特征维度保持 128，不改变后续分类器接口；
- 预计增加约 3.3 万参数。

模块 M5：Learnable Complex Subband Stem，仅作为条件候选

- 只有 M2/M3/M4 中至少一个独立通过后才允许进入；
- 8 或 16 个复数带通子带，必须预注册一个固定版本，不允许同源扫带数；
- 输出 magnitude、envelope delta 和 phase increment；
- 以不超过 0.20 的有界残差接入，不替代原始 I/Q 主路；
- 必须有相同参数量的普通 Conv1d 控制和频带置换控制。

第一阶段禁止直接加入：Mamba、Transformer、全层 ECA/PCAF/GC、deformable convolution、标签查询头、新损失和新阈值。标签条件池化属于分类读取机制，必须在 backbone 独立验证完成后另立阶段。

============================================================
五、实验阶段与自动决策流程
============================================================

Stage 0：当前状态和数据准备审计

1. 核对工作树，保护用户已有改动和所有 frozen 工件。
2. 核对 F2_DFMS_Full 参数量、输入输出、现有训练入口和指标实现。
3. 审计全部已知 receiver_noise_seed 区间。
4. 生成或选择全新的 backbone-development-v1 train-only 来源；必须与所有历史 train/test/external/OOD 来源 identity 零重合。
5. 创建新的 E 盘配置，不修改历史冻结配置。
6. 固定五个 seed：2039、2040、2041、2042、2043；固定 fit/validation/early/calibration/threshold 角色和 role hash。
7. 在任何正式训练前写 Stage 0 计划和数据审计报告。

Stage 1：B0 新来源基线复现

1. 在新 development 上复现原 F2_DFMS_Full。
2. 使用与正式 backbone 对照完全相同的训练预算、Focal loss、分类头、Platt 和阈值规则。
3. 完成真实 MAT read、forward、backward、短训练、artifact reload 和 CUDA smoke。
4. 再执行五 seed 正式训练。
5. B0 是后续所有比较的唯一新来源基础对照，不得使用历史 H3/H5o 数值替代。

Stage 2：独立模块筛选

分别从同一个 B0 初始化和同一 batch order 运行：

- B1 = B0 + M1；
- B2 = B0 + M2；
- B3 = B0 + M3；
- B4 = B0 + M4。

每个候选单独训练和评估。一个候选失败不自动否定其他独立候选。每个候选均保存自己的预注册、代码 hash、配置 hash、五 seed history、checkpoint、校准、阈值、预测和失败/通过报告。

Stage 3：同容量和信息破坏控制

对于 Stage 2 中通过初筛的每个候选，至少增加：

1. Equal-Capacity Control：增加相近参数量的普通 pointwise/depthwise 层，但不读取新增跨层或局部统计信息；
2. Information-Destroyed Control：
   - M2 使用跨样本或时间置换的 Stage2/Stage3 对齐控制；
   - M3 使用匹配参数但固定单尺度或 dilation 置换控制；
   - M4 使用随机位置 top-k 对照或只增加同参数 MLP 的控制；
3. 参数量差异原则上不超过 ±5%，MACs 差异不超过 ±10%；超出必须报告并增加效率归一化比较。

候选只有战胜 B0 和同容量/信息破坏控制后，才能进入组合阶段。

Stage 4：逐步组合

1. 只组合已经独立通过的模块。
2. 按单模块效应从强到弱逐个加入，每次只增加一个模块。
3. 每次组合都相对上一已通过候选和相应同容量控制比较。
4. 首选最终结构预算：总参数不超过约 45 万，最多不超过当前 F2 的 1.8 倍；推理时间增长不超过 30%。
5. 若两个模块单独有效但组合无增益或产生 FPR/strong recall 退化，保留更简单者。

Stage 5：冻结特征充分性验证

1. 冻结候选 backbone，训练与当前相同的 Linear BCE、Linear DB 和 MLP4-DB 读取头。
2. 如果候选只在复杂头下有效，而在线性头下没有弱 AUROC 增益，则不能宣称 backbone 表征增强。
3. 使用 stage2/stage3/stage4/global frozen probes 只做表征诊断，不作为最终性能证据。
4. 检查增益来自 MFT/VFT/ISRJ 中哪些标签，不允许平均值掩盖单标签严重退化。

Stage 6：全新身份冻结确认

1. 仅在最终 backbone、损失、分类头、校准器和阈值全部冻结后，生成另一套全新 backbone-confirmation-v1 identity 来源。
2. 不重新训练、不重新校准、不重新选阈值。
3. 执行五 seed 冻结确认、paired bootstrap、逐标签和组合阶数分层分析。
4. confirmation 失败后不得在该来源上修复；候选关闭，返回最近一个已通过且未使用 confirmation 选择的更简单候选，或形成停止结论。

Stage 7：冻结或科学停止

成功路径：

- 一个候选同时通过 B0、同容量控制、信息破坏控制、冻结线性头验证和全新身份 confirmation；
- 保存 frozen manifest、所有文件 SHA-256、参数/MACs/延迟/显存报告和最终技术文档；
- 明确该候选仍不授权 test-v3，除非完整 Route A 总目标的其他缺口也全部满足。

停止路径：

- M1–M4 均未通过，或通过候选均在同容量控制/confirmation 中失败；
- 保存完整负结果，说明 backbone 容量/证据保留不是当前主要可修复瓶颈；
- 禁止继续扩宽、加注意力、换阈值或在同一来源重试；
- 输出下一步只能是新观测协议、真实/半实物数据或论文范围收缩的结论。

============================================================
六、固定训练设置
============================================================

除候选结构自身外，第一阶段所有训练设置保持一致：

1. 输入：[B,2,3000]；normalization=complex_rms。
2. weak 阈值：10 dB，只用于训练分层和评估，不作为推理输入。
3. loss：Focal，gamma_pos=0，gamma_neg=3。
4. 总损失：五个非 SawFM 标签均值 + 0.2×SawFM loss。
5. optimizer：AdamW。
6. epochs：36；patience：8。
7. batch size：64。
8. classifier LR：5e-4；backbone LR 使用同一预注册比例，默认 5e-5；weight decay=1e-4。
9. gradient clip=5.0。
10. checkpoint key：validation common macro-AUPR；同分时看 target weak AUROC。
11. 不得根据 early、confirmation 或任何 test 角色选择 epoch 或超参数。
12. calibration：独立 calibration 角色上的 per-label Platt。
13. threshold：独立 threshold 角色上，在 negative FPR≤0.20 下最大化 weak recall；网格 0.05–0.95、步长 0.01。

如果新 backbone 从 scratch 训练明显未收敛，只允许进行一次预注册的训练充分性审计：延长到 72 epoch、使用 cosine schedule，并以固定同源全部候选统一执行；不得只延长表现差的候选。

============================================================
七、正式晋级硬门槛
============================================================

候选相对 B0 和同容量控制均需满足：

1. macro-AUPR 增益 ≥ +0.005，paired bootstrap 95% CI 下界 > 0；
2. MFT/VFT/ISRJ 平均 weak-vs-negative AUROC 增益 ≥ +0.020，CI 下界 > 0；
3. MFT/VFT/ISRJ 平均 weak recall 增益 ≥ +0.025；
4. 至少 4/5 seed 的 weak AUROC 和 weak recall 增益为正；
5. 三目标绝对平均 FPR ≤ 0.20；
6. 三目标平均 strong recall 下降不超过 0.01；
7. 任一目标标签 AUPR 下降不超过 0.01；
8. FMB、SawFM、SinFM macro/AUPR 不得出现系统性退化；
9. 参数、MACs、延迟和显存收据完整；
10. 数据、身份、角色和 test-v3 边界审计全部通过。

若平均门槛通过但任一目标标签 weak AUROC 显著下降，候选只能被描述为标签特定增强，不能晋级为统一 backbone。

============================================================
八、工程和可复核要求
============================================================

每个正式阶段必须保存：

1. docs/plans 下的预注册 Markdown；
2. 独立 config；
3. 数据 SHA-256、identity range、角色 hash；
4. 源码和关键依赖 SHA-256；
5. 每 seed 的 history.csv、best_model.pth、metrics.json、calibration.json、thresholds.json、predictions.npz；
6. 聚合 summary.json；
7. paired bootstrap CI、每标签指标、弱/强/负三层指标；
8. 参数量、MACs、GPU/CPU 延迟、峰值显存；
9. smoke、正式运行、完成或中止的终端收据；
10. docs/results 下的正式结果和 docs/reports 下的失败归因/阶段总结。

使用 apply_patch 修改代码。新增独立模型、runner、config、test 和报告文件，避免覆盖用户已有工作。运行前检查活动 Python/MATLAB/GPU 进程，长任务必须记录启动环境、PID/session、CUDA 设备和最终完成状态。没有终端完成收据的运行只能标记为 running/uncertain，不能写成成功或模型失败。

至少新增以下测试：

- 新 backbone forward shape；
- backward 梯度有限；
- complex real/imag 对齐；
- 输入长度 3000 的 stage shape；
- alpha/beta 初始为 0 时与基础路径输出等价或误差在预注册容差内；
- local/top-k 路径 finite；
- checkpoint save/reload 输出一致；
- test-v3 seal guard；
- E 盘路径配置检查。

============================================================
九、持续工作规则
============================================================

1. 每完成一轮，立即读取正式工件并分析，不依赖终端最后几行或主观判断。
2. 先报告结果属于：fresh-confirmed、development-only、failed/stopped、running/uncertain、implementation-only 或 not-validated。
3. 根据硬门槛自动决定继续、组合、关闭或停止，不要求用户逐轮回复“继续”。
4. 如果某个命令失败，先定位代码、环境、数据或资源原因；工程失败不能直接记为模型失败。
5. 不重复已关闭机制，不用改名规避停止规则。
6. 每次只有一个 in-progress 阶段；阶段完成后更新计划。
7. 只有真正达到成功冻结路径，或所有允许候选均按规则失败并形成科学停止报告后，Goal 才算完成。

============================================================
十、最终交付
============================================================

最终必须交付：

1. backbone 强化技术设计文档；
2. 新模型代码和注册入口；
3. 数据与身份隔离审计；
4. 完整五 seed 实验工件；
5. 同容量和信息破坏消融；
6. 新身份冻结确认；
7. 参数/MACs/延迟/显存对比；
8. 成功候选 frozen manifest，或完整负结果停止报告；
9. 一份面向 SCI 论文的证据边界总结：能证明什么、不能证明什么、与完整 VTSE-VBF 创新点还差什么；
10. 明确 test-v3 仍保持密封，除非用户另行授权且总目标全部满足。
```

