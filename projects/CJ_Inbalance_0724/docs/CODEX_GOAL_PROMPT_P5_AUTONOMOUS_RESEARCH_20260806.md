# Codex Goal Prompt：P5 路线自动研究与创新证据闭环

将下面整段提示词复制到 Codex，并要求创建 Goal。不要删减证据边界、停止规则或 locked-test 限制。

```text
请创建并持续执行一个长期 Goal，目标如下：

在 E:\LJQ_code\CJ_Inbalance_0724 内，以冻结的 P5e 五模型 JNR-only 附件为当前候选，自动完成执行链路修复、三全新身份池开发、两全新身份池确认、输出严格等价的成本验证，以及形成可靠创新点仍缺少的 OOD、组合不相交、多轴长尾和安全性证据。整个过程中严格执行一次只改变一个核心机制、预注册、方法冻结、新身份验证、失败归因和证据边界管理。最终目标是在不读取 locked test 的前提下，形成 E0、P5e、P5-final 及必要消融的完整预注册与证据包；到达 locked-test 授权门槛后停止并向用户申请一次性明确授权。

一、工作位置与运行环境

1. 唯一活动项目：E:\LJQ_code\CJ_Inbalance_0724。
2. 数据和旧模型源码只从 E:\LJQ_code 下解析。旧 C/D/F 路径仅可作为冻结历史证据，不得用于当前文件访问。
3. Python 使用 D:\anaconda3\envs\PY10\python.exe；MATLAB 使用 D:\Matlab\bin\matlab.exe。
4. 长 GPU 实验必须使用独立隐藏监督进程、原子回执、分块检查点和 CUDA 独占守卫。没有终态回执时只能标记为 running/uncertain，不得判断成功或模型失败。
5. 保留全部已有文件、失败结果和负结果；不得删除、覆盖或重写冻结证据。使用新文件名和新的实验编号继续。

二、当前确认状态

1. P1n 的 full-12 predicted-gate NMAE 由 0.72879 降到 0.67814，约改善 6.95%，但仅是 reference-only。
2. P5e 是当前最强的有限正结果：五个冻结模型、两个全新身份池上，JNR-only NMAE 约改善 MFT 52.8%、VFT 53.8%、ISRJ 46%。它不能被表述为分类、全 12 参数、OOD、组合不相交、长尾或实际加速结论。
3. P7a 已关闭“直接由 IQ 分组卷积产生主要收益”的因果主张；P2w 的同可观测量 VFT 深化路线不足；P3o-U、P8b、P8c 已因响应/分离或安全门槛失败而关闭，不得在已消费身份上复活。
4. P5p、P5q、P5r 分别因粗粒度中断、旧 C 盘审计路径和交互会话无终态而停止，其身份永久隔离。
5. P5s 成功完成首池全部 30 个解析特征块，证明独立监督和分块回执有效；随后在 source model 3041 加载前失败：
   FileNotFoundError: Legacy model source is missing: C:\LJQ_code\pythonProject\新建文件夹\model.py
   P5s 没有模型输出、等价性或延迟结果，必须冻结为路径可迁移性失败。
6. 正确的现有源码位置至少包括：
   E:\LJQ_code\pythonProject\新建文件夹\model.py
   SHA-256 = 406e5868ff71c1f038f5b43dd7bbbf40f740dd6ca04bddbb1bab8e5a70ac043d
   E:\LJQ_code\pythonProject\新建文件夹\DBmodel.py
   SHA-256 = a0e5aee08561cfc5a9d70b64755ecd551df1bfc3b89bc64fc7821102ac6a5de3
7. 当前硬编码位于 E:\LJQ_code\CJ_Inbalance_0724\cj0724\models.py 的 LEGACY_SOURCE_ROOT。先检查工作区当前内容与哈希，不要假设文件未被用户修改。
8. P5s 已消费身份种子 3510000000、3520000000、3530000000；不得重跑、恢复、重审或用于后续选择。P5s confirmation 身份也不得提前生成。
9. locked test 尚未授权，禁止生成、读取、扫描、统计或间接选择。

三、立即执行的 P5t 单机制修复

本轮唯一允许改变的核心轴是 legacy 冻结模型源码的路径可迁移性。不得同时修改模型结构、权重、阈值、normalizer、解析特征、batch、warm-up、计时重复数、损失、数据采样或延迟门槛。

按顺序执行：

1. 审计 P5s 的 supervisor_terminal.json、attempt_000_abort.json、30 个解析块回执、stderr 和进程状态；创建 P5s outcome-freeze JSON/Markdown，明确它是“解析完成后、模型加载前的旧路径失败”，无性能结论。
2. 搜索 E:\LJQ_code\pythonProject\新建文件夹 下 P5e 构建真正依赖的所有 legacy Python 源文件，包括 model.py、DBmodel.py、EffNet.py、EffNetV1.py、EffNetV2.py、EffNetV3.py 及其局部依赖。生成路径、大小、SHA-256 和导入依赖清单。
3. 检查冻结 checkpoint、protocol.json、summary.json 和模型 source 名称，证明 P5e 实际只需要哪些 legacy 文件。不要因为目录中存在文件就假设全部参与模型构造。
4. 设计 P5t 预注册。推荐实现是从 cj0724/models.py 所在位置推导 E:\LJQ_code 根目录，并读取一个显式 SHA-256 锁定的 legacy-source manifest；禁止重新复制或重写 legacy 模型源码。若环境变量仅作为可选覆盖，必须验证其绝对路径和哈希，默认路径仍必须可迁移。
5. 添加无数据、无 CUDA 的测试：
   - 从无关 cwd 导入 legacy_mlcnn；
   - 验证实际解析路径位于 E 盘；
   - 验证所有必需源码哈希；
   - 构造与 P5e protocol 相同的模型；
   - 加载五个冻结 checkpoint 的 state_dict，要求 key、shape 和 dtype 全部匹配；
   - 证明没有访问 C/D/F 路径；
   - P5s 分块检查点和独立监督回归测试继续通过。
6. 运行相关 pytest 回归、显式 protocol validation 和 MATLAB dry-run。全部通过后才创建 P5t method-freeze。
7. P5t 使用三个全新 uint32 train-only 身份池。建议开发种子为 3410000000、3420000000、3430000000，确认种子为 3440000000、3450000000；生成前必须扫描历史配置，若任一种子已出现则自动选择新的未使用 uint32 种子并记录理由。
8. 每个新池仍为 10/15/20/25/30 dB、63 个组合、每组合 60 样本；所有功率视图按 receiver_noise_seed 绑定，不得当成独立样本。
9. 使用显式 E 盘基线和全部本地历史 train MAT 做审计；unreadable=0、overlap=0、五功率身份对齐、mask/non-JNR 参数一致、JNR 契约全部通过后才能启动 GPU。
10. 复用 P5s 的隐藏独立监督器和 630-position 解析块机制。P5t 运行必须写入 supervisor start、child PID、environment、每解析块、每模型、每池、restore 和 terminal 回执。

四、P5t 开发与确认硬门槛

开发阶段三个新身份池、五个冻结模型全部满足：

1. full-pool logits、概率、hard decisions、base parameters、dense-head values、candidate parameters 与非目标参数逐元素严格等价。
2. 固定 benchmark batch 为索引 0..125，禁止数据依赖采样。
3. batch size=126，warm-up=5，timed repetitions=15。
4. shared-dense/reference 的 median latency ratio 对每个数据池和每个模型均 <=0.85。
5. 每个 CUDA 守卫都没有外部 Python CUDA compute PID；Windows 无权限设置 EXCLUSIVE_PROCESS 时只能使用预注册的 P5h 守卫降级方案。
6. supervisor、子进程、restore、pool 和最终 result/abort 回执完整。
7. test_generated=false、test_opened=false、locked_test_used=false。

任一门槛失败：立即冻结 P5t，隔离全部 P5t 身份，不调 batch、编译器、重复数、模型、阈值、代码路径或数据后重跑。

仅当三个开发池全部通过，才生成两个全新确认池，使用相同实现和门槛一次性确认。确认失败同样冻结，不回调。

五、P5t 之后的自动研究路线

若 P5t 开发和确认均通过，只能形成窄结论：P5e 的 shared frozen-forward + dense JNR attachment 在固定 batch 上输出严格等价并具有已确认的延迟下降。不得扩大为分类或通用加速结论。

随后继续执行以下证据路线，每轮只改变一个机制并使用全新身份：

A. P5-U：P5e JNR-only 的 OOD 与模板失配泛化
- 新功率范围、参数边界外和模板失配；
- 五模型、三开发种子、两确认种子；
- 报告 MFT/VFT/ISRJ 各自 NMAE、per-power、per-combination、worst cell 和 bootstrap CI；
- 分类 logits、阈值、非目标参数必须保持不变。

B. P5-CD：composition-disjoint 泛化
- 训练/开发/确认的组合身份严格不相交；
- seen/unseen combination gap、hardest unseen combinations 和组合大小分层；
- 不得把同一 receiver_noise_seed 的功率视图拆到不同角色。

C. P5-LT：多轴长尾安全验证
- 标签频率、功率/弱分量和组合复杂度三轴分别控制；
- 必须有 matched compute/control；
- 报告 macro/micro AUPR、subset accuracy、per-class recall、Hamming loss、困难负样本 FPR、separation、参数 NMAE 和最差单元格；
- recall 提升但 AUPR、分离度或困难负样本 FPR 恶化时，归类为响应放大并停止。

D. P5-COST：完整成本审计
- 参数量、MAC、峰值显存、解析前端耗时、完整池耗时、固定 batch p50/p90/p95、监督/检查点开销；
- 把“实际推理延迟”与“离线证据生成时间”分开报告；
- 不得只报告理想 lazy-call 次数。

每条路线先做三种子开发筛选，再用两个全新种子确认。只有同时满足五模型一致性、统计检验、安全门槛和成本门槛，才可进入 P5-final 候选。

六、创新点整理要求

最终创新点必须按证据强度写成：

1. 已证明：五模型、多新身份池、输出/分类不变条件下的 bounded JNR-only correction。
2. 已证明或否定：输出严格等价的 shared frozen-forward 执行加速。
3. 若通过才写：对 OOD、组合不相交和三轴长尾具有稳定泛化。
4. 负结果边界：IQ 分组直接因果、同可观测量 VFT 深化、P3o-U 和 P8b/P8c 不再作为正创新点。
5. 每项创新都列出 matched control、种子数、身份角色、统计检验、最差单元格和成本证据。

七、自动执行与汇报规则

1. 不要只给计划；持续读取现场状态、实现、测试、生成全新数据、审计、启动独立实验、监控回执、分析结果并推进下一阶段。
2. 长任务通过独立监督进程运行。每次 Goal 续跑先审计已有 PID、回执和终态，禁止因为交互回合结束而重复启动。
3. 每轮更新计划状态，并把证据分为：reference-only、fresh-confirmed bounded result、stopped/failed、running/uncertain、not validated。
4. 失败必须区分：模型失败、数据审计失败、协议失败、路径/依赖失败、CUDA 冲突、外部终止和无终态；不得合并为总体失败。
5. 所有预注册、方法冻结、数据审计、launch manifest、terminal receipt、结果和 outcome-freeze 都保存到 E:\LJQ_code\CJ_Inbalance_0724 的 configs/docs/outputs 对应目录。
6. 不得在 consumed 或 locked 数据上调参。不得为通过门槛而改变阈值、重复数、batch、模型深度或统计方法。
7. 不进行 destructive deletion，不修改原始 MAT，不覆盖用户现有工作。
8. 如果发现工作区内容与本提示词记录不一致，先做只读核验，以现场文件和冻结哈希为准，并记录差异。

八、Goal 完成和停止条件

只有满足以下条件才可将 Goal 标记 complete：

1. P5-final 在五模型、三开发身份池和两个全新确认身份池上满足预注册的效果、安全和成本门槛；
2. OOD、组合不相交、多轴长尾和完整成本审计均有明确通过或否定结论；
3. E0、P5e、P5-final 和必要消融已冻结；
4. locked-test 预注册已经生成并完成哈希审计；
5. 尚未读取 locked test，并已向用户明确申请一次性授权。

不要因为任务耗时、上下文压缩、token 接近预算或单轮结束而把 Goal 标记 complete。仅当同一外部阻塞条件连续出现至少三次且无法安全推进时，才按系统规则标记 blocked。在获得用户 locked-test 明确授权前，绝不读取 locked test。
```

