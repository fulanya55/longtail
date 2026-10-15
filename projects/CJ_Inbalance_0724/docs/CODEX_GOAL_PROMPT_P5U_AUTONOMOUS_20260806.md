# Codex Goal Prompt：P5u 证据闭环与后续泛化验证

将以下文本原样交给 Codex，并要求它创建并持续执行一个 Goal。

```text
请创建并持续执行一个长线 Goal。目标是在 E:\LJQ_code\CJ_Inbalance_0724 内，以已冻结的 P5e 五模型 JNR-only 附件为唯一候选，先完成 P5u 的输出严格等价、CUDA 隔离和实际端到端成本证据闭环；只有通过五个全新身份池的开发与确认后，才依次推进 OOD/模板失配、组合不相交和多轴长尾安全验证。最终形成可复核的创新证据包与 locked-test 预注册；在获得用户明确的一次性授权前，绝不读取、扫描、生成或间接用于选择 locked test。

## 0. 强制运行边界

1. 唯一活动项目是 `E:\LJQ_code\CJ_Inbalance_0724`。C/D/F 盘历史路径只能作为既有报告证据，不能作为当前文件访问来源。
2. Python 固定为 `D:\anaconda3\envs\PY10\python.exe`；MATLAB 固定为 `D:\Matlab\bin\matlab.exe`。
3. 保留全部已有代码、配置、MAT、结果、冻结文件、失败结果与回执；禁止删除、覆盖或改写历史证据。新工作必须使用新编号和新文件名。
4. 每一轮只能改变一个核心机制。每轮开始前必须有预注册、方法冻结、数据身份与哈希审计；失败后冻结该轮及其身份，不得通过改 batch、阈值、重复数、模型深度、采样、损失权重或统计口径回调重跑。
5. 任何结论必须标注为 `reference-only`、`fresh-confirmed bounded result`、`running/uncertain`、`stopped/failed` 或 `not validated`；不得合并不同数据角色的结果。
6. 不得修改正在运行的 P5u supervisor、executor、monitor、final-evidence-freezer 的源码、配置或 manifest；不得启动重复 GPU 任务。每次续跑前先只读审计 PID、父子关系、checkpoint/terminal receipt 和已存在结果。
7. 禁止访问 locked test：不得生成、打开、列目录、扫描、统计、校准、选阈值或以任何方式用于模型选择。到达 locked-test 门槛后必须停止并向用户申请明确授权。

## 1. 已冻结的科学边界

1. P5e 是当前最强的正向证据：五个冻结模型、两个新身份确认池上的 JNR-only NMAE 改善约为 MFT 52.8%、VFT 53.8%、ISRJ 46%。它不证明分类、全 12 参数、OOD、组合不相交、长尾或实际加速。
2. P7a 已关闭“IQ 分组卷积本身导致主要收益”的归因；P2w 不支持在同一 Doppler/phase 家族继续深挖 VFT；P3o-U、P8b、P8c 是已冻结负结果，不得复用其身份或作为正向创新。
3. P5p/P5q/P5r/P5s/P5t 的身份永久隔离，不得重跑、复活或用于后续选择。
4. 当前唯一活动执行是 P5u：冻结五模型 JNR-only dense attachment 的 `shared frozen-forward` 与 `reference` 逐元素输出等价，并验证实际延迟比。它不是通用加速或分类结论。

## 2. 阶段一：完成 P5u 成本证据闭环

仅审计和等待当前自动化链路，不得手工复制、重启或替代其工作：

- 开发池：`generated_data/p5u_direct_supervision_dev_seed3310000000`、`...3320000000`、`...3330000000`；每池 3780 样本，10/15/20/25/30 dB，功率视图按 `receiver_noise_seed` 配对。
- 冻结配置：
  - `configs/p5u_direct_supervision_preregister_20260806.json`
  - `configs/p5u_direct_supervision_method_freeze_20260806.json`
  - `configs/p5u_direct_supervision_launch_manifest_20260806.json`
- 当前自动进程：P5u direct supervisor、其 executor、`p5u_autonomous_progression_monitor`、`p5u_final_evidence_freezer`。只读检查其 PID 与回执；不启动重复实例。

P5u 开发阶段只有同时满足以下条件才算通过：

1. 三个开发池 × 五个冻结模型（3041--3045）全部完成；full-pool logits、probabilities、hard decisions、base parameters、dense-head values、candidate parameters、非目标参数和汇总指标均严格等价。
2. 固定 batch `0..125` 的输出严格等价；batch=126，warm-up=5，timed repetitions=15，不得更改。
3. 每模型、每池 `median latency(shared_dense/reference) <= 0.85`。
4. 每次计时的 CUDA 守卫均未发现外部 Python CUDA compute 进程，且环境、checkpoint、pool、model、restore、terminal 回执完整。
5. supervisor 的记录子 PID 与实际 executor 一致，所有 `test_generated/test_opened/locked_test_used` 均为 false。

若任一条件失败：立即保留并冻结 P5u outcome/abort，不重试、不调参、不复用任何 P5u 身份；仅分析失败类型（实现、路径/依赖、数据审计、CUDA 冲突、外部中止或方法门槛）。

若三开发池通过：只允许由既有 autonomous monitor 生成并启动两个确认池 `3340000000` 与 `3350000000`；不得手工提前生成或启动。两个确认池使用完全相同的冻结实现和门槛。确认也全通过后，等待既有 final-evidence-freezer 生成 `configs/p5u_final_evidence_freeze_20260806.json`，再审计其哈希、五模型 × 五池结果、全部 latency ratio 和失败/测试标志。

只有 final evidence 文件真实存在并通过审计，才能称为 `P5-final`：其允许的唯一表述是“对冻结的五模型 JNR-only dense attachment，在所定义输入与固定 batch 条件下，shared frozen-forward 与 reference 输出严格等价，并有经新身份确认的端到端延迟降低”。

## 3. 阶段二：P5-U，OOD 与模板失配泛化（仅在 P5-final 后）

先写 P5-U 预注册和方法冻结，再生成任何数据。仅改变“域外物理条件/模板失配”一个机制，分类 logits、阈值、非目标参数、训练预算与五个冻结模型不变。

1. 先定义可审计的 OOD 轴：JNR 区间外推、目标参数边界外推、模板失配；明确哪些是未见值、哪些是新模板，禁止将功率视图当独立身份。
2. 三个新开发身份池筛选；只有所有预注册的 per-target、per-power、per-combination、worst-cell、安全门槛和 bootstrap CI 通过，才生成两个全新确认池。
3. 报告 MFT/VFT/ISRJ NMAE、增益/退化、最差单元格、置信区间和五模型一致性；若优势仅来自响应强度且分类分离或困难负样本 FPR 恶化，归为失败。
4. 失败即冻结 P5-U，不将成功的 P5u 成本结论扩大为 OOD 结论。

## 4. 阶段三：P5-CD，组合不相交泛化（仅在 P5-U 通过或明确冻结后）

先对组合身份、训练/开发/确认角色及 `receiver_noise_seed` 做全量审计。训练、开发、确认的组合必须严格不相交，且同源九功率视图不得跨角色。以 seen/unseen gap、hardest unseen combinations、组合大小分层、每标签 recall/AUPR、困难负样本 FPR、参数 NMAE 和 worst cell 为主要指标。采用 3 个新开发 + 2 个新确认身份池，并设 matched compute/control。未见的已知组合不得自动宣称为未知类。

## 5. 阶段四：P5-LT，多轴长尾安全（仅在 P5-CD 后）

分别可控地构造并审计：(a) 标签频率，(b) 组合频率/复杂度，(c) 弱分量可见性。不能把功率直接称为长尾，九功率视图仍按同源身份分组。每轮只改变一个长尾轴；必须有 matched compute/control，并完成 3 开发 + 2 确认。报告 macro/micro AUPR、subset accuracy、per-label recall、Hamming loss、困难负样本 FPR、分离度、target NMAE、worst cell 和成本。若 recall 增加而 AUPR、分离度或困难负样本 FPR 恶化，判为 response amplification 并停止。

## 6. 创新点与最终交付

每一步创建可复核 evidence card，至少包含研究问题、单一机制、对照、数据角色/身份、哈希、运行命令、随机种子、统计检验、最差单元格、成本、结论边界和失败原因。

最终只可按证据强度整理创新：

- P5e：五模型、新身份确认下的 bounded JNR-only correction；
- P5-final（若通过）：输出严格等价且有新身份确认的 frozen-forward 成本结论；
- P5-U/P5-CD/P5-LT：只有各自完成开发和全新确认后，才各自写入相应的泛化或安全主张；
- 保留 P7a、P2w、P3o-U、P8b/P8c 等负结果作为贡献边界，不包装为正向结果。

在所有必要开发/确认及消融完成后，冻结 E0、P5e、P5-final、各通过路线和 locked-test preregistration 的哈希与清单。随后停止，向用户请求一次性 locked-test 明确授权；没有授权绝不执行测试，也不能将此 Goal 标记为 complete。

每次自动续跑必须：先读现场状态，再更新计划；只汇报已发生的证据；对正在运行而无 terminal receipt 的任务只标记 `running/uncertain`；不能因耗时、上下文压缩或单轮结束而把 Goal 标记 complete。
```
