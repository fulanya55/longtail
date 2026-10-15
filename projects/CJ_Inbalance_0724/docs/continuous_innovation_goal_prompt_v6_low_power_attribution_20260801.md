# 低功率联合感知归因与可靠创新 Goal Prompt V6

下面文本可直接作为新的 Goal Prompt 使用。

```text
目标名称：复合干扰联合感知的低功率可辨识性、联合选模与有限样本安全闭环

总体目标：
在 C:\LJQ_code\CJ_Inbalance_0724 中，使用 D:\anaconda3\envs\PY10\python.exe，持续设计、实现、执行和分析可复现实验。在不考虑半监督和长尾学习的前提下，把当前“MLCNN 多标签分类 + IQ-Grouped 12 参数回归 + P1n 安全路由”推进为能够经受论文审稿的可靠联合感知方法。

本 Goal 不以继续堆叠损失、专家或后处理为目标。首先必须把训练数据规模、checkpoint 目标、阈值校准和低功率表征可辨识性四个因素拆开。只有前三项被公平修复后仍然失败，才允许引入新的物理双视图表征。

未经用户明确一次性授权，禁止生成、打开或使用 locked test。

一、当前冻结事实

1. 当前联合模型：
   - 分类主干：MLCNN，多尺度一维卷积，输入 [B,2,3000] I/Q；
   - 参数分支：IQ-Grouped 1D backbone；
   - 分类输出：6 类多标签 logits，类别顺序为 [FMB, SawFM, SinFM, MFT, VFT, ISRJ]；
   - 参数输出：[B,6,2]，共 12 个参数；
   - 参数损失只在 label=1 AND parameter_mask=1 AND finite(target) 上计算，inactive NaN 禁止填 0。

2. P3n-A 冻结训练配置：
   - BCE + 0.1 × masked Smooth-L1；
   - lr=1e-4，weight decay=1e-4；
   - warmup=2 epochs，cosine scheduler；
   - epochs=20，batch size=63；
   - model seeds=2041/2042/2043；
   - checkpoint objective=early role 上 oracle parameter macro-NMAE 最小；
   - calibration=逐类 F1 网格。

3. P3n-A 已完成 B1/B2/B3 三种子 formal development：
   - B1 原始 BCE：Recall=0.8589，最坏单元 FPR=0.2591，Macro-AUPR=0.9118；
   - B2 hard-negative：Recall=0.8619，最坏单元 FPR=0.2731，Macro-AUPR=0.9111；
   - B3 条件 margin：Recall=0.8658，最坏单元 FPR=0.2677，Macro-AUPR=0.9095；
   - B3 三种子最坏单元 FPR=0.20645/0.39032/0.20645；
   - B3 相对 B1 Recall 差值=+0.02313/+0.00729/-0.00979；
   - B3 没有稳定击败 B2，不满足 3/3 同方向，不进入 P3n-B。

4. 参数任务在 B1/B2/B3 间基本不变：
   - oracle parameter macro-NMAE 均值约 0.4336；
   - predicted-gate parameter NMAE 均值约 0.4263–0.4268；
   - 说明 margin 没有破坏参数任务，但也没有形成联合收益。

5. 旧模型在同一 seed770000000 development 上只作为外部参考：
   - old main：Recall=0.8964，最坏单元 FPR=0.1441，Macro-AUPR=0.9642；
   - old P1n：Recall=0.9130，最坏单元 FPR=0.1172，Macro-AUPR=0.9661；
   - 旧模型训练数据和协议不同，禁止把该差值直接写成架构因果结论。

6. 已冻结的失败归因：
   - 风险主要集中在 25 dB，但横跨 ID_control、weak_target、close_spacing、ultra_close，不再只是 ultra-close MFT；
   - 参数最优 checkpoint 位于 epoch 10/15/18，而分类 AUPR 峰值多在 epoch 1–2，early AUPR 损失 0.0130–0.0278；
   - P3n fit 只有 6615 个配对视图，与旧主干训练覆盖不等；
   - 逐类 F1 阈值随种子明显漂移；
   - 已消费 development 上的只读诊断表明，把最坏单元经验 FPR 压到 0.10 后，Recall 仍只有约 0.70–0.75，说明阈值不是唯一瓶颈；
   - 条件 margin 很快趋近零，却没有改善困难单元，继续搜索 lambda 或 margin 被禁止。

7. 已停止分支：
   - P3n 条件 margin 创新主张；
   - P3n-B 路由重建；
   - 在 seed770000000 上调损失、阈值、margin 或模型；
   - P3e/P3g/P3h lazy/cascade；
   - P3i/P3j/P3k 蒸馏；
   - P3m 冻结表示后处理守卫；
   - 共享浅层前端的继续加宽、加深、移动 detach 点或放宽分类安全门。

8. seed760000000 和 seed770000000 已消费，只能用于冻结事实与诊断，不得再参与候选选择。当前 test_generated=false、test_opened=false、locked_test_used=false。

二、当前唯一研究问题

在不改变主干架构和基础损失的条件下，扩大身份安全的训练覆盖、采用分类约束的联合 checkpoint 选择、并进行有限样本最坏组安全校准，是否足以恢复 25 dB 的 ISRJ Recall-FPR 折中，同时保持完整 12 参数联合感知性能？

该问题必须先于任何新专家或新物理分支回答。

三、可证伪假设

H1：P3n-A 的主要退化由训练覆盖不足和参数单目标 checkpoint 引起，F1 校准进一步放大 FPR；在不改变网络的前提下，full-scale 身份覆盖 + 参数约束的联合 checkpoint + 有限样本安全校准可以显著恢复分类安全，并保持参数精度。

若 H1 成立：
1. full-scale 相对 small-scale 在 3/3 seeds 提高 25 dB ISRJ AUPR；
2. joint checkpoint 相对 parameter-min checkpoint 在参数 NMAE 恶化不超过 1% 的条件下，稳定提高 Macro-AUPR 或降低最坏 FPR；
3. safety calibration 能在不造成不可接受 Recall 损失的情况下使全部机制×功率单元 FPR<=0.10；
4. 不需要新表征即可恢复接近旧 P1n 的折中，此时不得把收益写成网络结构创新，而应收缩为联合选模与风险校准方法。

若 H1 不成立：
在 full-scale、joint checkpoint 和安全校准均已修复后，25 dB safe Recall 仍不足，则形成“低功率表示不足”的直接证据，才允许进入 P3o-B 物理双视图表征。

四、P3o-A：数据规模—checkpoint—校准三因素归因

1. 数据身份
   - 使用未生成、未打开的全新 receiver_noise_seed=780000000 作为训练/early/selection/calibration 身份池；
   - 使用全新 receiver_noise_seed=790000000 作为一次性 formal development；
   - 若开发全部通过，冻结后使用 receiver_noise_seed=810000000 和 820000000 做两套 independent confirmation；
   - seed780M/790M 原为未执行 P3n-B 的预留身份，必须在预注册中记录其重新分配原因；
   - 所有 root、既往 development、independent 和 locked 身份必须做零重叠审计；
   - 同一 receiver_noise_seed 的全部机制、组合、参数和功率视图必须归入同一角色。

2. 数据覆盖
   - 仍覆盖 63 个组合、ID_control/low_count/close_spacing/ultra_close/weak_target、25/35/50 dB；
   - Small：每组合 7 个 fit 身份，复现 P3n 的训练规模；
   - Full：每组合至少 27 个 fit 身份，使训练视图规模达到或略高于 25,200；
   - early、selection、calibration 必须使用互不重叠的身份，且 Small/Full 共享相同评估角色；
   - 所有参数列、label、parameter_mask、样本索引、组合、机制、功率和 receiver_noise_seed 必须先审计。

3. 网络和训练
   - 固定 P3n B1 的 `mlcnn_iq_grouped_task_routed`；
   - 固定 BCE + 0.1 masked Smooth-L1；
   - 不使用 hard-negative、margin、蒸馏、router 或新专家；
   - 固定 lr=1e-4、weight decay=1e-4、warmup=2、cosine、20 epochs、batch=63、seeds=2041/2042/2043；
   - Small 与 Full 除 fit 身份数量外完全相同；
   - 若数据量使每 epoch 更新步数变化，必须同时报告 epoch-matched 和 optimizer-step-matched 边界，禁止把更多更新步数隐藏为数据规模收益。

4. 同一训练轨迹保存两个 checkpoint
   - C_param：early oracle parameter macro-NMAE 最小；
   - C_joint：先找出 parameter macro-NMAE <= 1.01 × 该轨迹最小值的 epoch，再从中选择 early Macro-AUPR 最大的 epoch；
   - 两个 checkpoint 的规则必须在查看 seed790M 前冻结；
   - 不允许使用 development 选择 epoch。

5. 两种阈值方案
   - T_F1：原逐类 F1 网格；
   - T_safe：只在 calibration 身份上，按机制×功率最坏组校准 ISRJ 阈值；使用单侧 95% Clopper-Pearson 上界，要求上界<=0.10；若有限样本下无可行阈值，明确报告 infeasible，不得放宽门槛；
   - 其他类别保持冻结校准规则；
   - calibration 样本量必须足以报告每个单元的正负例数和置信上界。

6. 必须形成的 2×2×2 归因表
   - Small/Full；
   - C_param/C_joint；
   - T_F1/T_safe；
   - checkpoint 和 threshold 分析来自同一训练轨迹，不额外增加训练预算；
   - 所有候选均在 seed790M 上一次性评估，评估后不得重新选择规则。

五、P3o-A 指标

分类：
- Subset accuracy、Micro/Macro-F1、Macro-AUPR；
- 六类 Recall/Precision/FPR/AUPR；
- ISRJ 总体 Recall/FPR；
- 逐机制、逐功率、逐组合 ISRJ Recall/FPR/AUPR；
- 25 dB 单独结果；
- maximum cell FPR；
- calibration 与 development 的风险迁移差；
- 安全阈值下的 coverage/abstention（若使用 abstain，必须预注册且不替换错误为缺失）。

参数：
- 12 参数逐目标 MAE、NMAE、Pearson；
- oracle-positive macro/micro NMAE；
- predicted-label gate NMAE、coverage、false-positive gate rate；
- miss-penalty NMAE；
- worst-combination parameter NMAE；
- 逐功率和弱分量参数结果。

训练冲突：
- 每 epoch early parameter NMAE 与 Macro-AUPR；
- C_param 与 C_joint 的 epoch 差；
- 分类收益对应的参数代价；
- 不同数据规模的更新步数、训练时间和收敛曲线。

统计：
- 3 个模型种子的配对差；
- receiver_noise_seed 聚类 paired bootstrap 95% CI；
- 不得把功率视图作为独立样本。

六、P3o-A 冻结准入门

只有 Full + C_joint + T_safe 同时满足以下门，才允许认为 H1 获得支持：

1. identity overlap=0；
2. 3/3 seeds 的 maximum mechanism×power cell ISRJ FPR<=0.10；
3. 3/3 seeds 的 ISRJ Recall 相对 Full+C_param+T_F1 下降不超过 0.02，且总体 Recall>=0.85；
4. 25 dB ISRJ Recall>=0.78，且 25 dB 每个机制单元 FPR<=0.10；
5. Macro-AUPR 相对 old P1n 外部参考的差值>=-0.005，或相对 Small+C_param+T_F1 的提升在 3/3 seeds 为正且聚类 bootstrap CI 下界>0；两种证据边界必须分别报告；
6. oracle 和 predicted-gate 12 参数 macro-NMAE 相对 Full+C_param 不得恶化超过 1%；
7. miss-penalty 和 worst-combination 不得恶化；
8. Full 相对 Small 的 25 dB AUPR 3/3 seeds 同方向提高；
9. C_joint 相对 C_param 的分类改善不能完全由参数明显恶化解释；
10. T_safe 在 calibration 上有可行解，且 development 风险迁移仍通过；
11. 全量测试通过；
12. test_generated=false、test_opened=false、locked_test_used=false。

若只通过部分门，不得重新解释为成功，也不得事后降低 0.10 FPR、0.78 低功率 Recall 或 1% 参数退化门。

七、P3o-A 归因规则

1. Full 明显优于 Small，C_joint/T_safe 没有额外稳定收益：
   - 结论为训练覆盖不足；
   - 不支持新结构创新；
   - 后续方法主张只能是 identity-safe 数据协议与规模效应。

2. C_joint 稳定优于 C_param，且参数代价<=1%：
   - 支持“参数约束联合 checkpoint”作为训练选择机制；
   - 必须在新身份 confirmation 复现后，才能写成方法贡献。

3. T_safe 使 FPR 通过但 Recall<0.78：
   - 结论为校准可控但表征不充分；
   - 不允许继续调阈值，进入 P3o-B 表征实验。

4. Full+C_joint 在排序指标上仍低，development 的 25 dB AUPR 仍约 0.85：
   - 形成低功率表示不足证据；
   - 允许进入 P3o-B。

5. T_safe 在 calibration 上不可行：
   - 说明有限样本或分数重叠使安全门不可满足；
   - 不得使用经验 FPR 替换 Clopper-Pearson 门；
   - 进入 P3o-B 前先报告样本充分性界限。

八、P3o-B：仅在 P3o-A 证明表示不足后执行

唯一新机制：训练期低功率跨干扰物理双视图表示，不继续调 margin。

1. 视图 A：多尺度复自相关/延迟峰序列，显式描述 MFT 延迟峰结构与 ISRJ 周期截断结构；
2. 视图 B：局部 STFT/瞬时频率与时频占用序列，保留低功率局部周期和断续结构；
3. 只在 MFT=1,ISRJ=0 与 ISRJ=1,MFT=0 的独占样本上施加 supervised contrastive 或 prototype separation；
4. MFT=1,ISRJ=1 共存样本不得施加互斥损失，只做多标签 BCE；
5. 加入同参数量的普通时序卷积对照，以及“仅视图 A”“仅视图 B”“双视图但无对比损失”消融；
6. 参数头、基础损失、checkpoint 和安全校准采用 P3o-A 冻结规则；
7. 使用新的训练/development 身份，不得复用 780M/790M 调节分支宽度、对比温度或权重；
8. 先做 oracle separability 和线性 probe，若相对常数/原主干在 25 dB 的 AUPR 或受控 FPR Recall 无实质改善，则在训练大模型前停止。

P3o-B 只有在 3/3 seeds 的 25 dB safe Recall、maximum cell FPR、Macro-AUPR 和参数安全均通过，并稳定优于等参数普通时序对照时，才允许形成“物理双视图跨干扰可辨识表征”的创新主张。

九、独立确认

只有 P3o-A 或 P3o-B 的方法、checkpoint、归一化器、阈值和所有规则完全冻结后，才执行 seed810M/820M：

1. 不训练、不调参、不重新校准；
2. 6/6 模型 seed×数据 seed 的核心方向一致；
3. 每个数据 seed 和总体 receiver-seed clustered bootstrap CI 分别报告；
4. maximum cell FPR<=0.10；
5. 25 dB safe Recall>=0.78；
6. Macro-AUPR 不退化；
7. predicted-gate 参数 NMAE、miss-penalty 和 worst-combination 不退化；
8. 失败则冻结，不得读取 locked test。

十、每轮强制闭环

1. 总结最近三轮的研究问题、数据角色、模型、损失、超参数、结果、失败门和证据等级；
2. 每轮只解决一个主要问题；
3. 在数据生成或结果访问前写 preregister JSON、method freeze 和 SHA-256；
4. 审计参数列、样本索引、波形分量、组合、功率和 receiver_noise_seed；
5. 先跑单元测试与 smoke，只证明代码路径；
6. 再执行三种子 formal development；
7. 保存 manifest、identity hash、角色索引、配置、日志、checkpoint、逐样本预测、阈值、指标、bootstrap、failure audit 和 Markdown 报告；
8. 严格区分 smoke、development、independent confirmation 和 locked test；
9. 失败时冻结失败机制，不得在已消费身份上调参或降低门槛；
10. 每轮结束提出更窄的可证伪假设，并继续执行，直到达到停止条件。

十一、停止条件

仅在以下情况之一停止：

1. P3o 方法通过 formal development、两套 independent confirmation、完整 12 参数安全门和效率审计，随后向用户申请一次性 locked-test 授权；
2. P3o-A 与 P3o-B 连续证明 25 dB 下缺乏可恢复的物理可辨识信息，形成观测长度、功率或生成合同限制的明确负结论；
3. 继续必须访问 locked test 或用户未授权的外部数据。

十二、每轮输出

必须输出：
1. 当前可写论文主张与证据等级；
2. 本轮唯一假设、预注册文件和哈希；
3. 数据身份、角色、组合、功率和参数审计；
4. 网络、损失、训练方式、超参数和 checkpoint 规则；
5. 三种子总指标、逐功率、逐机制、逐组合和 12 参数结果；
6. paired bootstrap CI 和最坏组；
7. 失败门、根因、停止分支和保留知识；
8. 下一轮唯一机制；
9. test_generated、test_opened、locked_test_used。

开始执行：
先读取并核对 P3n-A 的 summary、failure_audit、preregister、method freeze、逐种子 history 和 predictions；随后建立 P3o-A 预注册与身份种子占用审计，实现 full-scale 数据协议、双 checkpoint 保存和 Clopper-Pearson 最坏组校准，完成测试后执行三种子 formal development。不要停留在建议、代码修改或 smoke test。
```
