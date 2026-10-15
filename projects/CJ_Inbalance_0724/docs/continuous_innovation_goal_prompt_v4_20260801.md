# 可直接用于 GPT/Codex Goal 的专业 Prompt（V4）

```text
目标名称：
复合干扰联合感知的跨干扰安全主干与 Level-4 持续验证

总体目标：
在 C:\LJQ_code\CJ_Inbalance_0724 中，基于已经冻结的 P1n Level-3 系统，持续设计、实现、执行和分析可复现实验。当前首要任务不是继续增加后处理专家，而是解决 ultra-close MFT 对 ISRJ 的跨干扰假阳性，使识别、12 参数 predicted-gate 提取、风险控制与计算代价形成完整论文证据链。未经用户明确授权，禁止访问 locked test。

当前已确认事实：
1. P1n 主干为 mlcnn_iq_grouped_task_routed，输入双通道长度3000 I/Q，输出6类多标签概率和6×2共12个参数。
2. 主损失为 BCEWithLogitsLoss + 0.1×masked Smooth-L1；参数损失仅在 label=1、parameter_mask=1、target有限的位置计算，inactive NaN不得填零。
3. 主干超参数：lr=1e-4，weight decay=1e-4，warmup=2 epochs，cosine scheduler，epochs=20，batch size=63，seeds=2041/2042/2043。
4. P1n 已有 Level-3 证据：ISRJ Recall平均增益+0.08372，95% CI=[+0.07407,+0.09244]；12参数 predicted-gate NMAE从0.72879降到0.67814，相对改善6.95%，3/3 seeds改善。
5. P3l 两套新身份独立确认中，6/6模型×数据组合Recall增益为正；总体增益+0.01806，receiver-seed聚类95% CI=[+0.00875,+0.02740]；但最坏单元FPR=0.15161，故仍未达到Level 4。
6. FPR失败集中于 ultra_close MFT。多数情况下P1n已降低主干FPR，根因是主干本身未显式区分MFT超密延迟峰列与ISRJ周期截断结构。
7. P3i/P3j/P3k表明紧凑蒸馏不能3/3稳定保留教师收益；P3e/P3g/P3h表明lazy/cascade不能同时满足性能与<=1.5×延迟；P3m后处理风险守卫仍有一个seed最大FPR=0.11613。以上分支全部冻结，不得在已消费开发身份上调参。
8. MFT count专家仍未通过复合波形与弱分量准入门槛，不得接回主系统。
9. locked_test_used=false。

当前唯一主假设 P3n：
在 MLCNN 主干训练阶段加入 MFT↔ISRJ 条件间隔监督，而非冻结后处理，可以产生新的可辨识表示并降低 ultra-close MFT 场景的 ISRJ FPR。

冻结损失形式：
L = L_cls + 0.1 L_param + lambda_xci L_xci

仅在 MFT=1, ISRJ=0 时约束 z_MFT - z_ISRJ >= margin；
仅在 ISRJ=1, MFT=0 时约束 z_ISRJ - z_MFT >= margin；
MFT与ISRJ同时存在或同时不存在时不施加互斥间隔。

执行规则：
1. 每轮只选择一个可证伪问题；访问开发结果前冻结数据角色、网络、损失、lambda_xci、margin、阈值、训练超参数、随机种子和统计脚本。
2. 下一轮建议使用全新 train/selection身份680000000、development身份690000000；通过后再用700000000和710000000做两身份独立确认。
3. receiver_noise_seed是独立物理身份；同一身份的不同功率和机制视图必须成组，paired bootstrap按身份抽样。
4. 不得使用P3l的640M/650M或P3m的670M选择损失权重、阈值或模型。
5. 参数损失继续严格mask；不得把inactive NaN填零。
6. 每次模型或规则变化后必须使用全新身份；smoke和单seed只证明代码路径，不能作为方法证据。
7. 保存 preregister、method_freeze及SHA-256、manifest、identity hash、checkpoint、history、逐样本预测、metrics、bootstrap、test receipt和Markdown报告。
8. 每轮结束必须区分：已验证结论、失败结论、推测性解释、证据边界、下一准入门槛。

P3n开发准入门槛：
1. 3/3 seeds 相对对应主干/P1n的ISRJ Recall方向为正。
2. 所有机制×功率单元最大ISRJ FPR<=0.10。
3. macro-AUPR delta>=-0.005。
4. 12参数 predicted-gate NMAE相对原P1n不得恶化超过1%。
5. miss-penalty与worst-combination不得恶化。
6. ultra_close的25/35/50 dB三个单元均需单独报告。

独立确认门槛：
1. 使用完全冻结的P3n在700M/710M一次性评估。
2. 6/6模型seed×数据seed Recall增益为正。
3. 每个模型seed及总体identity-clustered bootstrap Recall差值CI下界>0。
4. 最大单元FPR<=0.10；macro-AUPR delta>=-0.005。
5. 12参数 predicted-gate NMAE改善方向3/3一致，NMAE及miss-penalty delta的CI上界<0。

效率路线：
只有P3n通过独立确认后才重新评估效率。优先利用“训练时附加损失、推理时零新增模块”的优势；报告参数量、FLOPs边界、batch1/64延迟、吞吐和显存，并与main-only及always-on P1n比较。

停止条件：
1. 达到Level 4且效率证据通过后，停止并向用户申请一次性locked test授权；或
2. 连续多轮证明条件间隔监督仍不可泛化，形成不可辨识证据和明确替代路线；或
3. 后续必须访问locked test或生成用户未授权外部数据才能继续。

开始执行：
先审计现有MLCNN训练入口、分类logit和参数mask实现，建立P3n预注册与单元测试；随后生成全新680M/690M身份数据，正式训练3个seeds并报告完整开发结果。不要停留在建议、代码修改或smoke test。
```
