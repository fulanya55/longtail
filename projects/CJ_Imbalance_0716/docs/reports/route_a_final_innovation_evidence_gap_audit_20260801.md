# Route A 当前创新证据链审计与下一步准入条件

## 审计结论

当前 `H1 + B2 + MFT shared-decision + VFT Ridge-STFT` 候选已经具备可靠的**开发集机制证据**，但尚不具备“完成可靠创新点证明”的资格。不能访问 test-v3，也不能把 development-v12 称为全项目新鲜确认；完整候选在 ISRJ 参数 OOD 上失败，且两项针对该失败的低风险修正均被否定。

因此目前可写入论文的方法主张只能是：在身份隔离的开发协议下，标签特异的物理/时序证据与 MFT 的共享安全决策能提高弱分量识别并保持 RawSaw SawFM 输出；不能主张已经获得完整跨参数泛化、外部鲁棒性或最终独立测试验证。

## 已有的正向证据

| 证据问题 | 当前结论 | 关键量化证据 | 可信边界 |
|---|---|---|---|
| 六标签整体改善 | 通过 | v12 macro-AUPR `+0.00660`，macro-F1 `+0.01099` | 相对 H1，5 seeds |
| 弱分量改善 | 通过 | MFT/VFT/ISRJ 平均弱召回 `+0.03759`，弱 AUROC `+0.03886` | v12，5 seeds |
| SawFM 不受损 | 通过 | SawFM 概率与预测逐样本完全一致 | RawSaw 路由冻结 |
| VFT 专家不是容量假象 | 通过 | v12 相对 capacity 弱 AUROC `+0.01706`，相对 shuffled `+0.01757` | 本线路冻结确认 |
| MFT 的负迁移安全性 | 通过 | 高阶难负 FPR `-0.02658`，且 0 个 MFT negative-to-positive flip | 共享父级校准、非正残差 |
| 统计证据 | 通过 | 四项主指标 bootstrap CI 均排除 0；Holm 配对 t 检验均小于 0.05 | n=5；精确 Wilcoxon 经 Holm 后为 0.125，应如实报告 |
| 分层开发鲁棒性 | 通过但有限制 | 96 个支持的 AUPR 分层无低于 -0.01 的退化；所有 receiver cohort 的目标弱召回增益为正 | worst ISRJ subgroup FPR=`0.27391` 必须单列报告 |
| 工程代价 | 通过 | GPU 2465.94 sample/s，CPU 319.34 sample/s，GPU 延迟比 2.36，神经增量参数 817 | FLOPs 仅统计 profiler 支持的神经算子，明确不含 FFT/匹配滤波/树遍历 |
| 冻结组件必要性 | 通过（决策级） | 移除 VFT、MFT、ISRJ 的五种子冻结 lesion 均产生预期标签特异变化 | v12 后验诊断，非重新训练消融 |
| 弱分量阈值敏感性 | 通过（评估级） | 5/10/15 dB 均保持正弱 AUROC；15 dB 弱召回未达到原升级阈值 | 评估定义敏感性，不改变模型 |
| VFT 门函数形状 | 已完成，差异不显著 | sqrt 相对线性弱 AUROC `+0.00010`，CI 跨零 | 不作为创新点 |
| MFT 残差上界 | 已完成，构造集微弱偏向 cap=1 | hard-FPR 额外下降约 `0.00333`，但未在新鲜数据确认 | 不改变已冻结 0.75 候选 |
| VFT 弱正监督必要性 | 已完成（构造期） | 强弱混合正例相对仅弱正例在 early 严格弱视图的 AUROC `-0.03265`、AUPR `-0.02264`；两者 bootstrap CI 均不跨 0 | 仅监督机制消融，不是融合候选晋级 |
| RawSaw top-k=4/8/16 | 已完成（等预算重训） | top-k=16 的 SawFM 弱 AUROC 点估计最高；4/8 相对 16 的 CI 均跨 0，且重训 16 未逐权复现历史 CUDA checkpoint | 作为敏感性/消融，不作为最优池化或新候选主张 |

## 已证伪或停止的设计

| 假设 | 证据 | 裁决 |
|---|---|---|
| 只用较小 VFT 残差与共享决策即可解决 OOD | ISRJ-OOD 的 VFT FPR 约 `0.3768`，弱 AUROC 增益不足门槛 | 停止该限幅策略 |
| 加入 ISRJ logit 的低维交叉头可排除 VFT 假阳 | 在 v3/v12 VFT AUPR 均约 `-0.016`，且相对 capacity/shuffle 的弱 AUROC 为负 | 停止交叉头；不再调权重或阈值 |
| 常规浅层 STFT-CNN 能提供更强来源判别 | VFT/ISRJ 均未通过可辨识门槛 | 不扩宽或加深 STFT-CNN |
| ISRJ copy-consistency 可增强周期专家 | 相对原 310D 专家 early AUROC `-0.00367` | 不接回当前候选 |
| 低维相位复制/脊线/复位摘要可替代 VFT Ridge-STFT | 新 source-v1 的五角色诊断中相对 Ridge-STFT early AUROC `-0.06972`，5/5 为负；虽相对相位打乱对照 `+0.08470` | 停止低维摘要替代路线；相位组织有效但信息压缩不足 |

## 目标要求的逐项缺口

| Goal 要求 | 当前状态 | 需要的直接证据 |
|---|---|---|
| 14 项核心消融 | 未完成 | 在当前候选/同一身份协议下将每项消融映射为单一机制问题；不能用旧 split 的数值冒充 |
| BCE、pos-BCE、Focal、ASL、DB、CB、LA、公平容量对照 | 已完成 | 已有 development-v3 固定训练预算实验；最强传统损失基线为 Focal-36 |
| 两阶段解耦分类器 | 已完成 | 新增严格等预算 `Focal24 -> Frozen Linear BCE12`；总预算 36 epoch，未优于 Focal-36，排除额外训练预算解释 |
| 最新多标签长尾方法 | 缺失 | 需要官方可运行实现并改造成相同数据、角色、训练预算、校准和阈值协议；若不存在可审计实现，只能作定性文献对照 |
| 完整候选外部/OOD | 失败 | ISRJ 参数 OOD 中只 ISRJ 专家机制复现，完整统一候选 target FPR=`0.20914`、VFT FPR=`0.32653`，未过线 |
| 全项目新鲜确认 | 缺失 | v12 数据 SHA 已被 FC-PTV 线路使用；后续任何新候选必须冻结后才生成并仅使用 development-v13 |
| 噪声失配/外部生成器 | 缺失 | 至少一种身份隔离的新噪声模型或生成器，完全冻结后一次评估 |
| test-v3 | 禁止执行 | 仅在全部缺口补齐、候选与代码/校准/阈值冻结、bundle 完整后创建一次 started guard 并执行一次 |

## 下一步唯一准入的模型假设

不再对当前开发集追加同类低维融合头。下一项模型工作必须先做**来源可分性诊断**：

> 长时密集 ISRJ 与 VFT 的混淆来自两者的局部时频能量相似；若在原始复 I/Q 中加入跨窗口相位复制稳定性、脊线持续度和脉冲复位不规则性的严格匹配特征，则 VFT-positive 与 VFT-negative/ISRJ-positive 难负样本应在独立早期角色上比固定 Ridge-STFT 分数提高 AUROC，且优于相同维度的相位破坏对照。

该诊断必须先在 construction-v3 的 5 seeds 上完成，不进行阈值或融合搜索；只有以下条件全部满足，才允许冻结一个新专家并生成 development-v13：

1. VFT 的严格匹配 early AUROC 相对 Ridge-STFT 至少 `+0.010`，且至少 4/5 seeds 为正；
2. 相对相位/时间破坏对照至少 `+0.020`；
3. validation-to-early 差距小于 `0.15`；
4. 同一特征在 ISRJ 参数 OOD 的 VFT-negative/ISRJ-positive 集合上不出现方向反转；
5. 不使用 test-v3，且不以 development-v12 的结果反向选择特征或超参数。

若诊断失败，停止 VFT-OOD 模块路线，保留当前候选仅作为受限开发结论，转而完成主流基线、消融和论文的边界化表述。

## 证据文件

- 当前候选与 v11/v12：[route_a_h1_b2_mft_vft_v11_v12_shared_policy_results_20260801.md](../results/route_a_h1_b2_mft_vft_v11_v12_shared_policy_results_20260801.md)
- 统计：`outputs/tapc/route_a_final_shared_policy_evidence/statistical_audit.json`
- 分层：`outputs/tapc/route_a_final_shared_policy_evidence/stratified_v12.json`
- OOD：`outputs/tapc/route_a_final_shared_policy_isrj_ood_frozen/route_a_final_shared_policy_isrj_ood_frozen_summary.json`
- 本审计机器可读产物：`outputs/tapc/route_a_final_innovation_evidence/audit.json`
- test-v3 密封清单：`frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json`

所有上述开发和审计工作均为 `locked_test_used=false`、`test_v3_used=false`。
