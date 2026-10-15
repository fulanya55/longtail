# Route A development-v12 FC-PTV 正式结果与停止决策

## 1. 本轮最终结论

预注册结论为 **`stop_fc_ptv`**。FC-PTV 在 development-v12 上不是一个得到验证的稳定化创新点：5/5 种子的约束均不活跃，`q*=(1.5,1.5)`，所以 FC-PTV、FullAlpha1p5 与 ParameterNormOnly 在本轮形成数值等价的参数点和同一组评价结果。候选自身门槛、相对 Full1.5 门槛、三标签同时改善门槛和机制特异性门槛均未全部通过。按停止规则，不进入 development-v13，不做鲁棒性扫描，不执行任何外部或测试集评估。

## 2. 研究问题与可证伪假设

研究问题是：能否仅依靠 C6 fit-only 功能半径，把 SawFM 标签删除产生的配对任务向量投影到一个共享层/分类器联合信赖域，在保留 Full1.5 收益的同时降低最差标签风险和跨种子波动。可证伪点包括：约束必须实际改变系数、FC 必须超过 Full1.5、三目标标签必须获得联合统计支持，并且必须超过参数范数、classifier-only 与正交功能匹配控制。

## 3. 文献差距与创新边界

预注册文献审计确认，task arithmetic、TIES/AdaMerging、Fisher/RegMean、TATR、Fisher-Rao/KL 几何和多任务梯度方法已覆盖参数合并、敏感度信赖域、功能距离或梯度冲突。因而本候选最多只能主张一个很窄的组合：**单标签 loss-deletion 配对向量 + C6 fit-only common-five 稳健 logit 半径 + shared/classifier 两块投影 + 禁止 test adaptation**。本轮约束没有激活，故这个窄创新也未形成实证支撑。

## 4. 证据边界与预注册状态

- 预注册先于 v12 正式模型评估，文档 SHA256：`65ac809176df91a1a2bc344c92b88ccada9e53af144cc9696e481732c128791c`。
- 实现冻结 SHA256：`f251fa2e3da279cd0750800e38480dde6b547a0ff0cd74b3d9d39134f3cba0ad`。
- development-v12 只承担开发证据；历史 locked test、test-v2、sealed test-v3 和预留 test-v12 均未使用。
- `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`、`test_v12_used=false`。

## 5. fresh development-v12 数据身份

- train-only 样本数：25200；文件大小：2549109222 bytes。
- 数据 SHA256：`358e08e6eecbf2c866a137ebeed847475ce1ea2efe2c399b6789a0a8a5bbffe9`。
- sample seed 契约：`[124000001, 124025200]`。
- receiver_noise_seed：`[322677743, 322702942]`，唯一数 25200。
- 与 development-v2--v11、test-v2/v3 契约及其他保留角色均不重叠；`test_files_opened=false`。

## 6. 网络结构与可训练作用域

底座为 `F2_DFMS_Full` 六类复数 I/Q 网络：双复数多尺度编码分支，随后进行 late fusion；并行 raw temporal 多尺度分支采用卷积核 3/7/15，融合到 128 维特征，最后由 128→6 分类器输出六类 logits。完整模型含 391,687 个浮点参数。任务向量只允许落在最后共享 `temporal_backbone.0.*` 和 `classifier.*`；其余复数编码器、融合与投影层保持 C6 锚点不变。

## 7. FC-PTV 冻结机制

每个种子从 `d=M5-C6` 构造 shared 与 classifier 两个块。C6 fit-only logits 按 common-five 标签及 SawFM-present/absent 分层，用 `1.4826×MAD` 给出半径；对每个标签/分层构造 2×2 功能 Gram 矩阵，并求解距离 `(1.5,1.5)` 最近的可行系数，约束为 `0<=q_shared<=q_classifier<=1.5` 且每个功能二次型不超过 1。精确前向 `Dmax<=1.05` 才有效。

## 8. 正式变体与公平性

正式比较包括 C6、M5、Full1.5、FC-PTV、参数范数匹配、classifier-only 2.0、模型中点、CAGrad0.4 和正交任务子空间功能匹配控制。五个其他标签删除端点只用于构造正交控制。所有方法共享初始化、split、long-tail manifest、训练角色、早停、校准、阈值、种子和可训练范围。

## 9. 正式运行完整性

正式种子为 `[2039,2040,2041,2042,2043]`，每个种子的 9 个评估变体均为 valid；数据、split、manifest、checkpoint、功能诊断和逐种子指标均已保存。正式总汇总状态为 `completed`，证据级别为 `development`，没有使用 `--resume` 选择结果，也没有结果后重算系数。

## 10. 各变体绝对性能

| 变体 | macro-AUPR | macro-F1 | target weak recall | target FPR |
|---|---:|---:|---:|---:|
| C6_Joint | 0.86432 | 0.74802 | 0.29722 | 0.19425 |
| M5_GradientIsolated | 0.86780 | 0.75623 | 0.30996 | 0.19414 |
| FullAlpha1p5 | 0.87027 | 0.76007 | 0.31531 | 0.19597 |
| FC_PTV | 0.87027 | 0.76007 | 0.31531 | 0.19597 |
| ParameterNormOnly | 0.87027 | 0.76007 | 0.31531 | 0.19597 |
| ClassifierOnlyAlpha2p0 | 0.86998 | 0.76541 | 0.31493 | 0.19038 |
| ModelAverage0p5 | 0.86564 | 0.75219 | 0.30338 | 0.19489 |
| CAGrad0p4TwoTask | 0.86551 | 0.75121 | 0.29961 | 0.19522 |
| OrthogonalTaskSubspaceFunctionalMatched | 0.86646 | 0.75385 | 0.30326 | 0.19392 |

## 11. FC-PTV 相对 C6 的候选效应

- macro-AUPR：`0.00595`，95% CI `[0.00435, 0.00756]`。
- macro-F1：`0.01205`，95% CI `[0.00795, 0.01693]`。
- target mean weak recall：`0.01808`，95% CI `[0.00755, 0.02841]`；低于预注册 `+0.025`，因此候选门槛失败。
- target FPR delta：`0.00172`；SawFM-present macro-F1 delta：`0.02103`；SawFM-absent delta：`0.00365`。

| 候选门槛检查 | 结果 |
|---|---|
| `each_target_weak_nonnegative` | 通过 |
| `macro_aupr_noninferior_minus_0p005` | 通过 |
| `macro_f1_gain_0p010` | 通过 |
| `positive_target_weak_4_of_5` | 通过 |
| `sawfm_absent_macro_f1_noninferior_minus_0p005` | 通过 |
| `sawfm_present_macro_f1_positive` | 通过 |
| `target_aupr_noninferior_minus_0p010` | 通过 |
| `target_fpr_cap_0p20` | 通过 |
| `target_weak_recall_gain_0p025` | 失败 |
| `valid_5_of_5` | 通过 |

## 12. FC-PTV 相对 FullAlpha1p5

所有逐种子和逐指标差值均为 0。最差标签改善为 `0.00000`，target weak、macro-F1、macro-AUPR 和 FPR 的 FC-Full 均为 `0.00000` 或数值等价的 0；没有正 CI 优势，跨种子标准差也未降低。

| 相对 Full1.5 检查 | 结果 |
|---|---|
| `macro_aupr_noninferior_minus_0p003` | 通过 |
| `macro_f1_noninferior_minus_0p003` | 通过 |
| `one_primary_advantage_positive_ci` | 失败 |
| `target_fpr_delta_le_0p005` | 通过 |
| `target_mean_weak_not_lower` | 通过 |
| `valid_and_finite_5_of_5` | 通过 |
| `weak_effect_std_lower` | 失败 |
| `worst_target_improves_by_0p005` | 失败 |

## 13. 三目标标签同时改善检验

| 标签 | FC-C6 weak recall mean | 普通 paired 95% CI | family-wise 95% lower |
|---|---:|---:|---:|
| MFT | 0.03290 | [0.01124, 0.04688] | 0.01202 |
| VFT | 0.00380 | [-0.01375, 0.01777] | -0.01708 |
| ISRJ | 0.01755 | [0.00224, 0.03148] | -0.00333 |

三标签均值为正，且每个标签至少 4/5 种子非负；但 VFT 与 ISRJ 的 family-wise 下界分别为 `-0.01708` 和 `-0.00333`，未超过 0。因此只能说本轮均值方向非负，不能声称三个标签均获得统计可靠改善。

## 14. 机制特异性控制

- FC 与 ParameterNormOnly 完全相同，无法证明功能约束优于普通全方向缩放。
- FC 相对 ClassifierOnly2.0：target weak mean `+0.00038`，但 macro-F1 为 `-0.00533`，且主要 weak 指标无正 CI 优势，门槛失败。
- FC 相对正交功能匹配控制：target weak `+0.01204`，95% CI `[0.00316, 0.02081]`；这一项通过，但不足以建立整体机制特异性。

| 机制检查 | 结果 |
|---|---|
| `beats_classifier_only_control` | 失败 |
| `beats_orthogonal_functional_control` | 通过 |
| `beats_parameter_norm_control` | 失败 |
| `orthogonal_functional_match_5_of_5` | 通过 |
| `shared_component_nonzero_5_of_5` | 通过 |

## 15. 功能距离、系数与失败根因

| seed | q_shared | q_classifier | 线性 Dmax | 精确 Dmax | 正交匹配相对误差 |
|---:|---:|---:|---:|---:|---:|
| 2039 | 1.500 | 1.500 | 0.22221 | 0.16911 | 0.01401 |
| 2040 | 1.500 | 1.500 | 0.28824 | 0.23392 | 0.02893 |
| 2041 | 1.500 | 1.500 | 0.36804 | 0.20237 | 0.03934 |
| 2042 | 1.500 | 1.500 | 0.27061 | 0.26400 | 0.00870 |
| 2043 | 1.500 | 1.500 | 0.30410 | 0.18086 | 0.06832 |

所有线性 Dmax 仅为 0.222--0.368，精确 Dmax 仅为 0.169--0.264，远小于半径 1。因此 `(1.5,1.5)` 本身已可行，投影没有发生。根因不是求解失败，而是 **C6 MAD 半径相对于该任务向量过宽，预注册约束在当前尺度下没有辨别力**。由于不允许结果后收紧半径或重新搜索，FC-PTV 必须停止，而不是改半径再试。

## 16. 统计、计算成本与证据缺口

- 五种子 8 个训练端点/基线，共 468 个训练 epoch、约 14040 个 optimizer step；训练函数记录的累计时间约 180.39 s。
- 每种子额外 6 次完整 fit-anchor 模型遍历用于 base/shared/classifier/FC/orth-unit/orth-matched 功能诊断，共 30 次数据遍历。
- 保存 70 个 checkpoint，总大小 114.07 MiB；代表性 FC checkpoint 为 1.63 MiB。
- RTX/CUDA 推理基准：batch=128，均值 10.372 ms/batch，即 0.0810 ms/sample，吞吐 12341.2 sample/s；推理峰值 allocated memory 332.2 MiB。
- 正式训练未预先埋点 peak GPU memory，因此该项标记为缺失，不能用推理峰值替代训练峰值，也不为补指标而重跑正式实验。
- 五种子 bootstrap 的样本量很小，CI 只支持 development 级证据；且 v12 与历史数据来自同一生成体系，不是外部域证据。

## 17. 决策、停止规则与下一创新方向

正式决策：**停止 FC-PTV 方向**。不生成 development-v13，不进入全 canonical baseline 阶段，不进行鲁棒性扫描，也不访问任何 test。该失败消耗一个机制周期；它证明“固定 MAD 半径的两块投影”没有实际施加约束。

若未来另立新 Goal，必须是新的可证伪机制而非继续调当前半径。较合理的问题是先验证“功能半径能否从可接受功能漂移/风险边界中获得尺度可识别性”，例如在完全新的开发身份上预注册无监督尺度校准或局部曲率归一化，并把“约束激活率”设为先验机制门槛；在机制门槛未通过前，不再做性能性实验。历史 FullAlpha1p5 仍只能作为当前最强 task-vector 结果，不能包装成 FC-PTV 创新。
