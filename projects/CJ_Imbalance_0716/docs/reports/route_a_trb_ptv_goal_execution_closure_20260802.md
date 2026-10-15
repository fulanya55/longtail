# Route A 配对任务向量效应异质性定位与 TRB-PTV 可靠创新验证：Goal 执行结案

日期：2026-08-02  
工作目录：`C:\LJQ_code\CJ_Imbalance_0716`  
本轮性质：对既有制品的最终核验；未训练、未生成数据、未打开 test 文件。

## 1. 本轮结论

**Goal 结论：`STOP_PAIRED_TASK_VECTOR_STABILIZATION`。**

阶段 A 已按只读边界完成，发现 SawFM 共现条件下 MFT、VFT 的 FullAlpha1p5 弱召回效应差异跨四轮历史制品和 20 个保存 seed 对方向一致，因此具备进入 TRB-PTV 预注册的资格。TRB-PTV 随后完成文献审计、数学/门槛冻结、实现冻结和新鲜身份生成；但冻结 SLSQP 在首个 smoke 的 fit-only 风险约束求解失败。根据预注册，任何非成功状态或约束残差超过容差均令该 seed invalid，且不得修改求解器、约束、初值或重用 v14。

FC-PTV 是第一个已停止的配对任务向量机制循环，TRB-PTV 是第二个。故“配对任务向量稳定化”总方向已达到两次 fresh-development 机制失败的停止线；不得再开启第三个同类循环。

## 2. 最近三项关键结果

|顺序|结果|数据/证据角色|结论|
|---|---|---|---|
|1|FC-PTV：五个 seed 的 `q*=(1.5,1.5)`，功能约束未激活；target weak recall `+0.01808 < +0.025`，三目标 simultaneous family-wise 下界未全为正|已消耗的 development-v12，正式停止事实|不得修改 FC-PTV 重新尝试|
|2|Stage A：SawFM-present 减 SawFM-absent 的 Full−C6 target weak recall 差异，在 MFT、VFT 上均为四轮×五 seed 的 `20/20` 同向；ISRJ 不稳定|v4/v5/v6/v12 历史开发制品，只读、假设生成|`ELIGIBLE_FOR_TRB_PTV_PREREGISTRATION`，不是新机制有效|
|3|TRB-PTV smoke：SLSQP status `8`，返回 `q=(1.5,1.5)`；MFT 风险差 `+0.01018602`，大于 `1e-7` 容差|全新 train-only development-v14；smoke|`stop_trb_ptv_before_formal_evaluation`|

## 3. 唯一科学问题

FullAlpha1p5 在 MFT、VFT、ISRJ 弱分量上的收益是否由可重复的可见性/共现条件控制，而不是对所有数据身份和目标标签一致成立？

阶段 A 的答案仅为：**存在 SawFM 共现关联的条件异质性线索，且它在 MFT、VFT 上足以触发预注册。** 该结论不能升级为相对功率、SNR/JSR 或因果可见性机制，因为保存制品没有相应逐样本字段。

## 4. 可证伪假设与实际裁决

TRB-PTV 假设：若以 C6 fit-only 的 MFT/VFT/ISRJ 弱分量风险为不可恶化边界，最接近 Full1.5 的 shared/classifier 分块系数会产生与 `(1.5,1.5)` 有实质差异且数值可行的解，并在正式五种子评价中提升最差目标标签的弱召回。

裁决：**在冻结 smoke 的数值可行性先决条件处失败。** 因而未获得进入正式性能门槛的资格；不能把返回的中止点当作 TRB 结果，也不能用新的实现细节掩盖求解器契约失败。

## 5. 数据与身份隔离

|项目|已核验事实|
|---|---|
|Stage A|仅读取 v4/v5/v6/v12 的 summary、metrics、threshold、calibration、预测类制品、manifest、receipt 和报告；未重跑历史模型|
|v14 样本种子|`144000001–144025200`，25,200 个样本|
|v14 receiver identity|`342677743–342702942`，25,200 个且全部唯一|
|不重叠审计|与 development-v2 至 v13 预约身份、对应 test 契约以及 ISRJ parameter-OOD 身份均为不相交|
|训练数据完整性|`closedset_6class_train.mat`：2,549,133,388 bytes，SHA256 `7fdf86cb45feaef81ed8325a60ef99da4e5ab9f06cf1911b0815407700d35e58`|
|测试边界|identity audit 与 smoke receipt 都为 `test_files_opened=false`、`locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`、`test_v14_used=false`|

九个 power view 仍按 `receiver_noise_seed` 作为同一身份的配对视图，而非独立样本。

## 6. 基线和候选

预注册的正式公平比较清单为：`C6_Joint`、`M5_GradientIsolated`、`FullAlpha1p5`、`TRB-PTV`、`ParameterNormOnly`、`ClassifierOnlyAlpha2p0`、`ModelAverage0p5`、`CAGrad0p4TwoTask`、`OrthogonalTaskSubspaceFunctionalMatched`。

它们原本必须共享初始化、split、manifest、batch 顺序、训练预算、早停、校准、阈值、种子、可训练范围和数值策略。TRB 在 fit-only 求解阶段已失效，故没有任何正式多种子比较，也不把 smoke 产生的局部制品纳入比较。

## 7. 冻结方法公式

令 `d = theta_M5 - theta_C6 = d_shared + d_classifier`，仅允许冻结的 `temporal_backbone.0.*` 与 `classifier.*` 非零。冻结机制为：

`q* = argmin_q ||q - (1.5, 1.5)||_2^2`

满足 `0 <= q_shared <= q_classifier <= 1.5`，并对 `t ∈ {MFT, VFT, ISRJ}` 约束：

`L_weak,t(theta_C6 + q_shared*d_shared + q_classifier*d_classifier) <= L_weak,t(theta_C6)`。

所有损失只由 fit 角色中该目标标签且预先固定弱分量定义的样本计算。求解器冻结为 float64 SLSQP，`q0=(1.5,1.5)`、`maxiter=200`、`ftol=1e-12`，再以独立 float64 前向复核，约束容差 `1e-7`。激活门固定为至少 `4/5` seed 满足 `||q*-(1.5,1.5)||_2 >= 0.05`。

## 8. 实现改动与冻结核验

TRB 核心、v14 identity、生成器、runner、配置、预注册 manifest 及三项单元测试均存在。冻结清单记录 11 个实现/计划/测试文件；本次重新计算确认它们的当前 SHA256 均匹配 repair-v2 freeze。

验证命令：

`D:\anaconda3\envs\PY10\python.exe -m pytest -q tests\test_route_a_trb_ptv_core.py tests\test_route_a_v14_identity.py tests\test_freeze_route_a_trb_ptv_v14_implementation.py`

结果：`7 passed in 0.41s`。

## 9. Smoke 结果

smoke 仅使用 seed 2039、1 epoch、截断角色，目的只是验证冻结路径和 fit-only 求解契约。SLSQP 返回 status 8（`Positive directional derivative for linesearch`），且 returned q 保持 `(1.5,1.5)`。MFT/VFT/ISRJ 的风险差相对 C6 分别为 `+0.01018602`、`-0.01626149`、`-0.03557511`；最大约束违反为 `0.01018602`。

该数值远大于 `1e-7`，故 smoke 结果是失败收据，不报告任何 macro-AUPR、macro-F1、校准、阈值或方法性能。

## 10. 正式多种子结果

**未启动，且按规则不得启动。** `formal_model_evaluation_started=false`、`formal_five_seed_started=false`。因此不存在正式 TRB 的均值、CI、最差 seed、最差标签或对照比较；所有此类字段应写为“未评价”，不能视为零、失败性能或缺失后可补算项目。

## 11. 配对 CI、seed 一致性、最差 seed/标签

- FullAlpha 历史开发结果保留其既有 CI 与跨 seed 差异作为阶段 A 的描述性线索。
- TRB 没有通过 solver 可行性，故没有合法的五种子配对 CI 或最差标签统计。
- FC 的 family-wise 未通过与 TRB 的数值无效是不同停止原因；二者均不能被选择性忽略。

## 12. 分标签弱分量结果

Stage A 已对 MFT、VFT、ISRJ 分别报告每轮/每 seed 的 weak recall、AUPR、FPR 与最差配对。关键条件线索为 MFT、VFT 的 SawFM presence difference-in-difference 一致；ISRJ 不满足相同稳定性。TRB 只产生 fit-only 风险诊断，不能产出可比较的分标签正式识别结果。

## 13. 机制分析

可支持的解释：FullAlpha 的相对收益在 SawFM 共现条件间不均一，且 MFT、VFT 的保存预测呈稳定差异，因此平均端点可能掩盖目标标签的风险差异。

不能支持的解释：纯 shared 梯度冲突抑制、相对功率/SNR 因果机制、所有条件的一致收益，或 TRB 已经解决最差标签风险。TRB 的风险约束想法具有可证伪性，但被冻结的数值契约未能给出合法解。

## 14. 失败与替代解释

1. 可能存在数学可行锚点 `(0,0)`，但这不改变预注册：正式候选必须由固定 SLSQP、固定初值和严格复核产生；status 非成功或残差超容差即 invalid。
2. 返回 `(1.5,1.5)` 不等于“约束未激活”：它伴随 MFT 约束违反，不能重新命名为 FullAlpha 成功。
3. Stage A 的 SawFM 条件差异可能混有组合频率、标签难度、阈值/校准效应；报告已明确这些替代解释和缺失的功率/SNR字段。

## 15. 预注册门槛执行表

|门|状态|依据|
|---|---|---|
|阶段 A 至少两个标签的跨 seed 条件差异|通过|MFT、VFT 满足严格 20/20 同向规则|
|文献差异矩阵、数学定义、新颖性边界、门槛、实现 SHA 和新身份方案|通过|预注册与 implementation manifest 已冻结|
|TRB solver 有限、成功且约束残差合格|失败|status 8，最大违反 `0.01018602 > 1e-7`|
|TRB 激活门|未评价|solver 失败后不可定义合法 `q*`|
|相对 C6 / FullAlpha 的 12+8 项正式性能门|未评价|正式五种子被禁止启动|
|阶段 C/D/E|不适用且禁止|TRB 未通过，test-v3 继续密封|

## 16. 证据等级

- `[F]`：文件哈希、identity audit、冻结清单、smoke receipt、测试结果和历史指标。
- `[A]`：基于保存预测的 Stage A 条件异质性描述；仅能生成假设。
- `[D]`：TRB 与整个 PTV 稳定化方向的停止决定，直接执行预注册 stop rule。

## 17. 可追溯制品

- Stage A 报告：`docs/reports/route_a_fullalpha_effect_heterogeneity_pretrb_audit_20260801.md`
- Stage A JSON：`outputs/tapc/route_a_fullalpha_heterogeneity_audit/heterogeneity_audit.json`
- 文献审计与预注册：`docs/plans/route_a_trb_ptv_literature_audit_and_fresh_development_preregistration_20260801.md`
- 实现 manifest：`docs/plans/route_a_trb_ptv_implementation_manifest_20260801.json`
- repair-v2 freeze：`outputs/tapc/route_a_trb_ptv_implementation_repair_v2_freeze/implementation_freeze.json`
- v14 identity audit：`outputs/tapc/route_a_development_v14_generation/identity_audit.json`
- TRB 失败 receipt：`outputs/tapc/route_a_development_v14_trb_ptv_smoke_v2/run_receipt.json`
- TRB 失败报告：`docs/results/route_a_development_v14_trb_ptv_smoke_failure_20260801.md`

## 18. 下一步唯一决定

**停止“配对任务向量稳定化”研究方向。** 不生成 v15 作为 TRB 重试，不修改 FC/TRB 的 alpha、约束、solver、初值、激活阈值或门槛，不重用 v12/v14，也不访问 test-v3。

若未来开展新研究，必须另立一个与 PTV 稳定化实质不同的科学问题，并先完成文献冲突审计、可证伪机制、身份隔离方案和预注册；它不属于本 Goal 的延续。
