# 路线 A 整体实验与当前网络结构审计 Prompt（2026-07-26）

## 使用目标

本 Prompt 用于系统整理路线 A 的数据协议、实验演进、当前网络、训练方式、正式指标、失败机制、证据边界和下一决策。它适合交给科研智能体、代码审计智能体或论文写作智能体直接执行。

执行时不得把历史候选当作当前网络，不得把 development 结果写成 locked-test 结果，也不得把 0716 根数据直接称为长尾数据。

## 可直接复制的专业 Prompt

```text
你是一名专注于雷达复合干扰识别、复数 I/Q 时序网络、多标签长尾学习、弱分量检测、概率校准和可复现实验审计的高级科研人员。请以 SCI 论文证据链、可复现工程和可证伪研究为标准，对“路线 A”进行整体审计和专业中文整理。

一、项目边界

项目根目录：
C:\LJQ_code\CJ_Imbalance_0716

主要环境：
D:\anaconda3\envs\PY10\python.exe

数据根目录：
C:\LJQ_code\干扰数据\复合干扰数据0716

六类固定顺序：
[FMB, SawFM, SinFM, MFT, VFT, ISRJ]

输入：
RMS 归一化真实复数 I/Q，形状 [B,2,3000]。

必须遵守：
1. 0716 根训练集包含 63 个非空标签组合，每组合 400 个样本，共 25200 个样本，根数据本身平衡。
2. 路线 A 的标签长尾由可复现 label-LT manifest 构造；当前审计配置为 seed=2043、训练样本 2400。按头到尾顺序 `FMB→SinFM→MFT→VFT→ISRJ→SawFM` 的计数为 `1600/1213/919/696/528/400`；按固定代码类别顺序 `[FMB,SawFM,SinFM,MFT,VFT,ISRJ]`，manifest 中 `target_label_counts=achieved_label_counts=[1600,400,1213,919,696,528]`。报告时必须注明顺序，防止错位。
3. power sweep 是同一 receiver_noise_seed 身份的配对视图，不能当作独立样本跨 split。
4. train、head validation、ordinary calibration、ordinary threshold、ordinary early、power fit/threshold/development 必须按现有产物审计其身份隔离。
5. locked test 当前未使用。没有用户单独明确授权，不得读取或运行 locked test。
6. 优先核对 summary JSON、predictions、run receipt、manifest/split/checkpoint SHA256；Markdown 只作为解释层。

二、必须优先复核的最新证据

1. 最新五种子结果：
C:\LJQ_code\CJ_Imbalance_0716\docs\results\route_a_raw_platt_five_seed_label_routing_results_20260726.md

2. 最新正式汇总：
C:\LJQ_code\CJ_Imbalance_0716\outputs\tapc\route_a_raw_platt_five_seed_label_routing\route_a_raw_platt_five_seed_label_routing_summary.json

期望 SHA256：
1C456FF579F0FA2EFCCBA202152F5F2CA117CA215A7D84131447D4086D491EFC

3. 当前执行程序：
C:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_raw_platt_five_seed_label_routing.py

4. 当前主干与专家源码：
C:\LJQ_code\CJ_Imbalance_0716\cjlt\models\phase_multiscale.py
C:\LJQ_code\CJ_Imbalance_0716\cjlt\models\evidence_readout.py

5. 历史关键结果：
- route_a_six_class_db_path_results_20260723.md
- route_a_pam_iq_backbone_screen_results_20260724.md
- route_a_pam_iq_v2_structure_screen_results_20260724.md
- route_a_dfms_iq_structure_screen_results_20260724.md
- route_a_dfms_iq_frozen_heads_results_20260724.md
- route_a_dfms_iq_feature_sufficiency_results_20260725.md
- route_a_sawfm_stage2_power_teacher_student_results_20260725.md
- route_a_sawfm_fixed_fusion_cross_power_results_20260725.md
- route_a_stable_expert_visibility_calibration_results_20260725.md
- route_a_selective_unknown_decision_results_20260725.md
- route_a_stage2_reliability_identifiability_results_20260725.md
- route_a_early_complex_sawfm_identifiability_and_confirmation_results_20260725.md
- route_a_trainable_raw_frontend_and_new_split_confirmation_results_20260725.md
- route_a_h2_raw_frontend_bounded_reconnection_results_20260725.md
- route_a_raw_frontend_independent_classifier_audit_results_20260725.md

三、首先回答的核心问题

1. 路线 A 现在研究的科学问题是什么？必须分开“标签/组合频率偏置”和“功率可见度退化”。
2. 从 Base-ASL/Base-DB 到 PAM-IQ、F2、H2、Stage2 专家、校准/拒判、Raw 前端、Raw 独立判决和五种子标签路由，实验如何演进？
3. 每个阶段的科学假设、主要对照、关键指标、是否通过、失败根因和下一门禁是什么？
4. 当前候选到底是单一端到端网络，还是多个冻结模块构成的标签级路由系统？
5. 当前候选包含哪些模块，哪些历史模块已经退出最终路径？
6. 当前结果属于 development、locked test 还是 proposal？
7. 当前唯一合理的下一动作是什么？

四、当前候选网络必须按下述口径复核

当前 development 候选为：

F2_DFMS_Full + H2_PosBCE（FMB、SinFM、MFT、VFT、ISRJ）
+ RawStage1SawFMFrontend(mode="raw") + Platt（SawFM）
+ 固定标签级路由 RawSaw+H2Others。

它不是端到端联合训练的单网络，而是冻结、可解释的标签级组合系统。

1. F2_DFMS_Full
- 输入 [B,2,3000]；width=16；feature_dim=128；dropout=0.20。
- stage1：复数残差 1→16，kernel=7，长度 3000。
- AvgPool stride=2。
- stage2：复数残差 16→32，kernel=5，长度 1500。
- stage3：复数多尺度 32→48，kernel 3/7/15，dilation 1/2/4，长度 750。
- stage4：复数多尺度 48→64，长度 375。
- I/Q 到 stage4 仍保持独立 real/imag 复数流。
- 晚期融合：Cartesian + magnitude + lag 1/4/16 phase。
- 128 通道多尺度 depthwise residual。
- GlobalStatisticsPooling mean/std：256 维。
- Linear + LayerNorm + GELU + dropout：128 维共享表征。
- 原六标签分类器在当前最终路由中不直接使用。
- 含原分类器总参数 257511；特征提取部分 256737。

2. H2_PosBCE
- F2 完全冻结并保持 eval。
- Linear(128,6)，774 参数。
- pos_weight BCE，60 epochs，AdamW，lr=1e-3，weight_decay=1e-4，batch=256，patience=12。
- 当前最终输出只保留 FMB、SinFM、MFT、VFT、ISRJ 五列；H2 的 SawFM 仅作对照。

3. RawStage1SawFMFrontend(mode="raw")
- 只读原始 RMS-normalized I/Q，不读真实功率/JNR/receiver_noise_seed。
- 六通道物理描述：幅度均值、最大值、标准差、相邻幅度差、相邻归一化相位增量 cos/sin。
- Conv1d 1×1：6→16。
- depthwise temporal kernels 7/31/63。
- 1×1 融合回 16 通道，AvgPool stride=4。
- 编码特征和原始描述分别做 mean/std/max/top-k mean，合成 88 维。
- MLP 88→32→1，输出 SawFM logit。
- hidden_channels=16，topk=16，dropout=0.1，总参数 5521。

4. Platt 与阈值
- ordinary calibration 上拟合 LogisticRegression(C=1.0, solver=lbfgs)。
- 仅输入 Raw logit。
- ordinary threshold 与 40/45/50 dB power threshold 联合搜索统一阈值，grid=0.01。
- 所有 threshold 域 FPR≤0.20。
- ordinary early 与 power development 只评价，不二次调参。

5. 标签路由
- SawFM：Raw+Platt probability + Raw threshold。
- 其他五类：H2 probability + H2 per-label threshold。
- 不训练 gate、不加权融合、不使用 oracle power、不按标签组合路由。
- 有效执行参数约 263034，包括 F2 特征提取、H2、Raw 和两参数 Platt；不计未使用的 F2 原分类头。

五、实验演进必须按“阶段—指标—根因—下一门禁”整理

至少覆盖：
1. Base-DB 主基线与 ASL/混合损失。
2. PAM-IQ 与容量匹配 B1。
3. PAM-IQ V2 的全局旁路/标签查询。
4. F2 延迟融合与多滞后相位。
5. H2 冻结头与 Q1/Q2/Q3/特征充分性负结果。
6. Stage2 SawFM 专家、固定 C2 融合和路由/校准/ordinal/unknown/reliability 失败链。
7. 早期 Raw/Stage1 可辨识性筛选。
8. 可训练 Raw-only 结构、phase/envelope/time-shuffle 消融。
9. Raw→H2 有界残差失败。
10. Raw 独立分类器成功。
11. 五种子 RawSaw+H2Others 标签路由。

六、最新结果必须准确报告

Raw+Platt 五种子：
- ordinary AUROC 0.9384；AUPR 0.9547；recall 0.8813；FPR 0.1960。
- without-swept recall 0.8653。
- power mean recall 0.9879；50 dB recall 0.9758；max power FPR 0.0760。
- Brier 0.0977；ECE 0.0504。
- 单分类器门禁 4/5 seeds 通过；seed 2039 ordinary early FPR=0.2806 是唯一失败项。

六类标签路由五种子：
- macro-AUPR：0.8667→0.9001，Δ=+0.0334。
- macro-F1：0.7573→0.8027，Δ=+0.0454。
- Hamming：0.2307→0.1959。
- SawFM recall：0.4870→0.8813，Δ=+0.3943。
- exact match：0.2280→0.2747，Δ=+0.0467。
- 其他五类最大 FPR 变化：0。
- 路由门禁 5/5 seeds 通过。

必须同时披露：
- seeds 2039/2040 使用 R0_FullPhysics checkpoint；2041–2043 使用 N0_RawOnly checkpoint。
- 网络与优化超参数相同，但训练角色抽样协议不同，不能写成完全同协议五次重复。
- locked_test_used=false。

七、输出格式

请输出专业中文报告，至少包括：
1. 500 字内执行摘要。
2. 数据与审计边界。
3. 实验演进总表：阶段、假设、对照、关键指标、结论、根因、下一门禁、状态。
4. 当前网络结构图，标明张量尺寸、参数量、冻结/训练状态和最终标签路由。
5. F2/H2/Raw/Platt/threshold 的训练与校准流程表。
6. 最新五种子结果及 bootstrap 95% CI。
7. 已验证事实、历史负结果、尚未验证假设三张独立清单。
8. 当前网络相对论文目标的优势、局限和创新表述边界。
9. seed 2039 FPR 风险和 Raw checkpoint 协议异质性的专项说明。
10. 当前状态判定与唯一下一动作。

八、写作和研究纪律

- 排序指标 AUROC/AUPR 与阈值后 recall/FPR 分开报告。
- development 与 locked test 分开报告。
- proposal 与 validated result 分开报告。
- 不删除失败结果，不以平均值掩盖最差 seed。
- 不把 power views 称为独立样本。
- 不把根平衡数据称为天然长尾。
- 不使用 ordinary early 或 locked test 重新选择模型、校准器和阈值。
- 不把 Stage2/C2/A0/Unknown 等历史模块画入当前最终路径；它们只能出现在实验演进和负结果/机制结果中。
- 若给出下一步建议，必须服从当前冻结决策：只允许审查 locked-test 申请包；未经明确授权不得执行测试。
```

## 本轮执行说明

本 Prompt 已用于生成：

`docs/reports/route_a_overall_experiment_network_status_20260726.md`
