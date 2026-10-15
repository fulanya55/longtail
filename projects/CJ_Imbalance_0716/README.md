# CJ Imbalance 0716

SawFM stage2同身份功率Teacher–Student试验已完成：rank-8专项证据头配合组合匹配hard negative，在45 dB独立development上达到AUROC 0.7972、AUPR 0.7949，泛化落差0.0097；hard negative相对teacher+随机负样本提高AUROC 0.1739。Oracle门通过后执行的最大0.20受限H2残差只提高AUROC 0.00219，说明下一瓶颈是SawFM专家路由而非stage2特征提取。结果见`docs/results/route_a_sawfm_stage2_power_teacher_student_results_20260725.md`。

F2 特征充分性与硬约束弱证据残差试验已完成：H2 仍是总体最稳的弱证据读取器（弱 AUROC 0.6524）；stage2 物理统计只在 SawFM 上显示有限潜在增量（oracle 弱 AUROC 0.5586，相对 H2 +0.0382），但弱正 probe 存在 0.33～0.37 的内部验证到独立 development 泛化落差，四个受限残差变体均未通过晋级门，继续执行 `stop_before_unfreeze`。结果见 `docs/results/route_a_dfms_iq_feature_sufficiency_results_20260725.md`。

DFMS-IQ标签条件局部证据结构筛选已完成：LabelQuery在label-LT清单上学习了显著逐类logit先验偏移，top-k局部旁路未改善弱正排序，stage2 SawFM专家因输出无幅度约束而破坏SawFM弱AUROC；Q1/Q2/Q3均未通过门槛，未解冻主干。结果见 `docs/results/route_a_dfms_iq_label_local_saw_results_20260724.md`。

DFMS-IQ 冻结分类头阶段已完成：冻结重训 DB 头证明旧联合头未充分利用弱证据，pos-weight BCE 取得当前最佳冻结结果（Macro-AUPR 0.8670、弱正/负 AUROC 0.6524）；统一弱排序与共享型 SawFM 专家未提供独立增益，预注册门槛未通过，因此未解冻主干末级。结果见 `docs/results/route_a_dfms_iq_frozen_heads_results_20260724.md`。

DFMS-IQ正式筛选已完成：完整F2取得当前最高macro-AUPR 0.8562和最低Hamming 0.2485；冻结线性probe接近B1，但主工作点弱召回仍低，说明瓶颈已从早期I/Q融合转向强正主导的分类目标、阈值和SawFM特异弱证据。结果见 `docs/results/route_a_dfms_iq_structure_screen_results_20260724.md`。

最新主干候选 DFMS-IQ 已实现：I/Q 复数双流保持到长度375，使用复数多尺度分支和延迟 Cartesian/幅度/多滞后相位融合；参数量257,511，与 B1 容量匹配。方案见 `docs/implementation/route_a_dfms_iq_delayed_fusion_network_plan_20260724.md`。

最新路线 A 结构筛选：PAM-IQ V2 的全局旁路降低了误报，标签查询恢复了部分弱召回但同步增加误报，未通过相对 Base-DB 的预注册门槛；第二阶段弱正损失按规则暂停，详见 `docs/results/route_a_pam_iq_v2_structure_screen_results_20260724.md`。

最新路线 A 骨干筛选：PAM-IQ 相对 Base-DB 提高 SawFM-AUPR，但未通过容量匹配 WideCapacity 对照的 macro-AUPR、weak recall 与 Hamming 综合门槛；暂不接入功率可见性轴，结果见 `docs/results/route_a_pam_iq_backbone_screen_results_20260724.md`。

固定 WeightFusion-0.2 未通过新种子确认门槛，Base-DB-Macro 继续作为六类历史主基线；结果见 `docs/results/route_a_sawfm_weight_fusion_confirmatory_results_20260723.md`。

新特征骨干候选 PAM-IQ 已完成实现与 GPU 冒烟：深层复数残差、相位增量、多尺度时序编码、稀疏 ECA/PCAF/GC 和注意力统计池化；设计与验证计划见 `docs/implementation/route_a_phase_aware_multiscale_backbone_plan_20260723.md`。

面向 0716 雷达复合干扰数据的 long-tailed multi-label、弱成分、半监督与参数感知识别实验工程。

## 数据与安全边界

- 原始数据只读：`C:\LJQ_code\干扰数据\复合干扰数据0716`
- `C:\LJQ_code` 下其他算法程序可作为参考；除本工程外不覆盖历史原程序。
- 根训练集与功率扫描训练集共享同一批 `receiver_noise_seed`；功率文件只能作为同身份配对视图。
- `closedset_6class_test.mat` 和功率扫描 test 文件在配置冻结与用户授权前禁止读取。
- 当前四类 pilot：FMB、SinFM、MFT、ISRJ，并严格排除含 SawFM/VFT 的样本。
- 六类正式阶段：FMB、SawFM、SinFM、MFT、VFT、ISRJ。

## 两条研究路线

- 论文 1 / TAPC-CJNet：三轴长尾、标签条件 prototype、identity-safe 功率方向排序、稳定性、参数与决策接口，正式入口为 `experiments/run_tapc.py`；旧版最小消融入口保留为 `experiments/run_student_a.py`。
- 师弟 B：DAPS-CJNet-v2，包含 nested labeled split、EMA Teacher、逐标签伪标签、prototype reliability 与 label-gated parameter heads，入口为 `experiments/run_student_b.py`。

两条路线共用复数 I/Q backbone、五路数据角色、CB-ASL、温度校准、阈值选择与多标签评价接口，但论文贡献和消融矩阵相互独立。

## 六类阶段主要对比基线

- 路线 A 的六类主要对比基线为 **Base-DB**：随机初始化、端到端训练、完整六标签监督和 DB Loss。
- Base-ASL 保留为传统损失参照，但新候选必须首先报告相对 Base-DB 的同 seed、同 split、同 manifest 配对结果。
- 当前晋级原则同时约束 macro-AUPR 非劣性和目标尾类 AUPR，禁止仅凭单个尾类提升替换主基线。
- 基线聚合入口：`experiments/aggregate_route_a_six_class_basedb.py`；基线结论见 `docs/results/route_a_six_class_db_path_results_20260723.md`，SawFM 损失与选点归因见 `docs/results/route_a_sawfm_loss_selection_factorial_results_20260723.md`，早期—后期融合结果见 `docs/results/route_a_sawfm_checkpoint_fusion_results_20260723.md`。

## 环境与测试

```powershell
D:\anaconda3\envs\PY10\python.exe -c "import torch,h5py; x=torch.randn(64,64,device='cuda'); print(torch.__version__, torch.cuda.get_device_name(0), float((x@x).mean()))"
D:\anaconda3\envs\PY10\python.exe -m pytest -q
```

## 四类运行示例

```powershell
D:\anaconda3\envs\PY10\python.exe experiments/run_tapc.py `
  --config configs/student_a/tapc_four_class.yaml `
  --variant T4 --protocol mixed_lt --loss cb_asl `
  --epochs 60 --batch-size 64 --amp

D:\anaconda3\envs\PY10\python.exe experiments/run_student_b.py `
  --config configs/student_b/daps_param.yaml `
  --variant B6 --labeled-ratio 0.20 `
  --epochs 100 --batch-size 64 --amp
```

## 运行计划与冻结

```powershell
D:\anaconda3\envs\PY10\python.exe experiments/run_matrix.py --output planned_jobs.csv
D:\anaconda3\envs\PY10\python.exe experiments/freeze_configuration.py `
  --output-dir frozen/20260721_pilot_v1
```

每个 TAPC run 自动保存 `scientific_config.json`、`environment.json`、`events.jsonl`、长尾清单及哈希、`history.csv`、模型、校准、阈值、预测、分层指标、参数误差和成功/失败 receipt。实验设置变化会产生新的配置 SHA-256，禁止静默覆盖旧结果。

`planned_jobs.csv` 当前登记 91 个 pilot/formal development 任务；`frozen/20260721_pilot_v1/configuration_freeze.json` 保存 8 个配置文件的 SHA-256，并明确 `locked_test_authorized=false`。

## 当前证据边界

`docs/results` 同时包含 smoke、screen 和完整 development 运行，必须以每份文档记录的协议边界为准。当前 Base-DB 晋级依据来自六个尾类 × 3 seeds 的 development 配对结果，locked test 仍未使用；最终论文性能主张仍须满足预注册种子数、均值/标准差/配对差及冻结后的单次正式测试。
