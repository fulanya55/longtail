# Route A H5o：冻结统一候选的联合 OOD 外部确认结果

日期：2026-08-03  
决策：**未通过；关闭该冻结候选在 H5o 联合 OOD 条件下的确认，不作任何 H5o 后修复。**

## 预注册问题与边界

H5o 只检验一个问题：已经冻结的
`H1_B2_MFTSharedDecision_VFTRidgeFusion_v2` 是否能在新 identity、ISRJ 高副本参数
区与 `rho=0.65` proper-complex AR(1) 噪声的联合条件下，维持 H3 同一套冻结统一门。

- 五个训练 seed 为 2039--2043；Q0/Q1--Q6、backbone、RawSaw、三个物理路径、
  校准器、阈值和共享 MFT 决策均与 H3 完全冻结。
- H5o 新来源包含 5,040 条标准六类记录（63 组合 × 80），5,040 个唯一 receiver
  identity，范围 `1048677743--1048682782`，与全部可读历史 train-only 来源零重合。
- ISRJ 参数为官方生成器允许的高副本区 `25--30 us`、9 副本；I/Q 仅替换接收机
  噪声为等功率 proper-complex AR(1)，`rho=0.65`。标签、参数、种子及非 I/Q
  数据集全部保持不变。
- H5o 中 `external_refit`、`hyperparameter_search`、`calibration_refit` 与
  `threshold_refit` 都为 `false`；locked test、test-v2、test-v3 均未使用。

该来源是项目内联合 OOD 压力条件，并非官方 0716 的替代测试集，也不能替代最终
一次性 test-v3。

## 生成与审计收据说明

MATLAB 基础生成及 AR(1) 物化成功生成了基础 I/Q 和 AR 负载，但二者的原始 formal
JSON receipt 意外为零字节。没有重写或重生成任何 I/Q。后续只读恢复记录保留该事实，
并验证既有 base/payload 的 hash、shape、63×80 标签合同、receiver 区间、ISRJ 参数；
独立 MATLAB 过程验证给出 200,000 点 probe：平均功率 `1.00208`、伪协方差模
`0.00569`、lag-1 相关 `0.65225`。因此数据合同通过，但该零字节收据是应在论文
工件表中透明披露的工程异常，不能被抹去或写成“原始 formal receipt 完整”。

## 五 seed 冻结结果

Q6 相对 Q0 的均值：

| 终点 | H5o Q6 - Q0 或绝对值 |
|---|---:|
| macro-AUPR | `+0.023760` |
| macro-F1 | `+0.033492` |
| MFT/VFT/ISRJ 平均弱召回 | `+0.017662` |
| MFT/VFT/ISRJ 平均 weak-vs-negative AUROC | `+0.044064` |
| 三目标平均 FPR | `0.213360` |
| 三目标平均强正召回 | `+0.068543` |
| MFT / VFT / ISRJ AUPR | `+0.005672 / +0.014850 / +0.122036` |
| SawFM AUPR、弱召回 | 精确不变 |

五个 seed 的三目标弱召回差均为正，但为
`+0.01543`、`+0.00756`、`+0.03596`、`+0.02119`、`+0.00816`，均值未达到预注册的
`+0.030`。三目标 FPR 超限主要来自 VFT：对齐 Q4 的 VFT FPR 均值为 `0.328065`，
超过 `0.20`；Q4 相对 Q2 的 VFT 弱召回仅 `+0.007445`，低于 `+0.010`。

MFT 共享决策仍保留了其结构安全性质：hard-negative FPR `-0.06708`、order-5 FPR
`-0.07750`、hard AUROC `+0.02391`，零 negative-to-positive 翻转，所有 seed 均降低
hard FPR。但其 MFT 弱召回代价为 `-0.03993`，超过允许的 `-0.020`。

## 门判定与解释

| 门 | 判定 | 原因 |
|---|---|---|
| 统一提升门 | 失败 | 弱召回 `+0.01766 < +0.030`，目标 FPR `0.21336 > 0.20` |
| VFT 对齐机制门 | 失败 | Q4-Q2 弱召回不足，VFT FPR `0.32806 > 0.20` |
| MFT 共享安全门 | 失败 | 虽降低 hard FPR 且无负转正，但弱召回损失 `-0.03993 < -0.020` |
| 数据/冻结/密封合同 | 通过 | 新身份、AR(1)、非 I/Q 保持、工件 hash、无 refit、test-v3 seal 均通过 |

这说明该冻结候选的 ISRJ 排序增益在新联合 OOD 下仍然很强，但 VFT 固定阈值运输
产生 FPR 超限，同时 MFT 安全规则在该背景中牺牲过多弱召回。由于所有决策规则在
H5o 前已经冻结，不能将此结果归因于阈值选择，也不能在 H5o 上调整阈值来“修复”。

H5o 不推翻 H3 的新身份标准源确认；两者共同给出的正确边界是：候选在同标准生成器
新身份上可复现，但尚未建立完整的参数/噪声 OOD 鲁棒性。它不能修复 C3/C4 的容量
因果失败，也不授权 test-v3。

## 工件

- 预注册：`docs/plans/route_a_h5o_frozen_unified_combined_ood_20260803.md`
- 生成前 identity 审计：
  `outputs/tapc/route_a_h5o_frozen_unified_ood_generation/pre_generation_identity_selection.json`
- 正式来源审计：
  `outputs/tapc/route_a_h5o_frozen_unified_ood_generation/identity_parameter_noise_audit.json`
- 零字节收据恢复与 AR 过程验证：
  `outputs/tapc/route_a_h5o_frozen_unified_ood_generation/formal_base_generation_recovery_receipt.json`、
  `outputs/tapc/route_a_h5o_frozen_unified_ood_generation/formal_ar1_materialization_recovery_receipt.json`、
  `outputs/tapc/route_a_h5o_frozen_unified_ood_generation/posthoc_ar1_process_verification_receipt.json`
- 冻结预测与总门：`outputs/tapc/route_a_h5o_frozen_unified_ood/summary.json`
