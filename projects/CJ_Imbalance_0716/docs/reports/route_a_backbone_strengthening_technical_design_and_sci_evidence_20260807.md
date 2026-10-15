# Route A Backbone 强化：技术设计、实验闭环与 SCI 证据边界

日期：2026-08-07  
最终状态：`completed_scientific_stop`  
适用范围：新的 receiver-identity 隔离 `backbone-development-v1`，不是独立测试结论。

## 1. 研究问题与基线

目标是验证：在不改变训练目标、分类器、校准或阈值规则的条件下，复数 I/Q backbone 是否能更可靠保留 MFT、VFT、ISRJ 的弱分量局部证据。

基础模型 B0 为 `F2_DFMS_Full`：输入 `[B,2,3000]`，complex RMS 归一化；复数 residual stem 后依次形成 Stage2 `[B,32,1500]`、Stage3 `[B,48,750]`、Stage4 `[B,64,375]`；Stage4 的 Cartesian、magnitude 和 phase-increment late fusion 输出 `[B,128,375]`；原 temporal multi-scale block 后经 global mean/std 统计、128 维投影和六标签线性分类器输出 logits。B0 有 257,511 个可训练参数。

固定协议为五个训练 seed `2039–2043`，receiver identity 在 fit/validation/early/calibration/threshold 五个角色间完全隔离；Focal `gamma_pos=0,gamma_neg=3`，五个非 SawFM 标签均值加 `0.2*SawFM`；AdamW、backbone LR `5e-5`、classifier LR `5e-4`、weight decay `1e-4`、36 epoch、patience 8。逐标签 Platt 仅用 calibration，阈值仅在 threshold 角色上以 `FPR<=0.20` 最大化弱正召回，early 角色只报告。

## 2. 四个单机制候选

| 候选 | 机制 | 参数 | 关键约束 |
|---|---|---:|---|
| M1 | 三次下采样前的固定复数 BlurPool `[1,4,6,4,1]/16` | 257,511 | 实虚共享固定核；无新训练参数 |
| M2 | Stage2/3/4 的 real-imag-magnitude 金字塔对齐与 128 通道融合 | 310,376 | `0.25*tanh(a)` 残差，`a=0` |
| M3 | `[B,128,375]` 上 3/1、15/2、31/4 depthwise EPB | 313,191 | 普通 residual，无 attention/gate |
| M4 | global mean/std 外的 max + top-k mean 局部统计 | 290,664 | `0.25*tanh(b)`，`b=0`；k=8/16 一次 validation 选择 |

M2/M4 均验证：同 seed 下 B0 共同参数逐字节一致，且 alpha/beta 为 0 时模型输出与 B0 完全一致。M4 仅用 seed-2039 的 validation history 一次性选择 top-k=16；未使用 early/calibration/threshold 指标选型。

## 3. 统一评价与门槛

每个候选从 scratch 训练，并与同 seed B0 配对。需要同时达到：macro-AUPR `+0.005` 且 CI 下界>0；三目标平均 weak-vs-negative AUROC `+0.020` 且 CI 下界>0；弱召回 `+0.025`；两个弱指标至少 4/5 seed 为正；绝对平均 FPR<=0.20；强正 recall 不下降超过 0.01；任何目标 AUPR 不降超过 0.01；并且 FMB/SawFM/SinFM 无系统性退化。

只有满足全部门槛才允许执行同容量控制、信息破坏控制、模块组合、冻结表征 probe 和全新身份 confirmation。该顺序避免把后续分析当成失败模块的修补筛选。

## 4. 结果

| 模型 | Δmacro-AUPR | Δ目标弱 AUROC | Δ目标弱 recall | Δ目标强 recall | 目标平均 FPR | 结论 |
|---|---:|---:|---:|---:|---:|---|
| M1 | +0.004655 | -0.000354 | +0.002794 | -0.000119 | 0.194677 | 失败：SinFM/VFT 退化 |
| M2 | -0.000002 | +0.000447 | +0.001075 | -0.000259 | 0.193871 | 失败：跨层残差几乎未被使用 |
| M3 | -0.000375 | +0.006936 | +0.008218 | +0.005884 | 0.191828 | 失败：FMB/SinFM 系统性退化 |
| M4 | -0.000404 | -0.000018 | -0.003301 | -0.004142 | 0.193871 | 失败：目标弱召回 5/5 seed 下降 |

M3 是唯一表现出部分正信号的候选：VFT 与 ISRJ 的局部弱证据排序有改善，但这种效果不足以构成统一 backbone 创新，且以 FMB 强正识别退化为代价。M1 的小幅总体 AUPR 增益也不是弱分量证据增益；M2/M4 的有界残差在多数 seed 被优化为近零或负值。

## 5. 可证明与不可证明的 SCI 结论

本轮可以证明：在该新的 identity 隔离 development 来源和固定训练读取协议下，四种预注册 backbone 机制均未稳定提高三目标弱分量识别，且 M3 仅呈现标签特定的 VFT/ISRJ 局部证据现象。

本轮不能证明：任何 F2-EPF 统一强化 backbone 优于 B0；跨层金字塔、抗混叠或 top-k 统计本身是弱分量普适机制；模型在独立身份、真实设备或 test-v3 上有效。没有候选通过 Stage-2，故没有资格将同容量/信息破坏控制、冻结 probe 或 confirmation 的缺失解释为正证据。

与完整 VTSE-VBF 创新点的差距在于：缺少直接针对弱可见性的信息监督、同 receiver 的高/低功率配对因果证据、真实或半实物分布验证，以及通过控制和确认链证明的机制特异性。

## 6. 科学停止与下一步

所有 M1–M4 失败工件、history、checkpoint、校准、阈值、预测和 terminal receipt 均保留。禁止在 `backbone-development-v1` 上重命名重试、扩宽、增加 attention、改阈值或组合失败模块。

合法下一步必须是新 Goal：

1. 建立同 receiver identity 的高/低功率配对观测协议，并引入成分级可见性监督或 hard negative；
2. 使用真实/半实物/HIL 数据检验局部高响应是否代表真实弱干扰证据；
3. 收缩论文叙事，将 B0 作为基线，把 M3 保留为标签特定现象和可复核负结果。

`test-v3` 在整个流程中保持 `access_authorized=false`、`analysis_allowed=false`、`used=false`，仍不授权访问。
