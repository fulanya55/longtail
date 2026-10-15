# Route A Frequency Representation V3 预注册（Stage 0）

日期：2026-08-25  
项目：`E:\LJQ_code\CJ_Imbalance_0716`  
环境：`D:\anaconda3\envs\PY10\python.exe`  
数据：`E:\LJQ_code\干扰数据\dual-axis-frequency-representation-v3`

## 研究问题

在完全新的 receiver identity 上，检验“冻结 B0 的 stage1–3、仅训练 stage4 rank-4 residual adapter + 六个标签条件证据投影 + feature-space hard-negative ranking”能否稳定改善 Label-LT 的弱标签识别。该试验验证的是频率/表征轴主效应，不把可见性、RawSaw 或旧测试结果混入主结论。

## 不可变边界

- 不读取旧 R2、locked、test-v2、test-v3 的样本级预测；只读取其 aggregate、receipt、hash 和封存元数据。
- 新 receiver identity 范围为 `3940000001..3940025200`，与历史生成范围隔离；VH/VL 共享 receiver_noise_seed、component_seeds、labels、参数和背景。
- 先完成本文件、Stage-0 审计、seed-overlap 审计和 resolved-config，再执行正式数据生成。
- 阈值、margin、rank、anchor、adapter rank 及 early key 在正式 evaluation source 上不得重调。
- 训练只用 VH（SawFM 30 dB）；VL（SawFM 20 dB）用于成对可见性诊断，不作为频率主效应训练输入。
- test-v3 仅读取 seal metadata，不打开数据或预测。

## 数据协议

63 个非空六标签组合，每组合 400 个新 identity，共 25,200；VH 为所有活动分量 30 dB，VL 仅 SawFM 为 20 dB、其余活动分量 30 dB；I/Q 长度 3000；只生成 train split。按每组合 240/40/40/40/40 切分 optimization/early/calibration/threshold/evaluation，五个角色两两不重叠。由 optimization 派生 FB、LabelIF10/20/50、ComboIF10/20/50 清单；IF20 为主要 Label-LT 条件。

## 固定模型与训练

B0 为 `dfms_iq`，输入 `[B,2,3000]`，width=16，feature=128，dropout=0.2，线性六标签头；主干阶段 1–3 冻结。候选仅训练 stage4 rank-4 adapter、六个 label-conditioned evidence projection 及受限残差头，并保留 128 维 mean/std 安全旁路。禁止输入 frequency、weak mask、power、pair id 或 oracle。

候选损失固定为：

`L = L_label_balanced + 0.20 L_feature_rank + 5e-4 L_anchor + 1e-4 L_adapter_norm`。

五个新训练种子：`261021..261025`；core backbone lr=5e-5，classifier lr=5e-4，adapter lr=1e-5，evidence head lr=3e-4，weight decay=1e-4，batch=64，cached-head batch=256，core 36 epochs，adapter 60，head 100，clip=5，2400 draws/epoch。

## 证据与硬门

主比较为候选相对每个 seed 的 best(G0 B0-Focal, G1 B0-DB, G2 plain-BCE cRT, G3 RBC-cRT, G4 rank-4 adapter)。LabelIF20 macro-AUPR 平均增益≥0.010，三尾标签平均增益≥0.020，ComboIF20≥0.003；5,000 次 receiver-paired bootstrap CI 下界>0；至少 4/5 seed 为正；FB 下降≤0.005、任一标签平均下降≤0.010；IF10/IF50 各至少 3/5 seed 非负；击败 random-neighbor、no-adapter、no-label-projection、capacity-matched MLP 控制；参数增量≤5%，batch=1 延迟增量≤15%，batch=64 吞吐下降≤15%。任一失败即 `frequency_representation_failed`，保存完整结果并停止 RawSaw、2×2 和 confirmation，不得同源调参。

## 产物

必须保存 prereg、resolved-config、源文件/清单 SHA256、compile/tests/GPU smoke、训练历史、checkpoint、逐标签预测与指标、paired bootstrap、控制组、成本测量、evidence ledger、中文报告和 Terminal receipt；所有 receipt 固定 `locked_test_used=false, test_v2_used=false, test_v3_used=false`。
