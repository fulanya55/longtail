# Route A 九类特征充分性门禁（2026-08-26）

## 当前结论

本轮结果为 **provisional_insufficient / 六类筛选停止**，不是 locked-test 结果，也不是九类最终性能结论。原因是：

1. 已找到并加载每个 seed 对应的可追溯 F2 checkpoint：`frozen/20260731_route_a_visibility_balanced_fusion_v2/seed_{2039,2040,2041}/backbone/best_model.pth`；A0 与 A1 使用同一 seed 的同一 F2 初始化。
2. 三个 development seed 的 A0→A1 30 dB Macro-AUPR 增益约 **+0.064**，仍显著超过预注册 0.02 容差；加入冻结 F2 + 两层 MLP 头的 A0b 后，A1−A0b 仍约 **+0.045**，因此不能解释为单纯线性头容量问题。
3. A2 EfficientNet-B1-1D 在本轮两 epoch、小样本统一预算下没有超过 A1；这属于 development-only 参考结果，不能单独证明 EfficientNet-B1 不适合九类。

因此禁止读取 locked test，禁止运行 84 个六类子集筛选，也不允许用删类掩盖骨干欠表征。

## 数据与角色

- 数据根：`E:\论文相关\干扰数据\复合干扰数据0716\九类`
- 九个已识别标签：`FMB, AMA, SawFM, SinFM, SN, MFT, VFT, ISRJ, RangeFakeTarget`
- 9 个功率：10–50 dB，步长 5 dB；每个功率均有独立 train/test MAT。
- 每个 MAT 的 `iq_complex` 为 `[3000, 186000]`，标签为 `[9, 186000]`；`receiver_noise_seed` 用于身份审计。
- 30 dB train 的 1024 个样本用于开发训练；每个功率 test 的 256 个不同样本用于只读 development 曲线。train/test MAT 未混用；同一功率视图未被当成独立身份重复计数。
- 角色与身份清单：[`role_manifest.json`](../../outputs/route_a_nineclass_feature_sufficiency_20260826/model_screen_f2_init_dev_audit/role_manifest.json)、[`split_manifest.npz`](../../outputs/route_a_nineclass_feature_sufficiency_20260826/model_screen_f2_init_dev_audit/split_manifest.npz)。fit 与 development 的 `receiver_noise_seed` 交集为 0；locked-test 未使用。
- 原始 `power_sweep_manifest.csv` 中的 G 盘路径只作为历史记录；运行入口只接受 E 盘路径。
- 18 文件的元数据审计产物：
  - [`nineclass_data_audit.json`](../../outputs/route_a_nineclass_feature_sufficiency_20260826/data_audit/nineclass_data_audit.json)
  - [`nineclass_e_drive_manifest.csv`](../../outputs/route_a_nineclass_feature_sufficiency_20260826/data_audit/nineclass_e_drive_manifest.csv)
  - 完整 SHA256 审计正在写入 `outputs/route_a_nineclass_feature_sufficiency_20260826/data_audit_hashed/`。

## 模型与统一训练设置

| 单元 | 骨干/头 | 可训练参数 | 作用 |
|---|---|---:|---|
| A0 | F2 `DFMS_Full`，冻结 backbone；9 类 Dropout+Linear 浅层头 | 1,161 | 使用逐 seed 可追溯 F2 checkpoint 的冻结特征对照 |
| A0b | F2 `DFMS_Full`，冻结 backbone；128 维两层 MLP 九类头 | 17,673 | 区分浅层头容量与骨干表征 |
| A1 | F2 `DFMS_Full` + 9 类原生分类头，端到端微调 | 257,898 | 当前路线 A 九类主模型 |
| A2 | `EfficientNet-B1 1D` + 9 类 Linear head | 6,324,937 | 强参考骨干 |
| A3 | F2 `DFMS_CSFP` + 9 类头 | 310,763 | 跨阶段复用改进预筛 |
| A4 | F2 `DFMS_LocalStatsK16` + 9 类头 | 291,051 | 局部统计旁路改进预筛 |

F2 输入接口为 `[B, 2, 3000]` 真复数 I/Q（I、Q 两通道）；包含 delayed complex multi-scale encoder、延迟笛卡尔融合、幅相特征、全局均值/标准差统计池化和 128 维共享特征（width=16）。A2 直接使用已有 `E:\LJQ_code\CJ_Inbalance_0724\cj0724\models.py` 的 1D EfficientNet-B1。

共同设置：GPU `cuda:0`（RTX 5080）、BCEWithLogitsLoss、AdamW、weight decay `1e-4`、梯度裁剪 5.0、输入 complex-RMS 归一化、固定阈值 0.5；本轮 3 seeds 为 2039/2040/2041，2 epochs，batch 16。该预算用于门禁筛查，不构成正式最终模型比较。

## 观察到的开发结果

来自 [`feature_sufficiency_summary.json`](../../outputs/route_a_nineclass_feature_sufficiency_20260826/model_screen_f2_init_dev_audit_v2/feature_sufficiency_summary.json)：

| 单元 | 30 dB Macro-AUPR 均值 | 最差功率 Macro-AUPR 均值 | 30 dB 最弱类 AUPR 均值 |
|---|---:|---:|---:|
| A0 | 0.739 | 0.653 | 0.547 |
| A0b | 0.757 | 0.655 | 0.539 |
| A1 | 0.803 | 0.664 | 0.602 |
| A2 | 0.487 | 0.452 | 0.433 |
| A3（低预算预筛） | 0.698 | 0.581 | 0.506 |
| A4（低预算预筛） | 0.696 | 0.583 | 0.511 |

A1−A0 = **+0.064**；A1−A0b = **+0.045**；A2−A1 = **−0.316**。A2 的负增益发生在很短预算下，属于 partial/development-only 证据，不能据此宣称 EfficientNet-B1 不适合九类。

A1−A0 在 2039/2040/2041 三个 seed 均为正（约 +0.057～+0.071）；A1−A0b 也在三个 seed 均为正（约 +0.035～+0.053），说明冻结 F2 的九类适配不足具有稳定方向，而不是单次随机波动。

组合级结果也已写入每个 seed/功率单元：30 dB A1 的 256 个评估样本中，真实组合约 156 种、预测组合约 75 种、组合精确率约 0.082；10 dB 和 50 dB 分别约 0.027 和 0.016，表明组合泛化仍明显不足。

此外，对 seed 2039 的每个功率 test MAT 按真实标签逐组合抽取一条样本，完整覆盖 465 个组合，生成了只读全覆盖图：[combination_error_map.json](../../outputs/route_a_nineclass_feature_sufficiency_20260826/combination_audit_v2_seed2039/combination_error_map.json)。30 dB 组合精确率：A0 0.0366、A0b 0.0452、A1 0.0710、A2 0.0022、A3 0.0538、A4 0.0731；该结果明确标记为 development-only，未使用 locked test。

F2 A1 的成本审计（batch=1，GPU）约为 4.45×10^9 Conv/Linear MACs、4.3 ms；参数量 257,898。每个 checkpoint 的 SHA256 和成本字段均保存在 seed JSON 中。

改进预筛产物位于 [`backbone_improvement_screen`](../../outputs/route_a_nineclass_feature_sufficiency_20260826/backbone_improvement_screen/)。在 512 个训练样本、1 epoch 的低预算下，A3/A4 的 30 dB Macro-AUPR 分别约 0.698/0.696，和同预算 A1 约 0.695 接近；没有形成稳定、可晋级的增益证据，必须在正式等预算下复核。

## 下一步唯一推荐动作

1. 先进行 F2 表征改进，而不是删类：优先测试跨阶段复用/局部统计旁路（`dfms_iq_csfp`、`dfms_iq_local_topk16`）并从现有 F2 checkpoint 继承共享权重。
2. 保持相同 identity split、训练预算和评价脚本，扩展到 10–50 dB 全曲线和完整 465 组合错误图谱；A2 需要增加有效训练步数后再做公平比较。
3. 只有当 A0/A0b→A1、A1→A2、最差功率、最弱类别和 3-seed 稳定性同时通过预注册门槛，才允许继续 84 个六类候选 screen。

所有当前结果保留为 development/partial 证据；未访问任何 locked test。
