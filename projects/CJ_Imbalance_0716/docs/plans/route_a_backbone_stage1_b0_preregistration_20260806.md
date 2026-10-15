# Route A Backbone 强化 Stage 1：B0 新来源复现预注册

日期：2026-08-06  
候选：`B0_F2_DFMS_Full`  
证据角色：`development-only baseline`，不授权 test-v3

## 固定数据与角色

- 数据：`E:\LJQ_code\干扰数据\route_a_backbone-development-v1_20260806\closedset_6class_train.mat`
- SHA-256：`c4122ffae67de855f97a74b42c17ef3df824fad31888dcd6e9dd28b9952c00ac`
- receiver identity：`2748677743–2748702942`，25200 个且唯一
- 五个 seed：`2039–2043`
- 每 seed：fit 1920、validation 480、early/calibration/threshold 各 2520
- 角色索引已经在 Stage 0 固化；本阶段不得重拆分、重采样或更换长尾计数
- test-v3 仅核对 seal manifest 的三个布尔标志，不读取其数据或结果

## B0 结构与训练

- `dfms_iq`, width 16, dropout 0.2，参数量固定 257511
- 从每个 seed 的确定性随机初始化开始训练全部 backbone 与 classifier
- 同一 seed 的后续 B1–B4 必须复用相同模型初始化 seed 和 batch order seed
- 输入 `[B,2,3000]`，`complex_rms`
- loss：Focal，`gamma_pos=0, gamma_neg=3`
- 总损失：`mean(FMB,SinFM,MFT,VFT,ISRJ) + 0.2*SawFM`
- AdamW；backbone LR `5e-5`，classifier LR `5e-4`
- weight decay `1e-4`，gradient clip `5.0`
- batch 64，最多 36 epochs，patience 8
- checkpoint 主键：validation 五个非 SawFM 标签 macro-AUPR；并列时使用
  MFT/VFT/ISRJ 平均 weak-vs-negative AUROC

## 校准、阈值与报告

- calibration：每标签独立 Platt，固定 `LogisticRegression(C=1e6, lbfgs)`
- threshold：固定网格 0.05–0.95、步长 0.01；在每标签负样本
  FPR≤0.20 下最大化弱正召回，再以 F1 和距 0.5 距离打破并列
- early：唯一 development 报告角色
- 保存 `history.csv`、`best_model.pth`、`calibration.json`、
  `thresholds.json`、`metrics.json`、`predictions.npz`、`run_receipt.json`
- 工程收据报告参数量、实际 Conv/Linear MAC 调用计数、batch-1 CUDA 延迟、
  峰值显存、代码/数据/角色/checkpoint SHA-256

本阶段只建立 B0 的新来源基准分布，不使用 B0 自身结果改变 M1–M4 的结构、
训练预算或晋级阈值。
