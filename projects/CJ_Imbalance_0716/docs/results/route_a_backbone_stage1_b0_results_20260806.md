# Route A Backbone Stage 1：B0 正式 development-v1 基线结果

日期：2026-08-06  
证据等级：`five_seed_development_only_b0_baseline`  
结论边界：本报告只描述全新、receiver-identity 隔离的 `backbone-development-v1` 开发源；未打开 test-v2、locked test 或 test-v3，不能作为独立测试结论。

## 1. 固定协议

- 模型：`B0_F2_DFMS_Full`（`dfms_iq`），从确定性随机初始化完整训练。
- 五个 seed：`2039, 2040, 2041, 2042, 2043`。
- 数据 SHA256：`c4122ffae67de855f97a74b42c17ef3df824fad31888dcd6e9dd28b9952c00ac`。
- 每个 seed 的 receiver identity 角色互斥：fit 1920、validation 480、early 2520、calibration 2520、threshold 2520。
- 损失：Focal，`gamma_pos=0, gamma_neg=3`；五个非 SawFM 标签均值加 `0.2 * SawFM`。
- 优化器：AdamW；backbone LR `5e-5`，classifier LR `5e-4`，weight decay `1e-4`，梯度裁剪 `5.0`。
- 最大 36 epoch，patience 8；validation 先比较五个非 SawFM 标签 macro-AUPR，再以 MFT/VFT/ISRJ 弱正对负样本 AUROC 打破平局。
- calibration 角色逐标签 Platt；threshold 角色在 `0.05:0.01:0.95` 网格、负样本 FPR 上限 0.20 下最大化弱正召回；early 角色只报告一次。

## 2. 五 seed 聚合结果

| 指标 | 均值 | 95% bootstrap CI |
|---|---:|---:|
| 六类 macro-AUPR | 0.843746 | [0.839407, 0.848450] |
| 六类 macro-F1 | 0.707800 | [0.701901, 0.714146] |
| Hamming loss | 0.272130 | [0.266931, 0.276944] |
| MFT/VFT/ISRJ 平均弱正 AUROC | 0.555864 | [0.547186, 0.563419] |
| MFT/VFT/ISRJ 平均弱正 recall | 0.237524 | [0.228417, 0.249375] |
| MFT/VFT/ISRJ 平均强正 recall | 0.771434 | [0.761293, 0.785444] |
| MFT/VFT/ISRJ 平均负样本 FPR | 0.195215 | [0.188710, 0.204624] |

## 3. 目标标签诊断

| 标签 | AUPR | 弱正 AUROC | 弱正 recall | 强正 recall | 负样本 FPR |
|---|---:|---:|---:|---:|---:|
| MFT | 0.877899 | 0.578229 | 0.236167 | 0.872100 | 0.202097 |
| VFT | 0.800840 | 0.549968 | 0.238309 | 0.658263 | 0.186774 |
| ISRJ | 0.855571 | 0.539394 | 0.238096 | 0.783939 | 0.196774 |

MFT 的跨 seed 平均 FPR 为 0.2021，略高于 0.20。原因是阈值约束在各 seed 的 threshold 角色上执行，表中是独立 early 角色的泛化结果；这必须作为风险保留，不能解释成阈值门槛已经稳定满足。

## 4. 训练充分性与代价

- 最佳 epoch：`35, 24, 33, 35, 36`；实际运行 epoch：`36, 32, 36, 36, 36`。
- 三个 seed 的最佳点在 35/36，提示仍可能存在训练长度敏感性，但当前没有足够证据宣布统一 36 epoch 协议失效，也不允许只对有利 seed 延长训练。
- 参数量：257,511。
- Conv/Linear MACs：211,857,536 / sample。
- CUDA batch-1 latency：五 seed 均值 4.056 ms（该值是本机工程画像，不是跨设备性能保证）。
- 峰值 CUDA memory：最大 1,051,499,008 bytes。

## 5. 证据与可复现性

- 正式汇总：`outputs/tapc/route_a_backbone_b0_development_v1/b0_summary.json`
- 正式汇总 SHA256：`56c3c5d0fd92e17d7f839a60cd15f4b0a5215c2497c3eff240a87865c72d3d6d`
- terminal receipt：`outputs/tapc/route_a_backbone_b0_development_v1/terminal_receipt.json`，状态 `completed`，总耗时 289.574 s。
- 运行时 B0 runner SHA256：`e67803db0e5597f4c111e238953495d88cf22aeed5a725e715c350f12eee800`。
- M1 实现后模型文件快照：`phase_multiscale.py` SHA256 `a2429a9b2d42b63bbaad23f88923bcc19e1615e3772effca7063f22934c77cc`；`backbone_adapter.py` SHA256 `6493cb9f1e2434819cfb7cdea10410deb9341f2dc258481387fee7399f487038`。B0 默认路径的回归测试确认未启用 M1 固定低通。

## 6. 对 M1 的比较边界

B0 是同源、同角色、同 seed、同损失与训练协议的主要对照。M1 只允许改变三次 stride-2 前的固定复数共享抗混叠低通。必须先完成真实 MAT smoke，再完成相同五 seed 正式运行；任何换核、加宽、改变训练轮数、改变阈值目标或选择规则的版本都不是本次 M1。

当前状态：B0 fresh-confirmed；M1 仅实现与结构测试通过，尚无正式模型效果结论。
