# Route-A P1b：I/Q 分组卷积与容量匹配实值卷积对照（2026-08-27）

## 协议

本轮使用 C1 纯六类（FMB、AMA、SawFM、SinFM、MFT、ISRJ），排除 SN、VFT、RangeFakeTarget。所有模型从头训练，避免 F2 预训练初始化造成结构比较偏差。使用 30 dB train-only 的 4,096 个拟合样本、2,048 个验证样本，以及各功率 train-only 未见身份的 512 个开发样本。统一 BCEWithLogitsLoss、AdamW（1e-4/1e-4）、batch=16、2 epochs、split_seed=260827、seeds=2039/2040/2041；未读取普通 test 或 locked test。

## 模型

| 模型 | 结构 | 参数量 | 单样本 MAC |
|---|---|---:|---:|
| F2_full_scratch | 完整 F2_DFMS_Full | 257,511 | 4,449,008,256 |
| IQ_grouped | I/Q 两路 groups=2 卷积，width=41 | 253,222 | 2,789,722,656 |
| Real_conv_matched | 普通实值卷积，width=33 | 255,030 | 3,076,986,528 |

## 结果

| 模型 | 验证 Macro-AUPR | seed std | 验证组合准确率 | 10 dB 开发 AUPR | 30 dB 开发 AUPR | 50 dB 开发 AUPR |
|---|---:|---:|---:|---:|---:|---:|
| F2_full_scratch | 0.999222 | 0.000162 | **0.936361** | 0.858862 | 0.999312 | 0.814064 |
| Real_conv_matched | **0.999329** | 0.000140 | 0.892904 | 0.858485 | 0.999545 | 0.840085 |
| IQ_grouped | 0.995596 | 0.000437 | 0.864258 | 0.852699 | 0.995855 | **0.848077** |

单次 batch-1 CUDA 延迟测量约为 F2 4.986 ms、IQ_grouped 0.558 ms、Real_conv_matched 0.550 ms；该测量仅作成本记录，不能替代完整吞吐基准。

## 结论

1. I/Q 分组卷积把 MAC 降低约 37%，延迟显著降低，但验证 Macro-AUPR 和组合准确率均明显低于 F2；它不能作为当前主识别骨干，除非后续明确引入跨通道融合机制并重新验证。
2. 参数量匹配的普通实值卷积 Macro-AUPR 略高，但组合准确率低于 F2，说明平均排序指标不能代表多标签组合可分性。
3. F2 的复数延迟、多尺度和幅度/相位融合在当前组合级任务上仍然是更可靠的结构。该结论是 3-seed、2-epoch 探索证据，不是最终 5-seed 确认。
4. P1b 不读取 test/locked test，不能据此宣称跨域泛化已经解决。

## 门控

- F2_full：继续作为主骨干。
- Real_conv_matched：保留为容量匹配控制。
- IQ_grouped：作为低成本结构负向结果保留，当前停止晋级。

下一步进入 P2：在 F2 主干固定的条件下，测试功率均衡采样、功率扰动、分层归一化和可见性辅助任务；重点关注 10–20 dB、40–50 dB 的 train-only 功率开发表现。

## 工件

- [P1b 汇总](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p1b_iq_grouped_screen_20260827\p1b_iq_grouped_summary.json)
- [角色清单](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p1b_iq_grouped_screen_20260827\role_manifest.json)
- [终端回执](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_p1b_iq_grouped_screen_20260827\terminal_receipt.json)
- [P1b 脚本](E:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_p1b_iq_grouped_screen.py)
