# Route-A C1 纯六类 F2 基线（2026-08-26）

## 1. 实验目的

在独立分支 `codex/route-a-ama-sixclass-baseline-20260826` 上，将九类数据中的 `FMB + AMA + SawFM + SinFM + MFT + ISRJ` 作为一个闭集，先建立可复核的 F2 基线。为避免标签投影污染，含有 SN、VFT 或 RangeFakeTarget 的样本全部排除；只保留至少含一个目标类且不含排除类的样本。

## 2. 数据角色

- 数据根目录：`E:\论文相关\干扰数据\复合干扰数据0716\九类\power_sweep_9level_v1`。
- 每个功率的九类 train/test 各有 186,000 个样本；纯六类筛选后各有 25,200 个样本。
- 拟合：30 dB train 的纯六类样本，按 `split_seed=260826` 划分 80%（20,160）拟合、20%（5,040）验证。
- 主测试：30 dB test 的全部 25,200 个纯六类样本。
- 跨功率诊断：10、15、20、25、35、40、45、50 dB test 各随机抽取 2,048 个纯六类样本；这些诊断结果不用于调参。
- 接收机噪声身份：拟合与全部测试功率的交集为 0；没有读取 locked test。

## 3. 网络和训练

- Backbone：`F2_DFMS_Full` / `dfms_iq`，width=16，输入 `[B,2,3000]`。
- 结构：复数 I/Q 延迟融合、四级多尺度复数编码、Cartesian/幅度/lag=1/4/16 相位晚期融合、全局 mean/std 池化和六类线性分类头。
- 参数量：257,511；单样本卷积/线性 MAC：4,449,008,256；batch-1 CUDA 延迟一次测量约 4.21 ms。
- 初始化：使用已有 Route-A F2 checkpoint（seed 2039）加载共享 backbone，六类分类头在当前训练中更新。
- 损失：`BCEWithLogitsLoss`。
- 优化器：AdamW，`lr=1e-4`，`weight_decay=1e-4`，梯度裁剪 5.0。
- batch=32，8 epoch，固定阈值 0.5，最佳验证 Macro-AUPR 选 checkpoint。

## 4. 结果

最佳 epoch 为 8。验证集 Macro-AUPR=0.9999997，Macro-F1=0.999936，六类组合完全匹配率=0.999603。

| 测试功率 | 样本数 | Macro-AUPR | Macro-F1 | 六类组合准确率 | Hamming loss |
|---:|---:|---:|---:|---:|---:|
| 10 dB | 2,048 | 0.849845 | 0.701463 | 0.279297 | 0.157715 |
| 15 dB | 2,048 | 0.866671 | 0.705153 | 0.288086 | 0.158366 |
| 20 dB | 2,048 | 0.932366 | 0.703381 | 0.274414 | 0.160400 |
| 25 dB | 2,048 | 0.997366 | 0.846981 | 0.462402 | 0.101969 |
| **30 dB 主测试** | **25,200** | **0.999999** | **0.999883** | **0.999325** | **0.000119** |
| 35 dB | 2,048 | 0.998258 | 0.950942 | 0.653320 | 0.057861 |
| 40 dB | 2,048 | 0.989188 | 0.698633 | 0.296875 | 0.228353 |
| 45 dB | 2,048 | 0.951871 | 0.604059 | 0.295410 | 0.255615 |
| 50 dB | 2,048 | 0.875040 | 0.596849 | 0.281738 | 0.254964 |

30 dB 主测试逐类表现：AMA/FMB/SawFM/SinFM 的 AUPR 均约为 1.0；ISRJ AUPR=0.999991、召回=0.999141；MFT AUPR=1.0、召回=1.0。由此可见，30 dB 闭集基线已经基本饱和，后续创新不能只报告该工作点。

## 5. 结论与边界

该基线证明了：在纯六类闭集、30 dB 同分布测试上，F2_DFMS_Full 能稳定完成 C1 组合识别；数据过滤和 I/Q 训练接口正常。跨功率诊断显示 10–20 dB 与 40–50 dB 的组合准确率明显下降，说明主要后续问题是功率迁移/可见性，而不是 30 dB 内的特征提取容量不足。

当前结果是单 seed 的基线，不是五种子确认，也不是 locked-test 结果。下一步应在相同纯六类角色上做 5-seed 基线确认，并以 C0 原六类作为连续性对照，再进行功率鲁棒或标签专项改进；不得用本轮跨功率诊断集反复调阈值。

## 6. 工件

- [基线结果](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_c1_baseline_full_20260826\baseline_result.json)
- [实验配置](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_c1_baseline_full_20260826\experiment_config.json)
- [数据角色清单](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_c1_baseline_full_20260826\role_manifest.json)
- [终端回执](E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_c1_baseline_full_20260826\terminal_receipt.json)
- [训练脚本](E:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_c1_baseline.py)

模型状态 SHA256：`dd91f0f36a6423df9474cd892cd356609ad20c65581a390e8c735080fffa740a`。
