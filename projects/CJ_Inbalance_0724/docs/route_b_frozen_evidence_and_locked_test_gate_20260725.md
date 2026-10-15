# 路线 B 冻结证据与 locked-test 决策（2026-07-25）

## 结论

开发阶段已经完成 paired bootstrap、预测标签门控参数评价、计算代价、三个物理专家筛选以及 identity-safe 九档功率验证。最终可冻结的候选是：MLCNN 分类骨干 + IQ-Grouped 参数骨干，并仅用周期序列专家替换 SawFM `sweep_period` 输出。

当前不申请 locked test。原因不是主 confirm 集失败，而是预先要求的功率及弱分量泛化尚未形成可接受的闭环：任务路由的参数 oracle NMAE 在 10--50 dB 从 0.748 漂移到 2.040；10 dB 弱分量召回只有 0.165。一次性测试预算应保留到功率鲁棒性方法在独立 development 身份上通过预设门槛之后。

## 1. Paired bootstrap 与端到端门控

任务路由相对 IQ-Grouped：分类 AUPR 增加 0.01849，95% CI [0.01611, 0.02082]；预测门控参数 NMAE 降低 0.04974，95% CI [-0.05430, -0.04538]；漏检惩罚 NMAE 降低 0.05971，95% CI [-0.06431, -0.05531]。

任务路由相对 MLCNN-global：分类 AUPR 小幅降低 0.000973，95% CI [-0.001391, -0.000585]；预测门控参数 NMAE 降低 0.05747，95% CI [-0.06363, -0.05157]；漏检惩罚 NMAE 降低 0.05541，95% CI [-0.06114, -0.04976]。任务路由相对该基线的假阳性门控率高 0.00896，95% CI [0.00600, 0.01205]，需要如实作为代价报告。

## 2. 双骨干计算代价

RTX 5080、PyTorch 2.11.0+cu128、FP32、无 AMP。任务路由为 12.891M 参数、459.43M MACs（约 918.87 MFLOPs）。相对 MLCNN-global，参数量增加 0.66%，MACs 增加 11.12%；batch=1 延迟由 1.002 ms 增至 1.315 ms，batch=64 吞吐从 22,162 降至 15,467 samples/s，峰值显存从 163.13 MB 增至 174.24 MB。双骨干的主要代价是约 31% 的单样本延迟和约 30% 的批吞吐下降，而不是参数量膨胀。

## 3. 物理专家晋级结果

固定门槛：相对训练中位数常数基线 MAE 至少降低 10%，Pearson r 至少 0.3，且两个指标 3/3 seeds 同方向。

| 专家 | MAE 相对改善 | Pearson r | 同方向 | 接回 |
|---|---:|---:|---:|---|
| SawFM 周期序列 | 19.05% | 0.502 | 3/3 | 是 |
| MFT 延迟峰 ordinal | 1.79% | 0.179 | 3/3 | 否 |
| VFT 瞬时频率序列 | -1.36% | -0.015 | 未满足 | 否 |

SawFM 专家相对任务路由原参数头的 SawFM-period NMAE 平均降低 18.67%，相关系数从 0.101 提高至 0.502；替换后总体 oracle 参数 NMAE 降低 0.01371，预测门控 NMAE 降低 0.01559，漏检惩罚 NMAE 降低 0.01408，3/3 seeds 同方向且分类输出完全不变。

## 4. Identity-safe 功率与弱分量泛化

九档视图按 `receiver_noise_seed` 一一配对，只评估 root confirm 的 1,260 个身份，未使用任何 power test 文件。

| 功率 dB | 分类 AUPR | 弱分量召回 | 基础 oracle NMAE | 接入 Saw 后 oracle NMAE | Saw period NMAE（前/后） |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.904 | 0.165 | 0.748 | 0.732 | 0.873 / 0.672 |
| 15 | 0.946 | 0.564 | 0.627 | 0.611 | 0.874 / 0.678 |
| 20 | 0.968 | 0.862 | 0.709 | 0.694 | 0.875 / 0.694 |
| 25 | 0.981 | 0.950 | 0.772 | 0.761 | 0.879 / 0.746 |
| 30 | 0.989 | -- | 0.904 | 0.896 | 0.884 / 0.785 |
| 35 | 0.992 | 0.986 | 1.171 | 1.164 | 0.889 / 0.803 |
| 40 | 0.992 | 0.974 | 1.493 | 1.487 | 0.892 / 0.823 |
| 45 | 0.987 | 0.923 | 1.784 | 1.779 | 0.893 / 0.839 |
| 50 | 0.972 | 0.688 | 2.040 | 2.037 | 0.892 / 0.847 |

这说明周期专家具有跨功率正收益，但只修复一个参数，不能抵消其余参数的整体分布漂移；低功率弱扫频分量的识别门控仍是端到端系统的主要失败机制。

## 5. Locked-test 决策规则

在一次性申请之前，需要在不修改上述冻结候选与统计规则的前提下，另建 identity-safe development 方案解决功率漂移，并满足以下门槛：

1. 九档平均预测门控参数 NMAE 相对当前冻结候选至少降低 10%，且 3/3 seeds 同方向。
2. 两个极端档 10 dB 与 50 dB 的弱分量召回均不低于 0.80。
3. 10 dB 与 50 dB 的漏检惩罚参数 NMAE 均不高于 1.0。
4. 30 dB 主域分类 AUPR 相对冻结候选下降不超过 0.005。
5. 候选、阈值、归一化、权重、统计脚本和结果字段全部重新哈希冻结后，才由用户显式批准一次 locked-test 运行；测试后禁止调参或筛选 seed。

## 6. 产物

- `outputs/paper_core_evidence/paper_core_evidence.json`
- `outputs/paper_core_evidence/efficiency.json`
- `outputs/physics_parameter_specialists/summary.json`
- `outputs/physics_parameter_specialists/saw_attachment_results.json`
- `outputs/identity_safe_power_parameter_eval/identity_safe_power_results.json`
- `configs/paper_candidate_freeze_20260725.json`

上述所有结果均标记 `locked_test_used=false`。
