# 路线 A/B 下一阶段确认与机制筛选方案

日期：2026-07-23  
状态：已执行（路线 A：80 runs；路线 B：21 downstream runs/18 pretrained runs）  
环境：`D:\anaconda3\envs\PY10\python.exe`  
数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`  
边界：development roles only，locked test 禁止读取

## 1. 路线 A：失衡强度外部确认

原计划使用 IF=2/5。可行性审计显示：

- optimization 上限 1024：IF=2 为 12/12 可行，IF=5 为 0/12；
- optimization 上限 2048：IF=2 为 12/12 可行，IF=5 为 0/12；
- 完整 optimization role：IF=4 为 12/12 可行，IF=5 仍为 0/12。

因此 IF=5 在 20% 标签计数容差下受四标签共现结构限制，不可实现。本阶段冻结为 IF=2/4，并统一使用完整 optimization role，禁止只为强失衡条件放宽容差。

正式矩阵：IF=2/4，四尾类轮换，ASL/DB，seeds 2031/2032/2033，20 epochs，共 48 runs。温度只用 calibration role；阈值只用 threshold-selection role。确认主阈值固定为 `per_label_f1`，`shrinkage_per_label_f1` 作为保守工作点。

## 2. 路线 B：功率自监督 V2

预训练只使用 5% 标签方案中的 unlabeled optimization identities。预训练功率为 25/30/35 dB；20/40 dB 只作为 held-out-power development 诊断。

变体：RandomInit、SameViewAug、NaivePower、MultiPositive、HybridMasked、HybridPowerAux、RandomSource。为保持原计划 18 个主要训练单元，RandomInit 只执行下游训练，不计自监督预训练单元；其余 6 个预训练变体 × 3 seeds=18。

关键机制：

1. 同 identity 的跨视图样本全部作为正样本；
2. 同 identity 样本不再互为负样本；
3. Hybrid 以 SameViewAug 为主目标，跨功率多正样本权重 0.2；
4. HybridPowerAux 额外预测有符号相对功率差，权重 0.1；
5. seeds 2031/2032/2033，5% labeled，20 pretrain epochs，30 probe epochs，20 finetune epochs。

Go/no-go：相对 SameViewAug，候选至少提高 linear-probe macro-AUPR 0.01、5% 微调 macro-AUPR 0.005、held-out-power macro-AUPR 0.01，且 3/3 seeds 同方向。否则路线 B 停止作为主创新。

结果报告：`docs/results/two_route_confirmatory_results_20260723.md`
