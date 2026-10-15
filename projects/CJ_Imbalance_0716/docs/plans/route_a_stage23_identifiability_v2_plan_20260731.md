# Route A：stage2/stage3 弱证据可辨识性筛选（development-v2）

## 背景与目的

有界 final-temporal 局部读出器已经证明：即使残差达到平均 0.087、最大 0.335，
MFT/VFT/ISRJ 的弱正—负 AUROC 仍只提高 0.00015。因此本轮不再接回分类器，
而是回答更前置的问题：stage2 或 stage3 是否保留了 final temporal map 中已经
丢失的弱证据？

## 数据协议

- 仅使用 development-v2；test-v2 和历史 locked test 保持封存。
- 五种子 2039--2043；每种子复用对应冻结 F2 checkpoint。
- 使用完整 optimization role，而不是 2400 条 label-LT 子集，以解决旧 oracle
  probe 中弱正样本过少和 validation--early 落差过大的问题。
- optimization 内固定 80/20 fit/validation；early role 为独立 development 诊断。
- 每标签训练只使用“弱正样本”和“标签缺失负样本”，强正样本不参与边界拟合。

## 表征

- G：冻结 F2 的 128 维全局特征。
- T：final temporal map 的 mean/std/max/top-8 mean，共 512 维。
- S2：stage2 复数特征的幅度、幅度差分、相位增量 cos/sin 及其差分；每类描述符
  使用 mean/std/max/top-8 mean，共 768 维。
- S3：同样的物理描述符作用于 stage3，共 1152 维。
- S23：S2 与 S3 拼接，共 1920 维。

## Probe 与训练

- 每种表征训练三输出线性 probe，对应 MFT/VFT/ISRJ。
- masked pos-weight BCE；每标签的 `pos_weight=N_negative/N_weak_positive`。
- AdamW，lr=1e-3，weight decay=1e-4，最多 50 epoch，patience=10。
- 选择指标：validation 三标签 weak-vs-negative macro-AUROC。
- 独立 early role 报告 AUROC、AUPR、逐标签结果及 validation--early gap。

## 晋级门

最佳早期表征相对冻结 C6 必须满足：

1. MFT/VFT/ISRJ 平均 weak-vs-negative AUROC 提升至少 0.020；
2. 三标签中至少两个平均 AUROC 为正增益；
3. 至少 4/5 个种子平均 AUROC 为正增益；
4. validation--early AUROC 落差不超过 0.15；
5. early macro-AUROC 至少 0.63。

通过后，最佳层进入“低秩多尺度独立专家 + 受限残差”结构实验；未通过则表明
现有 F2 encoder 各层对 MFT/VFT/ISRJ 弱证据均不充分，需要改变主干训练目标，
例如同身份高低功率配对表征学习，而不是继续增加读出器。

