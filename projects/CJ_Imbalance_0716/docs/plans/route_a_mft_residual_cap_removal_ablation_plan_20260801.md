# Route A MFT 残差上界移除消融预注册

## 问题

当前 MFT guard 使用 `0.75 * sqrt(4p(1-p)) * min(tanh(e),0)`。需要验证困难负样本 FPR 安全性是否依赖额外的 `0.75` 上界。

## 固定对照

- A0：既有 cap=`0.75`；
- A1：移除额外硬上界，即 cap=`1.0`，保留 `tanh` 的自然有界性和非正残差不变量；
- 峰计数专家、B2 logit、角色、校准器、阈值、五个 seeds 均固定；
- 不重新拟合、不重新选阈值；报告 MFT hard-FPR、order-5 FPR、弱 AUROC、弱召回和非 MFT 保持性。

## 边界

这是 development-v3 的冻结机制消融。结果只用于判断上界是否必要，不改变已冻结 candidate-v2，不访问 test-v3。
