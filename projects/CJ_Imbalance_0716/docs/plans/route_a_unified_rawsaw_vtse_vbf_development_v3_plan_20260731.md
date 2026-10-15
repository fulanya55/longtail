# Route A 六标签统一 RawSaw + VTSE-VBF development-v3 预注册

日期：2026-07-31  
证据角色：阶段 C，development-v3 五种子；test-v3 已封存且禁止访问。

## 1. 单一机制假设

RawSaw 只替换 SawFM 的证据，VTSE-VBF 只向 MFT/VFT/ISRJ 写入有界残差，两个
专家在标签维度上解耦。因此统一模型应同时保留 C1 的 SawFM 收益和 C2 的三目标弱
分量收益，不改变 FMB/SinFM 安全旁路，也不产生 SawFM-present/absent 负迁移。

## 2. 固定模型组

- C0 `F2+C6`：development-v3 已训练的六标签 C6_Joint；
- C1 `RawSaw`：C0，仅用 RawStage1SawFMFrontend 替换 SawFM logit；
- C2 `VTSE-VBF`：C0，仅修正 MFT/VFT/ISRJ；
- C3 `Unified`：C1 与 C2 的标签解耦组合；
- C4 `CapacityControl`：保留 C1；三目标使用与 VTSE 和融合器同参数量的普通
  all-positive temporal expert 与普通 BCE 融合，不使用 weak mask、三层等权或
  不确定性门。

RawSaw 参数：raw-only，6 个物理描述量，16 通道，多尺度核 7/31/63，top-k=16，
32 维分类器。VTSE 参数：final temporal `[B,128,T]` 的 mean/std/max/top-8 mean，
512→3 线性头。VBF 参数：3 个非负专家权重和 3 个 bias，平方根不确定性门，残差
硬上限 ±0.50。C4 的 temporal head 和融合器参数量与 C3 对应部分完全相同。

## 3. 数据角色

直接复用每个种子已冻结的 development-v3 `fit/validation/early/calibration/threshold`
索引及其 SHA。fit 训练专家和融合器；validation 早停；calibration 对每标签拟合同一
协议的 Platt 校准；threshold 在负样本 FPR≤0.20 下最大化弱正召回；early 只用于本轮
最终 development 报告。所有角色 receiver identity 不重叠。

## 4. 训练参数

- RawSaw：AdamW，LR=1e-3，weight decay=1e-4，batch=64，30 epochs，patience=5；
- VTSE/C4 temporal head：AdamW，LR=1e-3，weight decay=1e-4，batch=128，
  60 epochs，patience=8；
- VBF/C4 fusion：AdamW，LR=1e-2，weight decay=1e-4，200 epochs，patience=20；
- 梯度裁剪=5；弱分量阈值=10 dB；top-k=8；
- VTSE 训练 mask：仅弱正与标签缺失负样本；C4 使用全部正负样本；
- VBF 损失：弱正、强正、负样本三层等权 + residual L2 0.02 + bias L2 0.05；
- C4 fusion：普通 BCE + 相同正则，不使用可见性分层。

## 5. 主要门槛

C3 相对 C0 必须同时满足：macro-AUPR +0.005、macro-F1 +0.010、三目标平均弱召回
+0.030、weak-vs-negative AUROC +0.015、三目标平均 FPR≤0.20、强正召回下降不超过
0.010、至少 4/5 种子弱召回增益为正、各目标 AUPR 下降不超过 0.010。

保留性门槛：C3 的 SawFM AUPR/弱召回不低于 C1；三目标弱召回/AUROC 不低于 C2；
SawFM-present 与 SawFM-absent macro-F1 相对 C0 均不明显退化。机制门槛：C3 应优于
同参数量 C4，证明收益不能仅由参数增加解释。

若失败，不降低门槛、不访问 test-v3；按 SawFM 保留失败、VTSE 保留失败、容量对照
不分离或阈值/FPR 失败四类根因停止或整改。

## 6. 保存产物

每种子保存角色索引哈希、RawSaw/VTSE/C4/fusion checkpoint、history、校准器、阈值、
early predictions、逐标签和分层指标、参数量及 receipt；顶层保存五种子均值、bootstrap
95% CI、配对差值和晋级判断。所有 receipt 固定写入 `locked_test_used=false`、
`test_v2_used=false`、`test_v3_used=false`。
