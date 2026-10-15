# Route A presence + visibility 辅助双头 development-v3 预注册

## 1. 单一假设

final-temporal 已含连续相对功率信息，但单一 presence 目标没有迫使共享表示保留该方向。
训练期加入 masked continuous-gap 回归，可在推理不使用物理真值的前提下改善弱分量排序。

## 2. 同容量网络

- 输入：冻结 F2 final-temporal 512 维统计。
- 共享 adapter：`Linear(512,32) + GELU`。
- presence head：`Linear(32,3)`，输出 MFT/VFT/ISRJ logits。
- visibility head：`Linear(32,3)`，只在训练期输出归一化 gap。
- D1 no-aux 与 D2 aux 的结构、参数量、初始化族、优化器、epoch、早停和融合完全相同。
- D1 的 visibility loss 权重为 0；D2 固定为 0.25，不搜索。
- 推理只使用 presence head；visibility head 和真实功率均不参与推理。

## 3. 损失与训练

presence 使用逐标签 `pos_weight` BCE。visibility 只在 `label=1` 且 gap 有限处使用
Smooth-L1，目标为 `clip(gap,-40,0)/40`。总损失为 `L_presence + 0.25 L_gap`。
F2、RawSaw、校准、FPR<=0.20 阈值、六标量 generic bounded fusion 和五数据角色冻结。

## 4. 对照与门槛

- C0：F2+C6。
- C4：线性 all-positive 同容量历史主对照。
- D1：低秩双头结构但无辅助损失，严格容量对照。
- D2：低秩双头 + 连续 gap 辅助损失。

D2 晋级需同时满足：相对 D1 弱召回 `>=+0.010`、弱正-vs-负 AUROC `>=+0.010`、
macro AUPR 不劣于 `-0.003`、强召回不劣于 `-0.010`；目标 FPR `<=0.20`；相对 C0
弱召回 `>=+0.030`、AUROC `>=+0.015`，且至少 4/5 种子弱召回增益为正。test-v3 禁止访问。

