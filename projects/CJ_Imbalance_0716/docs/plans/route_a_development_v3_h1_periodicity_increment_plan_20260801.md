# Route A development-v3 H1强基线周期增量预注册

## 目的

冻结当前最强 `H1_Linear_BCE`，验证310维对齐周期专家是否在强反向解耦分类器之上仍提供不可由重标定或打乱时序解释的增量。

## 方案

- B0：冻结 H1；
- B1：仅用 H1 ISRJ logit训练相同弱正加权 Logistic分数头；
- B2：使用 `[h, z, uncertainty*z, tanh(h)*z]` 的对齐周期融合；
- B3：同结构 block-shuffle周期融合。

`h`为冻结 H1 ISRJ logit，`z`为冻结周期专家标准化分数。Logistic固定 `C=0.1`，弱 ISRJ正样本权重4，不搜索。只替换 ISRJ logit，其余标签逐元素保持 H1。

## 晋级门

B2须同时满足：相对 B0弱召回 `>=+0.010`、弱 AUROC `>=+0.005`、AUPR非负；相对 B1弱 AUROC `>=+0.005`；相对 B3弱 AUROC `>=+0.005`；弱召回和弱 AUROC相对 B0至少4/5正向；ISRJ FPR `<=0.20`；H1重建误差 `<=1e-5`；test-v3未使用。

失败则停止周期专家统一接入线，把主要方法收缩为“Focal表征 + 冻结后 BCE线性头”。
