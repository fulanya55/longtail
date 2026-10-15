# Route A stage2 方向性可见度 margin development-v3 预注册

## 单一假设

同 identity 高低功率视图不应完全相等；高功率正证据只能作为低功率正证据的单边下界参考。
将对称一致性替换为“正样本单边 teacher margin + 同组合 hard-negative 排序”，可保留功率相关
判别差异，同时提升低功率正负排序。

## 固定设计

模型、rank=8、3,722 参数、数据角色、训练/validation/early 挡位、优化器、epoch、阈值和所有
分类损失保持上一轮不变。只替换 P2 的配对损失：

- `L_lower = relu(logit_high.detach - logit_low - 0.5)`，分别作用于 10/30 dB 正样本；
- `L_rank = softplus(0.5 - logit_positive_low + logit_hard_negative_low)`；
- hard negative 与正样本具有完全相同的其他五标签组合，只缺失当前目标标签；
- `L_lower` 和 `L_rank` 权重固定为 0.25；去除 map cosine，不搜索 margin 或权重。

对照为上一轮冻结的 P0 SameView、P1 NoPair、P2 symmetric identity pair，新候选为 P3
DirectionalMargin，结构和参数量完全相同。

P3 相对 P0 的晋级门槛：弱正-vs-负 AUROC `>=+0.015`、弱召回 `>=+0.030`、低功率 AUPR
`>=+0.010`、FPR `<=0.20`，至少 4/5 种子 AUROC 增益为正；相对 P1 AUROC `>=+0.010`。
若失败，停止 stage2 配对训练作为主创新候选。

