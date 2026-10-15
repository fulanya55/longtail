# Route A 三层等权标签条件头 development-v3 预注册

## 1. 单一假设

当前 C4 线性标签条件头使用全正样本 pos-weight BCE，强正样本仍占正样本主体。若主要缺陷确实是标签内可见度长尾，而不是特征中完全不存在弱证据，那么在不改变输入特征、模型结构、参数量和融合器的前提下，将每个标签的弱正、强正和负样本三层损失等权，应改善弱正与负样本的边界。

## 2. 唯一改动

- 冻结 F2 主干、C6 基础头与 RawSaw；
- 输入仍为 final-temporal 的 512 维 mean/std/top-k/max 统计；
- 分类头仍为 `Linear(512,3)`，融合器仍为 6 参数 GenericBoundedFusion，残差上限 0.50；
- T0 使用已冻结的 C4 SameCapacity-NoVisibility 结果；
- T1 仅将线性头训练损失从 pos-weight BCE 改为逐标签 `mean(BCE_weak, BCE_strong, BCE_negative)`；
- 每个样本-标签权重由 fit 角色中对应层的样本数预先计算，mini-batch 内按权重归一化；不搜索层权重；
- 早停仍按 validation 全正样本 macro-AUROC，融合仍用普通 BCE，避免同时改变选择目标和融合损失。

## 3. 固定训练协议

- development-v3 SHA-256：`b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`；
- 五种子：2039--2043；角色及身份隔离复用冻结 role indices；
- AdamW，学习率、weight decay、batch size、epoch、patience、gradient clip 与 C4 相同；
- 每标签 Platt calibration；threshold 角色按弱召回最大化且负样本 FPR `<=0.20`；
- 禁止访问 locked test、test-v2、test-v3。

## 4. 晋级门槛

T1 相对 C4 必须同时满足：macro-AUPR `>=+0.005`、macro-F1 `>=+0.010`、三目标弱召回 `>=+0.030`、弱正-vs-负 AUROC `>=+0.015`、强正召回不低于 `-0.010`、三目标 FPR `<=0.20`、各目标标签 AUPR 不下降超过 `0.010`、至少 4/5 种子弱召回增益为正、SawFM 完全保持、测试边界安全。

## 5. 失败动作

若同参数三层等权仍未改善弱排序，则排除“仅靠全局 final-temporal 统计上的损失重加权即可解决问题”。下一步不再搜索层权重，转向检查弱正与匹配负样本在不同时间层/局部证据图上的可分性，并只在可分性通过门槛的层上训练局部专家。
