# Route A：VTSE 平方根不确定性门控确认方案

## 改进依据

首轮 VTSE 在 5/5 种子提高弱召回，并同时提高 AUPR、F1、强正召回和
SawFM-present/absent 子集 F1，但弱召回 +0.02736、弱 AUROC +0.01283、macro-F1
+0.00655 略低于门槛。五种子最大残差均接近 0.35，而平均残差仅 0.041，表明
专家方向正确、硬上限安全，但线性 uncertainty gate 对中等置信样本衰减过强。

## 唯一改动

保持 probe、score normalization、残差上限、数据角色、校准和阈值全部不变，
仅将：

`gate = 4p(1-p)`

改为：

`gate = sqrt(4p(1-p))`

最终残差仍严格满足 `|residual|<=0.35`。平方根门在 `p=0/1` 时仍为零，在
`p=0.5` 时仍为一，但对中等置信样本衰减更缓，符合弱正通常处于非极端概率区间
的机制判断。

## 确认协议与门槛

- development-v2 五种子 2039--2043；test-v2 继续封存。
- 不筛选 cap、top-k、probe 或阈值规则；只确认预注册的平方根门。
- 晋级门与首轮 VTSE 完全一致：macro-AUPR 非劣、macro-F1 +0.010、目标弱召回
  +0.030、目标弱 AUROC +0.015、FPR<=0.20、强召回非劣、4/5 种子正增益、
  SawFM-present 非劣、残差硬上限不越界。

通过后立即冻结模型、probe、score normalization、温度和阈值，生成 test-v2
一次性执行授权与哈希清单；不再进行 development 调参。

