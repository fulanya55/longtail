# Route A 连续物理可见性加权 development-v3 预注册

## 1. 单一假设

弱正不应被建模为与强正不同的新类别。更低风险的机制是维持 all-positive presence
分类目标，同时根据训练期相对功率差连续提高低可见度正样本梯度。这样既保留强正提供
的稳定正类方向，又避免 10 dB hard mask 的边界不连续。

## 2. 固定定义

对复合样本中某正标签，令 `gap = P_label - max(P_other_positive) <= 0 dB`。

`severity = sigmoid((-10 - gap) / 2.5)`，范围 0–1；单标签正样本 severity=0；
负样本 severity=0。正样本 BCE 权重为 `1 + severity`，负样本权重为 1；仍保留每标签
pos_weight 处理正负数量。10 dB 中心和 2.5 dB 温度均预注册，不搜索。

## 3. 冻结项

final-temporal 512→3 线性头、RawSaw、六标量融合器、参数量、数据角色、五种子、
校准、FPR≤0.20 阈值、top-k=8 和 ±0.50 残差全部不变。test-v3 禁止访问。

## 4. 对照与门槛

主要对照为同参数量 all-positive C4；次要对照为 hard-mask C3 与 soft-anchor S1。
连续候选必须相对 C4 同时提高弱召回与 weak-vs-negative AUROC，强正召回下降不超过
0.010，FPR≤0.20；相对 C0 达到弱召回 +0.030、AUROC +0.015，至少 4/5 种子弱召回
增益为正，SawFM 保持 RawSaw。

若未通过，停止仅靠样本重加权的 VTSE 目标线，转向同 receiver identity 的高低功率
配对一致性或连续可见性辅助回归；不继续搜索温度或倍率。
