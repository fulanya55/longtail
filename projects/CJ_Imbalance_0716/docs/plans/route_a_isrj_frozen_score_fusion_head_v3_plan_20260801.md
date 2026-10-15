# Route A：ISRJ 冻结周期分数融合头预注册（development-v3）

## 目的

验证上一轮固定 `0.5*uncertainty*tanh(z)` 残差是否因表达受限而只保留了周期专家的一小部分收益。保持 C4、主干、310 维周期 HGB、校准协议与阈值协议冻结，仅训练极低容量 ISRJ 分数头。

## 数据与角色

- development-v3 SHA256：`b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`；
- 种子：2039--2043；
- fit：训练冻结分数头；
- validation：仅记录，不选超参数；
- calibration：Platt；
- threshold：FPR 不高于 0.20 的阈值选择；
- early：最终 development-v3 报告；
- 弱分量阈值：+10 dB；
- test-v3 不访问。

## 冻结输入与特征

每个样本只使用：

- `b`：冻结 C4 的 ISRJ logit；
- `z`：冻结周期 HGB 概率转 logit 后，用严格匹配 fit 分数均值/标准差标准化；
- `u=sqrt(4*sigmoid(b)*(1-sigmoid(b)))`；
- 融合特征 `[b, z, u*z, tanh(b)*z]`。

特征只用 fit role 均值/标准差标准化。分类器固定为 `LogisticRegression(C=0.1, solver=lbfgs, max_iter=1000)`，不做网格搜索、不早停。

训练 sample weight：弱 ISRJ 正样本 4.0，其余正样本和负样本 1.0。该权重在结果前固定，不扫描。

## 方案

- F0：原 C4；
- F1：上一轮固定对齐周期残差；
- F2：仅以 `[b]` 训练同损失 C4-only 头，控制弱正重加权/重标定收益；
- F3：以对齐周期 `z` 训练四特征融合头；
- F4：以 block-shuffle 周期 `z` 训练相同四特征融合头，控制参数量与局部统计。

所有方案只替换 ISRJ logit，其余五标签逐元素保持 C4。

## 晋级门槛

F3 必须同时满足：

- 相对 F0：ISRJ 弱召回至少 +0.030、弱 AUROC至少 +0.015、ISRJ AUPR不下降、强召回不低于 −0.010、FPR不高于 0.20；
- 相对 F2：弱召回至少 +0.015、弱 AUROC至少 +0.010；
- 相对 F4：弱召回至少 +0.010、弱 AUROC至少 +0.005；
- 相对 F1：弱 AUROC至少 +0.005；
- 相对 F0 的弱召回与弱 AUROC至少 4/5 种子为正；
- macro AUPR 和 macro F1 均为正；
- SawFM 指标与 SawFM 有/无子组非劣；
- C4 重建误差不高于 1e-5，test-v3 未使用。

若失败，不扫描 C、弱正权重或特征组合；回到证据总结并停止 ISRJ 统一接入线。

## 产物

保存配置、C4/专家哈希、分数头 joblib、归一化、系数、逐角色 logits/预测、校准、阈值、指标、逐种子收据与五种子汇总。

