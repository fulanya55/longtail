# Route A：ISRJ 周期专家有界残差接回预注册方案

## 1. 假设

已通过局部门槛的310维联合周期HGB包含C4未利用的ISRJ弱分量证据。若该证据可安全进入统一模型，将冻结HGB分数以fit严格匹配样本标准化，并通过C4自身ISRJ不确定性门控和0.5残差硬上限只修改ISRJ logit，应提高ISRJ弱召回和weak-vs-negative AUROC，同时保护强正、FPR、SawFM及其余标签。

## 2. 固定接回

对HGB概率取clipped logit，再用严格匹配fit均值/标准差得到 `z`：

`r = 0.50 * sqrt(4*sigmoid(l)*(1-sigmoid(l))) * tanh(z)`

`l_ISRJ_new = l_ISRJ_C4 + r`

HGB、特征、lag、归一化、残差cap和不确定性指数全部冻结；不使用sample_width/JNR/weak mask真值作为推理输入。

## 3. 对照

- I0：完全重建C4；
- I1：aligned周期HGB有界残差；
- I2：25点块置换周期HGB有界残差。

I1/I2使用同一公式和对应fit分数标准化。三个变体按相同calibration角色Platt、threshold角色FPR≤0.20规则评估early。C4 early logits重建误差必须≤1e-5。

## 4. 组件接回门槛

I1相对I0必须满足：

- ISRJ弱召回≥+0.030；weak AUROC≥+0.015；
- ISRJ AUPR不下降；强召回下降≤0.010；平均FPR≤0.20；
- 弱召回与weak AUROC至少4/5种子正向；
- macro-AUPR和macro-F1均正向；
- SawFM AUPR/弱召回不下降，SawFM-present/absent macro-F1下降≤0.005。

I1相对I2必须满足：弱召回≥+0.015、weak AUROC≥+0.010。重建、身份和测试边界必须通过。

本门槛只决定ISRJ组件是否可加入统一候选，不替代最终六标签宏观晋级门。

