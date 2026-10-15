# Route A：MFT 预测共现上下文门控预注册方案

## 1. 假设

上一轮只读诊断显示，MFT 残差在 SawFM 存在且 VFT/ISRJ 缺失的上下文中排序增益更强、FPR更低。若该规律可由推理期预测量近似，则冻结 C4 的共现标签校准概率构成的无参数软门控，应比全局 MFT 残差获得更高 weak-vs-negative AUROC，并保持弱召回收益。

## 2. 固定公式

只使用 C4 在 calibration 角色拟合的冻结逐标签 Platt 概率：

`g = (p_SawFM * (1-p_VFT) * (1-p_ISRJ))^(1/3)`

`r_context = 0.50 * sqrt(4*p_MFT*(1-p_MFT)) * g * tanh(z_MFT)`

不搜索标签集合、指数、阈值、cap 或权重；新增可学习参数为 0。真实共现标签不进入推理。

## 3. 对照

- K0：重建 C4；
- K1：上一轮无上下文的全局 MFT 有界残差；
- K2：预测上下文门控的对齐 MFT 残差；
- K3：K2 同公式、角色内打乱 MFT 专家分数。

五种子 2039–2043；相同 calibration/threshold 协议；test-v3 不访问。

## 4. 组件晋级门槛

K2 相对 K0 必须满足：MFT 弱召回 `≥+0.030`、weak AUROC `≥+0.015`、MFT AUPR不下降、强召回下降不超过0.010、FPR≤0.20，且弱召回和 weak AUROC均至少4/5种子为正。

K2 还必须：

- 相对 K1，weak AUROC `≥+0.005` 且 FPR不更高；
- 相对 K3，weak AUROC `≥+0.010`、弱召回 `≥+0.015`；
- macro-AUPR/F1均为正增益；SawFM保持；两个SawFM子集macro-F1下降不超过0.005；
- C4重建与测试边界审计通过。

这是单标签组件门槛，不替代最终统一模型的 macro-AUPR `+0.005`、macro-F1 `+0.010` 等总门槛。组件通过后仍需与其他合格标签专家联合验证。

