# Route A：MFT 冻结 final-temporal 专家有界残差接回预注册方案

## 1. 单一可证伪假设

严格双侧组合匹配筛选表明，MFT 的冻结 final-temporal 线性专家在 early 上达到 AUROC `0.65217`，高于本轮预设的 `0.620` 可用强度。假设该专家保留了 C4 统一分类头未充分利用的 MFT 弱分量局部证据，则在不训练新参数、不使用可见度真值的条件下，将其作为“只作用于 C4 的 MFT logit”的不确定性门控有界残差，应提高 MFT 弱召回与 weak-vs-negative AUROC，并优于破坏样本对应关系的专家打乱对照，同时保护强正、FPR、SawFM 及其他标签。

本轮只验证 MFT，不同时修改 VFT、ISRJ 或主干。

## 2. 数据、模型与冻结边界

- 数据：development-v3，SHA-256 `b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`
- 种子：2039–2043
- 基线：已完成的 `C4_SameCapacity_NoVisibility`，含冻结 F2、RawSaw SawFM 路由、all-positive temporal capacity probe 与 generic bounded fusion
- 专家：严格双侧组合匹配的 `T_FinalTemporal/MFT/best_probe.pth`
- 所有 C4 与 MFT 专家 checkpoint 完全冻结
- 新增可学习参数：0
- test-v3：保持 sealed、unauthorized、unused

## 3. C4 重建审计

从原 C4 checkpoint 重建全部角色 logits：

1. 冻结 F2 产生 base logits 与 temporal statistics；
2. 冻结 RawSaw checkpoint 只替换 SawFM logit；
3. 冻结 capacity probe 与 capacity fusion 只替换 MFT/VFT/ISRJ logits；
4. 将重建的 early logits 与原 C4 `predictions.npz` 中 logits 比较。

最大绝对误差必须 ≤ `1e-5`；否则实验立即停止，不产生方法结论。
为复现原 C4 的数值路径，F2 特征提取批量固定为 128。

## 4. MFT 专家分数与固定残差

严格 MFT 专家使用其 checkpoint 内特征均值/标准差。只用 fit 角色的严格组合匹配 MFT 弱正/缺失负样本，对专家 logit 计算均值与标准差，得到 `z_mft`。

对 C4 的未校准 MFT logit `l`：

- `p = sigmoid(l)`
- `u = sqrt(4 * p * (1 - p))`
- `r = 0.50 * u * tanh(z_mft)`
- `l_new = l + r`

性质：

- `|r| ≤ 0.50`；
- C4 越接近决策边界，专家作用越大；
- C4 对强正或强负越自信，残差越接近 0；
- 只修改 MFT logit，其余五标签逐元素保持不变；
- 不使用真实功率、弱/强 mask 或标签作为推理输入。

## 5. 对照与评估

三个变体：

- `M0_C4_Reconstructed`：零残差；
- `M1_MFT_AlignedBoundedResidual`：样本对齐的 MFT 专家残差；
- `M2_MFT_ShuffledBoundedResidual`：每个角色内固定置换专家分数，其余公式完全相同。

每个变体独立但按相同协议在 calibration 角色拟合逐标签 Platt，在 threshold 角色按 `negative FPR ≤ 0.20` 选择逐标签阈值，并只在 early 报告结果。保存置换 SHA、校准器、阈值和逐样本预测。

## 6. 预注册晋级门槛

M1 相对 M0 必须同时满足：

1. 六标签 macro-AUPR ≥ `+0.005`；
2. 六标签 macro-F1 ≥ `+0.010`；
3. MFT 弱召回 ≥ `+0.030`；
4. MFT weak-vs-negative AUROC ≥ `+0.015`；
5. MFT 强正召回下降不超过 `0.010`；
6. MFT AUPR 下降不超过 `0.010`；
7. MFT negative FPR ≤ `0.20`；
8. 至少 4/5 种子的 MFT 弱召回增益为正；
9. 至少 4/5 种子的 MFT weak-vs-negative AUROC 增益为正；
10. M1 相对 M2 的 MFT 弱召回 ≥ `+0.015`；
11. M1 相对 M2 的 MFT weak-vs-negative AUROC ≥ `+0.010`；
12. SawFM AUPR 和弱召回相对 M0 不下降；
13. SawFM-present / absent macro-F1 均不下降超过 `0.005`；
14. C4 重建误差 ≤ `1e-5`，且全部身份与 test 边界审计通过。

Smoke 不参与晋级。门槛在查看正式结果后不得修改。

## 7. 停止与后续规则

- 全部门槛通过：MFT 专家可进入统一候选；下一步做 `cap=0.25/0.50` 预注册消融、无 uncertainty gate 消融及同容量可学习 MLP 对照。
- 优于 C4 但不优于打乱：判定收益不能归因于样本级专家证据，停止接回。
- weak AUROC 提升但阈值后召回/FPR 不通过：保留排序证据，停止当前固定残差，下一轮只能研究预注册的校准/阈值机制，不能调残差权重。
- 强正或宏观指标退化：判定安全门控不足，停止该机制。
