# 路线 B 重启：纯监督复合干扰联合参数感知实验方案

日期：2026-07-23  
数据：六分类 0716 平衡根训练集  
边界：development roles only，locked test 禁止读取

## 1. 科学问题

在完全监督条件下，类别条件物理参数回归是否能够改善复合干扰共享表征，并在不损害多标签识别的前提下提供可用的端到端参数估计？

第一阶段只使用每类两个最小参数目标：JNR 与一个类别特有参数，共 12 个输出。参数损失仅在 `label=1 AND parameter_mask=1` 的位置计算。

## 2. 公平消融

| 变体 | 参数头 | 参数梯度进入骨干 | 参数权重 |
|---|---|---|---|
| C0 | 否 | 否 | 0 |
| C1 | 是 | 否 | 固定 0.1 |
| C2 | 是 | 是 | 固定 0.1 |
| C3 | 是 | 是 | 前 20% 训练为 0，20%–40% 线性升至 0.1 |

C1 用于判断参数是否可从分类表征中估计；C2/C3 用于判断参数监督是否能够改善共享骨干。四组共享完全相同的数据角色、随机种子、分类损失、训练预算和模型选择规则。模型只按 early-stop macro-AUPR 选点，参数指标不参与 checkpoint 选择。

## 3. 数据与统计

- seeds：2041、2042、2043；
- optimization / early-stop / calibration / threshold-selection 使用同 seed 的固定组合分层角色；
- 先执行 512/128 样本、2 epoch GPU smoke；
- 阶段 S1 使用组合分层的 4096 optimization、512 early-stop/calibration/threshold-selection，20 epochs、patience 5、batch size 64，完成三种子 C0-C3 机制筛选；
- 只有 S1 通过 go/no-go 的候选才进入 17,640 optimization 全量 40 epochs 确认；
- locked test 保持冻结。

## 4. 指标

- 分类：macro-AUPR、macro-F1、Hamming loss、subset accuracy；
- oracle 参数：在真实类别存在且参数有效的位置报告每目标原单位 MAE/RMSE、macro target NMAE；
- end-to-end 参数：按校准后类别预测门控，报告参数覆盖率和已检测位置 NMAE；
- 必须分别报告每类 JNR 与类别特有参数，不能只给一个总平均。

## 5. Go / no-go

相对 C0，候选需满足：

1. 三种子平均 macro-AUPR 不低于 C0 超过 0.005；
2. 至少 2/3 seeds 的 macro-AUPR 不下降；
3. 相对 C1，oracle macro target NMAE 有明确下降，或在相近 NMAE 下提高端到端参数覆盖率；
4. 若 C2 伤害分类而 C3 消除伤害，保留 warm-up；若 C2/C3 均无分类或参数收益，则参数头只作为附属输出，不声称反哺识别。
