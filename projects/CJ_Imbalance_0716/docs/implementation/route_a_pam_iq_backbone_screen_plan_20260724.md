# 路线 A：PAM-IQ容量匹配特征网络筛选方案（2026-07-24）

## 1. 目的

PAM-IQ是双轴长尾路线的特征底座，但当前只通过结构与训练链路验证。正式接入可见性轴之前，必须先证明PAM-IQ相对参数量匹配的旧结构具有特征归纳偏置收益，而不是仅凭参数增加获得改善。

本阶段只改变backbone，不启用功率双视图、蒸馏、排序、prototype或标签残差，以隔离特征提取网络的贡献。

## 2. 对照模型

| 编号 | backbone | width | 部署参数 | 作用 |
|---|---|---:|---:|---|
| B0 | `complex_1d_true` | 16 | 21,990 | 当前Base-DB主基线 |
| B1 | `complex_1d_true` | 56 | 260,630 | 容量匹配对照 |
| P0 | `phase_multiscale` | 16 | 253,325 | 完整PAM-IQ |

B1与P0参数量相差约2.8%。P0优于B0但不优于B1，只能说明增加容量有效，不能证明PAM-IQ结构有效。

## 3. 冻结协议

- 环境：`D:\anaconda3\envs\PY10\python.exe`；
- 数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`；
- 六类Label-LT：N=2400，SawFM=400，IF=4；
- label rank：`0 2 3 4 5 1`；
- seeds：2041、2042、2043；
- 随机初始化，T0，完整DB Loss；
- 20 epochs，patience=20，batch size=64，lr=3e-4；
- checkpoint按macro-AUPR选择，同时逐轮记录SawFM-AUPR；
- 每个开发角色最多300样本；
- locked test禁止使用。

每个seed的B0/B1/P0必须具有相同split SHA256和manifest SHA256。

## 4. 指标

主要指标：

- macro-AUPR；
- SawFM-AUPR；
- weak-component recall；
- macro-F1；
- Hamming loss。

训练动态指标：

```text
SawFM forgetting = peak SawFM-AUPR - final SawFM-AUPR
```

该指标用于判断PAM-IQ是否减弱路线A已经确认的训练后期尾类遗忘。

## 5. PAM-IQ相对B1晋级门槛

必须同时满足：

1. 平均delta macro-AUPR不低于-0.005；
2. 平均delta SawFM-AUPR不低于+0.010；
3. SawFM-AUPR至少2/3 seeds为正；
4. weak-component recall平均提高至少+0.020；
5. Hamming loss平均恶化不超过+0.005。

forgetting作为机制指标报告，不单独决定晋级。

## 6. 决策

- 若通过：冻结PAM-IQ结构，进入A1/A2/A3/A4功率双视图与标签条件可见性残差筛选；
- 若未通过但B1明显优于B0：容量是主要因素，PAM-IQ需要删减或调整结构；
- 若B1与P0均不优于B0：停止扩大网络，返回训练目标与梯度冲突方向；
- locked test继续冻结。

## 7. 执行入口

```powershell
D:\anaconda3\envs\PY10\python.exe experiments\run_route_a_pam_iq_backbone_screen.py
```

输出：`outputs/tapc/route_a_pam_iq_backbone_screen/`。

## 8. 执行状态（2026-07-24）

- 1-seed / 1-epoch GPU smoke 已完成；
- 3 模型 × 3 seeds 正式 development 筛选已完成，9/9 runs 成功；
- 同 seed 的 split 与 manifest 哈希一致，`locked_test_used=false`；
- P0 未通过相对 B1 的综合晋级门槛，暂不接入功率双视图；
- 详细结果与下一结构修订方案见 `docs/results/route_a_pam_iq_backbone_screen_results_20260724.md`。
