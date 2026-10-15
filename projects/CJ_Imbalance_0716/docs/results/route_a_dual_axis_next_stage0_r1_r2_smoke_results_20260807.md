# 路线 A 双轴下一阶段：Stage 0、R1 与 R2 smoke 阶段结果

## 1. 当前结论

当前 active 分支为 `R_sawfm_scoped_dual_axis`，已完成 Stage 0 观测准入、R1 全新数据生成与审计、R2 主模型接口 smoke。正式五 seed 性能实验尚未开始，因此当前没有任何新的模型有效性结论。

## 2. Stage 0：第二观测准入 `[N]`

机器判定：

```text
decision = no_admissible_non_oracle_observation
selected_branch = R_sawfm_scoped_dual_axis
```

当前路线 A 生成器仍为单路复数 I/Q。0724 的 `iq_reference` 已核实为同一 identity 的固定 30 dB 训练期 teacher，不是单视图部署时可获得的独立物理通道。M1 双接收机、M2 独立重复 PRI、M3 双斜率、M4 外部时延参考均没有已存在的正式数据合同。

Stage 0 未生成数据、未训练模型、未校准、未选阈值。test-v3 只读取封存元数据，状态保持 `access_authorized=false, analysis_allowed=false, used=false`。

## 3. R1：全新 SawFM 配对 development 数据 `[F: data contract]`

已生成：

- VL 20 dB：2,542,145,021 bytes，SHA256 `c2f67999d6ec93d64b2e8a05f615836dabe2dfdc88be3c2e9bfc7644a7324bb3`；
- VH 30 dB：2,542,102,770 bytes，SHA256 `d597a448c1a31322b42b43422192d50292e9e05af5251dcc243d0bfa7a134a8d`。

两视图各有 25,200 个全新 identity，63 个组合各 400 条，六标签正样本均为 12,800。标签、`receiver_noise_seed`、component seed、目标/unknown 元数据和角色在视图间严格对齐；与已登记历史生成器 seed 区间交集为 0。

干预核验：

- VH 所有激活分量 30 dB，weak component 数为 0；
- VL 只把 SawFM 调为 20 dB，其余激活分量保持 30 dB；
- VL 中复合 SawFM 弱正 identity 为 12,400；
- 单标签 SawFM 仍为 strong positive；
- relative power、weak mask 和 component target power 全部逐 identity 符合预注册。

五角色为 optimization 15,120，early/calibration/threshold/evaluation 各 2,520，交集全为 0。FB 清单为 2,400 identity、六标签正样本各 1,200；LT 清单精确为 `[1600, 400, 1213, 919, 696, 528]`；secondary combination-LT IF=4 清单为 8,200 identity。

## 4. R2 smoke `[T: execution only]`

已在 `FB-VH` 和 `LT-VL` 两个极端单元，以单 seed、2 epochs、256 train/256 early identity 完成：

- B0 Focal core；
- B0 DB core；
- 5,521 参数 RawSaw expert；
- Focal+RawSaw 与 DB+RawSaw 的 SawFM-only routing。

smoke 检查全部通过：loss/logit 有限；Focal/DB 共享 core 初始化；B0 参数量 257,511；RawSaw 参数量 5,521；路由后其余五标签 logits 与对应 core bitwise 相等；checkpoint、history、summary 和 receipt 完整。

smoke 的 AUPR/AUROC 数值不用于模型选择、论文结论或正式门槛判断。

## 5. 当前硬门状态

| 硬门 | 状态 |
|---|---|
| Stage 0 非 oracle 第二观测准入 | 失败，路线 S 关闭 |
| R1 新 identity、配对功率与角色合同 | 通过 |
| R2 工程 smoke | 通过 |
| MF 频次主效应 | 未运行 |
| MV SawFM 可见度主效应 | 未运行 |
| MFV 组合独立增量 | 未运行 |
| equal-capacity / threshold-only / 信息破坏控制 | 未运行 |
| 全新 confirmation | 未创建、未运行 |

## 6. 当前允许与禁止的表述

允许：已经建立一个身份隔离、只干预 SawFM 功率、可复核的 development 2×2 数据协议；SawFM-only routing 的实现保持其余五标签 logits 不变。

禁止：声称 DB、RawSaw 或 DB+RawSaw 已在新数据上有效；声称已证明双轴主效应、交互效应或六类统一创新；声称 test-v3 已验证。

## 7. 下一唯一动作

按正式执行补充预注册运行四 cell × 五 seed 的 M0/MF/MV/MFV development 主矩阵。只有 MF 与 MV 分别通过，才自动进入 MEC、MTS 和 RawSaw 信息破坏控制；之后才决定是否创建全新 confirmation。
