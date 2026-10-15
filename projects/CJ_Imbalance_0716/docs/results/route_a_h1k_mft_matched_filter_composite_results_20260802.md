# Route A H1k：MFT 匹配滤波有效支持度的复合弱证据验证

## 决定

**停止 MFT 的 Renyi-2 有效延迟支持度统计量；不进入独立复现、H2、安全集成或 test-v3。**

H0j 的配对 3-vs-8 副本物理干预曾表明该无拟合量可反映 MFT 副本数；H1k 在新的六标签复合源上检验其是否保留弱 MFT 识别价值。结果直接反向，因而不能把单成分可辨识性外推为复合弱证据。

## 冻结来源、统计量和对照

- 新 train-only 复合源 SHA-256：`97c3a9b8c17b53f6c777d3a0cb7672f8414ffae8821631d7b7edb33cd16f2659`；5,040 唯一 receiver identities，`1028677743--1028682782`，63 个组合各 80 条；与所有历史 train source 零身份重叠。
- 仅 evaluation 角色的固定组成匹配端点：weak MFT positives 与 MFT-absent negatives 只在其他五标签向量完全相同时比较；weak mask 在两臂分数计算后才作离线评价。
- 候选：已知 LFM 圆周匹配滤波、固定 delay ROI `[400,1000)`、归一化能量的 Renyi-2 有效支持度；控制：同长度、同模、固定相位置乱 LFM，FFT/ROI/归一化完全相同。
- 零训练参数、零阈值、零校准、零融合；10,000 次类别分层 paired bootstrap；五个固定 receiver 子折只作一致性门。
- `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。

## 结果：正确参考减相位置乱控制

| 端点 | 点估计 | 10,000 次 paired bootstrap 95% CI | 结论 |
|---|---:|---:|---|
| weak MFT AUROC | `-0.168547` | `[-0.232395,-0.103871]` | 显著反向 |
| weak MFT AUPR | `-0.075655` | `[-0.133186,-0.042116]` | 显著反向 |

所有效果、CI 与 receiver 子折门均失败；数据身份、零拟合、零阈值/校准/融合和 test-v3 封存门均通过，因此失败不能归因于实现、泄漏或决策阈值。

## 失败机制

H0j 的配对单 MFT 计数干预控制了副本数，只回答“正确匹配滤波的延迟能量分布会否随副本数改变”。在复合弱 MFT 场景，延迟能量支持度同时受其他干扰成分和低可见 MFT 遮蔽影响；正确参考产生的支持度不再是 MFT presence 的单调证据，甚至逊于相位置乱参考。因而它不是可部署的复合 MFT 局部证据。

## 永久停止范围

禁止在 H1k 身份或任何后续来源上调节/改名该统计量的：delay ROI、Renyi 阶数、能量归一化、匹配窗、参考相位控制、子折、阈值、校准、分类头或融合。新的 MFT 假设必须在数学和物理上不同于 matched-filter delay support/profile，而不能只是其参数或二维化变体。

机器可读结果：`outputs/tapc/route_a_h1k_mft_support/route_a_h1k_mft_support_summary.json`。
