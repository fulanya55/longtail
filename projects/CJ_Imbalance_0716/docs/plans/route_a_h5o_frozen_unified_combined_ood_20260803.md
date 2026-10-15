# Route A H5o：冻结统一候选的联合 OOD 外部确认

日期：2026-08-03  
状态：预注册；尚未生成正式 H5o 数据。

## 单一可证伪假设

冻结的 `H1_B2_MFTSharedDecision_VFTRidgeFusion_v2` 不仅能在 H3 的新标准身份源上
相对冻结 H1 改善 MFT/VFT/ISRJ，还能在**新的联合分布偏移**中保持该收益：ISRJ
参数处于预先固定的高副本区 25--30 us、9 个副本，接收机白噪声替换为等功率 proper-complex
AR(1) 噪声（`rho=0.65`）。

H5o 不修改任何模型。若失败，结论只能是该冻结候选未建立完整 OOD 鲁棒性，禁止在
H5o 上调结构、融合、损失、校准、阈值、JNR、参数范围或噪声系数。

## 冻结工件和比较臂

- 候选 Q6：`H1_B2_MFTSharedDecision_VFTRidgeFusion_v2`，使用
  `frozen/20260801_route_a_h1_b2_mft_vft_shared_policy_v2/frozen_manifest.json`
  中五个 seed（2039--2043）的原始工件。
- 基线 Q0：相同五 seed 的冻结 H1 + 相同 RawSaw SawFM 路由。
- 机制对照：既冻结的 Q3、Q4、Q5 VFT 容量/对齐/时间顺序控制，以及 Q1、Q2
  MFT/ISRJ 局部路径。所有项只推理。
- 禁止：任何 H5o fit、normalizer fit、early stopping、checkpoint 选择、校准、
  阈值搜索、融合系数更改或模型选择。

## 数据协议

- 基础随机种子：`850000000`；新 train-only 标准六类来源为 63 个组合各 80 条，
  共 5,040 条。
- 预期 receiver/noise identity 范围：`1048677743--1048682782`；生成前必须扫描
  可读历史 train-only 来源，确认零交集；正式审计再次检查实际 identity。
- 基础源仅将 ISRJ 参数设为 `sample_width_s=[25,30] us`、`num_replicas=9`；其他
  生成合同保持官方六标签配置。
- I/Q 只执行可审计变换：对每条 identity 去掉由该 identity seed 产生的精确白噪声，
  加入相同功率 proper-complex AR(1) 噪声，`rho=0.65`。标签、参数、种子和所有非
  I/Q HDF5 数据集必须逐值保持不变。
- 不生成 test split，且不读取 historical locked test、test-v2 或 test-v3 数据。

H5o 是项目内的物理 OOD 扩展，不是官方均匀白噪声 0716 的替代测试集；它不能替代
最终一次性 test-v3。

生成器 dry-run 先否决了最初的 10 副本、35--40 us/9 副本及 30--32 us/9 副本，
原因是它们违反官方 `validate_replica_capacity` 的物理容量合同，且未生成任何数据。
正式配置因此采用该合同允许的 25--30 us/9 副本。H5o 的新增分布偏移严格限定为
此前未用于完整候选的 `rho=0.65` AR(1) 接收机噪声与该高副本区的联合；不得声称
其 ISRJ 参数范围比 H0s 更远。这一修正只恢复物理可生成性，不来自 H5o 指标或模型结果。

## 指标与晋级门

按 H3 的冻结决策直接计算 Q6 对 Q0 的结果，不重新选阈值。以下条件必须同时通过，
H5o 才能作为“完整候选在一个新联合 OOD 条件下的冻结确认”：

1. macro-AUPR `>= +0.005`，macro-F1 `>= +0.010`；
2. MFT/VFT/ISRJ 平均弱召回 `>= +0.030`，平均 weak-vs-negative AUROC
   `>= +0.015`；
3. 每个目标 AUPR 不低于 Q0 `-0.010`，目标平均 FPR `<=0.20`；
4. 三目标平均强正召回不低于 Q0 `-0.010`，且至少 4/5 seed 的主弱召回增益为正；
5. SawFM 概率和决策严格不变；
6. MFT 的 Q1-negative-to-Q6-positive 翻转为 0、hard-negative FPR 平均降低至少
   `0.010`；
7. Q4 对齐 VFT 弱 AUROC 相对 Q3 和 Q5 各至少 `+0.005`，且至少 4/5 seed 为正；
8. 数据身份、AR(1) 过程、冻结工件哈希和 test-v3 密封合同全部通过。

失败将关闭“该已冻结候选的 H5o 联合 OOD 确认”，但不会推翻 H3 的标准新身份结果；
通过仅补充一个 OOD 条件，仍不解决 C3/C4 容量因果门，也不授权 test-v3。

## 保存与验证

保存 generation receipt、identity audit、源 SHA-256、五个 seed 的预测/指标、冻结
工件哈希和总 summary。先执行 artifact-only smoke 与单元测试；通过后才执行正式
数据生成及五 seed 冻结推理。
