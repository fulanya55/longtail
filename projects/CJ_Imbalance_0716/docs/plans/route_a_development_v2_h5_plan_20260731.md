# Route A development-v2 H5 头部负迁移验证计划（2026-07-31）

## 1. 研究问题

在 F2 表征完全冻结时，六标签共享线性分类头中的 SawFM 输出竞争，是否会压制
FMB、SinFM、MFT、VFT、ISRJ，尤其是 MFT/VFT/ISRJ 的弱分量证据？

本轮只验证分类头层面的负迁移，不修改主干、校准器算法或阈值算法。旧
locked test 已经消耗，禁止再次读取、筛选、校准或调参。

## 2. 新数据与证据边界

- 生成器：`可配置组合干扰数据生成`，物理参数与 0716 六类闭集保持一致。
- 新基础随机种子：`20260731`，与 0716 的 `20260716` 分离。
- development-v2：新目录中的 `closedset_6class_train.mat`，25200 条。
- test-v2：新目录中的 `closedset_6class_test.mat`，25200 条；生成后封存。
- 样本身份：train/test 由生成器固定 split offset 与独立 `receiver_noise_seed`
  构造，不把九挡功率视为独立身份。
- 本轮允许读取 development-v2；test-v2 只允许生成器完整性验证和文件级
  SHA-256，不允许进入训练、特征提取、校准、阈值或指标计算。

## 3. 对照与候选

共同条件：使用历史 development 阶段得到的 F2_DFMS_Full 五个冻结检查点，
只在全新 development-v2 上提取 128 维表征并训练新线性头。每个 seed 先在
development-v2 的 optimization role 内重建与历史路线一致的 `label_lt` 清单：
总样本数 2400，按固定代码类别顺序 `[FMB,SawFM,SinFM,MFT,VFT,ISRJ]`
的目标计数为 `[1600,400,1213,919,696,528]`。C6 和 H5 必须使用同一份
manifest，SawFM 共存样本全部保留。

- C6：`Linear(128, 6)`，六标签正样本加权 BCE。
- H5：`Linear(128, 5)`，标签为 `[FMB, SinFM, MFT, VFT, ISRJ]`；
  保留所有 SawFM 共现样本，但不产生 SawFM 输出，也不计算 SawFM 损失。

两者使用同一数据角色、初始化规则、训练预算和阈值约束。比较时只在五个共同
标签上计算指标。H5 的五个初始权重/偏置从 C6 对应标签行逐元素复制，以排除
初始化差异。

需要明确：冻结 F2 后，线性头各标签行在 BCE 下几乎独立。本阶段主要是一个
必要的头部/检查点选择/全局校准耦合 sanity check，不能单独证明共享表征负迁移。
因此同时报告：

- C6-AllCal：C6 使用六标签拟合全局温度；
- C6-CommonCal：同一个 C6 checkpoint 只用五个共同标签拟合温度；
- H5-CommonCal：H5 使用五标签拟合温度。

`C6-CommonCal - C6-AllCal` 归因于 SawFM 对全局校准的耦合；
`H5-CommonCal - C6-CommonCal` 才作为头部训练/检查点选择效应。

## 4. 训练、校准与阈值

- 五个预注册种子：2039--2043。
- 优化：AdamW，学习率 `1e-3`，weight decay `1e-4`。
- 批量：256；最多 60 epoch；patience 12；梯度裁剪 5.0。
- 损失：逐标签 `pos_weight = N_neg / max(N_pos, 1)` 的 BCEWithLogitsLoss。
- 早停主指标：五个共同标签的 macro-AUPR；并列时比较
  MFT/VFT/ISRJ 弱正对负样本 AUROC。
- 温度缩放只使用 calibration role。
- 每标签阈值只使用 threshold-selection role；目标是在负样本 FPR <= 0.20
  约束下最大化弱分量召回。
- 最终 development-v2 指标只在 early-stop diagnostic role 报告，明确标记
  `locked_test_used=false`、`test_v2_used=false`。

## 5. 预注册晋级门槛

H5 只有同时满足下列条件，才可声称“删除 SawFM 输出在冻结头阶段有正收益”：

1. 五标签 macro-AUPR 相对 C6 不劣于 -0.005；
2. 五标签 macro-F1 平均提升至少 0.010；
3. MFT/VFT/ISRJ 平均弱分量召回提升至少 0.030；
4. MFT、VFT、ISRJ 各自 AUPR 均不劣于 -0.010；
5. 三个目标标签的平均负样本 FPR 不超过 0.20；
6. 五个种子中至少四个在目标弱召回上为正增益；
7. SawFM-present 子集提升且 SawFM-absent 子集不发生明显退化（macro-F1
   不劣于 -0.005）。

若未通过，结论限定为“删除 SawFM 输出不能在冻结线性头阶段单独解决问题”。
这不等价于否定共享表征负迁移。由于历史 F2 已存在 SawFM 结构例外和
Raw→Stage2 证据衰减，冻结头筛选完成后仍允许设计一个严格受控的最后 stage
小学习率 C6/H5 梯度冲突实验，但必须使用新的 development-v2 角色、保留冻结
F2 对照并预注册 representation-drift 停止条件。

## 6. 必须保存的产物

- 数据生成 dry-run 与正式 receipt、生成配置、文件大小与 SHA-256；
- 每种子的 split hash、样本索引、冻结 F2 checkpoint hash；
- C6/H5 的 checkpoint、history、温度、阈值、预测、逐标签与分层指标；
- 五种子配对差值、置信区间、晋级门判断和最终报告；
- `locked_test_used=false`、`test_v2_used=false` 的机器可读审计字段。
