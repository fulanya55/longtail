# EfficientNet-B1 频次—可见性双轴长尾试验计划

## 1. 试验目标

固定 `efficientnet_b1_1d` 为唯一骨干，回答三个问题：

1. 标签/组合频次减少会造成多大识别退化；
2. 标签真实存在但相对功率降低时，会造成多大可见性退化；
3. 频次稀少与低可见性同时作用时，二者是否产生额外的交互退化，以及频次补偿和可见性训练能否互补。

正式表述统一为“频次—可见性双轴”，不把功率本身称为第三种长尾。根 0716 数据是平衡数据，所有频次长尾均由可复现清单人工构造。

## 2. 固定骨干与训练设置

- 骨干：`efficientnet_b1_1d`；
- 输入：RMS 归一化复数 I/Q，形状 `[B,2,3000]`；
- 分类头：`Linear(raw_dim,256) + LayerNorm + GELU + Dropout(0.2) + Linear(256,6)`；
- 标签顺序：FMB、SawFM、SinFM、MFT、VFT、ISRJ；
- 优化器：AdamW，学习率 `3e-4`，权重衰减 `1e-4`；
- batch size：64；
- 训练轮数：30，AMP，梯度裁剪 5；
- 第一轮随机种子：`20260724、20260725、20260726`；
- 检查点主选点：early-stop role 的 macro-AUPR；次选 tail-AUPR、loss；
- 锁定测试集：在配置冻结和明确授权前禁止读取。

现有 B1 在根训练文件上的 85.28% 组合完全准确率仅作为接口与容量参考。双轴正式试验改用身份分组和功率配对协议，不能与该数字直接横向宣称提升。

## 3. 身份安全的数据划分

只使用各个 `closedset_6class_train.mat`。先按组合和源身份划分，再展开功率视图：

- optimization/train：每种组合 280 个身份，共 17640；
- early-stop：每种组合 40 个身份，共 2520；
- calibration/threshold：每种组合 40 个身份，共 2520；
- development evaluation：每种组合 40 个身份，共 2520。

同一 `sample_id/receiver_noise_seed` 的根数据和九个功率视图必须属于同一角色。九档功率视图不能作为九个独立身份计数，也不能跨角色拆分。

每次运行保存：split SHA-256、长尾 manifest SHA-256、数据文件 SHA-256、配置 SHA-256、环境信息和 locked-test receipt。

## 4. 双轴定义

### 4.1 频次轴 F

频次轴保留两个统计层次：

- `label-LT`：将指定目标标签设为尾标签；
- `combo-LT`：按组合频次构造头—尾组合分布。

第一轮主协议使用受控 `label-LT`：总训练身份数固定为 2400，主筛选不平衡因子 IF=3；IF=4 只作为更强压力测试。平衡对照也固定为 2400 个身份，避免把样本总量误当成长尾收益。

`combo-LT` 在 label-LT 机制通过后作为第二层验证，组合不平衡因子暂定 10。不得直接调用同时混合功率的旧 `mixed_lt` 清单来证明频次效应。

### 4.2 可见性轴 V

以 30 dB 等功率档作为可见性基准。弱可见档根据目标标签所属功率组确定：

- 目标为 FMB、SawFM、SinFM：使用 40/45/50 dB 档；三类保持 30 dB，但相对 MFT/VFT/ISRJ 变弱；
- 目标为 MFT、VFT、ISRJ：使用 10/15/20 dB 档；目标分量本身变弱；
- 25/30/35 dB 用于中等可见度曲线，不作为第一轮最弱条件。

训练时每个身份每个 epoch 只抽取一个弱视图，不能把三个弱档复制成三倍样本。评估时分别报告九档完整功率曲线。

## 5. 第一阶段：双轴机制诊断

先选两个代表性尾标签：

- SawFM：代表固定 30 dB 组在高功率档中的相对弱可见；
- VFT：代表扫动组在低功率档中的绝对弱可见。

对每个目标标签构造 2×2 数据矩阵，模型和 BCE 损失保持不变：

| 单元 | 频次状态 | 可见性状态 | 目的 |
|---|---|---|---|
| D00 | 频次平衡 | 30 dB 等功率 | 公平基准 |
| D10 | label-LT，IF=3 | 30 dB 等功率 | 单独测频次偏置 |
| D01 | 频次平衡 | 目标标签弱可见 | 单独测可见性退化 |
| D11 | label-LT，IF=3 | 目标标签弱可见 | 测双轴联合与交互 |

每个单元运行 3 seeds，共 `2 个目标标签 × 4 单元 × 3 seeds = 24` 个正式 screen run。先做每单元单 seed、3 epoch 的链路冒烟，再启动正式训练。

双轴交互量按指标 `M` 计算：

`Interaction = M(D11) - M(D10) - M(D01) + M(D00)`。

对于 AUPR、F1、recall 等“越高越好”的指标，显著为负表示联合退化超过两个单轴退化的简单相加。

## 6. 第二阶段：补偿方法消融

只在第一阶段最困难的 D11 上比较四种方法，骨干、分类头和训练预算不变：

| 方法 | 频次补偿 | 可见性补偿 |
|---|---|---|
| M0-BCE | 无 | 单弱视图监督 |
| M1-DB | DB Loss | 单弱视图监督 |
| M2-Pair | BCE | 30 dB 与弱档同身份双视图监督 |
| M3-DB-Pair | DB Loss | 30 dB 与弱档同身份双视图监督 |

双视图损失先使用最小、可解释形式：

`L_pair = 0.5 * L_cls(x_30dB) + 0.5 * L_cls(x_weak)`。

为控制计算和数据曝光，M0/M1 使用同样的身份 batch 数和优化步数；M2/M3 以“身份”为 batch 单位。第一轮不加入 EMA、伪标签、无标签池、强一致性或功率 logit ordering，避免与此前已失败或有分类代价的目标混淆归因。

只有当 Pair 能稳定提高 weak recall 且 macro-AUPR 非劣时，才单独建立下一层“强视图到弱视图特征蒸馏”消融；不能直接把它与 M3 一次性捆绑。

## 7. 评价指标

### 7.1 主指标

- macro-AUPR；
- 目标尾标签 AUPR；
- weak-component recall；
- Hamming loss。

### 7.2 必报分层指标

- 6 个标签的 AUPR、F1、precision、recall、specificity；
- 63 个组合的 exact accuracy 和 mean label accuracy；
- head/medium/tail 组合分层结果；
- 9 个功率档逐档 macro-AUPR、目标标签 AUPR、weak recall；
- 按组合阶数 1～6 的指标；
- 预测标签基数误差；
- 固定阈值 0.5 与 calibration role 学习的逐标签阈值两套结果。

阈值无关的 AUPR 用于判断排序/表征是否改善；阈值后 F1、Hamming 和 weak recall 用于判断能否转化为实际决策改善。

## 8. 晋级门槛

### 8.1 双轴现象成立

D11 相对 D00 在至少 2/3 seeds 上同时满足：

- macro-AUPR 下降不少于 0.02；
- 目标标签 AUPR 下降不少于 0.03；
- weak recall 下降不少于 0.05。

同时分别保留 D10、D01，证明退化不是单一轴完全解释。

### 8.2 方法晋级

M3 相对 M0 的 3-seed 配对均值至少满足：

- macro-AUPR `>= +0.005`；
- 目标标签 AUPR `>= +0.010`；
- weak recall `>= +0.020`；
- Hamming loss 不恶化超过 `0.005`；
- 目标标签 AUPR 与 weak recall 均至少 2/3 seeds 为正增益。

若 M1 有效而 M2 无效，说明主要瓶颈是频次；若 M2 有效而 M1 无效，说明主要瓶颈是可见性；若 M3 超过 M1、M2，才支持双轴补偿互补。

## 9. 扩展与最终确认

1. Pilot 通过后，按同一超参数轮换 FMB、SawFM、SinFM、MFT、VFT、ISRJ 六个尾标签；
2. 再加入 combo-LT 和 label-LT×combo-LT 频次压力，不改变可见性定义；
3. 候选结构和超参数冻结后，用两个全新确认 seeds 复核；
4. 只有确认门槛通过，才冻结最终配置并申请一次性读取 locked test；
5. 最终统计以身份为聚类单位做配对 bootstrap，不能把九个功率视图当作独立观测。

## 10. 计划实现文件

以下为后续拟新增模块，本计划阶段尚未实现：

- `cj0724/dual_axis_data.py`：身份元数据与九档配对视图读取；
- `cj0724/longtail_manifest.py`：平衡、label-LT、combo-LT 清单及哈希；
- `cj0724/losses.py`：BCE 与 DB Loss；
- `cj0724/dual_axis_train.py`：D00～D11、M0～M3 统一训练入口；
- `cj0724/dual_axis_evaluation.py`：标签、组合、功率、弱分量和交互效应报告；
- `configs/dual_axis/`：冻结配置；
- `outputs/dual_axis_b1/`：检查点、预测、receipt 和聚合报告。

## 11. 执行顺序

1. 元数据/身份配对审计；
2. 构造并冻结 SawFM、VFT 的平衡和 IF=3 manifest；
3. 4 个数据单元的短冒烟；
4. 第一阶段 24 个 screen run；
5. 计算双轴主效应与交互效应；
6. 在 D11 上运行 M0～M3；
7. 达到门槛后再扩展六标签、combo-LT 和确认 seeds。

