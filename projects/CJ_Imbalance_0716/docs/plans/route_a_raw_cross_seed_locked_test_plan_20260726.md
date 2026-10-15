# Route A RawSawCrossSeedFPR15 一次性 locked-test 计划（2026-07-26）

## 1. 授权与对象

用户已明确授权执行一次 locked test。正式候选固定为：

`RawSawCrossSeedFPR15+H2Others`

冻结推理包：

`outputs/tapc/route_a_raw_saw_cross_seed_locked_test_preflight/frozen_inference_bundle.json`

执行期间不得训练、拟合、搜索、重新校准、调整阈值、学习融合权重或根据测试结果选择 seed。

## 2. 冻结推理规则

- 输入：`[B,2,3000]`，`complex_rms`；
- FMB、SinFM、MFT、VFT、ISRJ：F2 + H2_PosBCE，使用冻结温度和逐标签阈值；
- SawFM：RawStage1SawFMFrontend，使用冻结 Platt 系数/截距；
- SawFM 阈值：seed 2039/2041/2042/2043 为 0.48，seed 2040 为 0.45；
- 不做 seed 概率平均，不形成新 ensemble；五个 seed 分别报告，再汇总均值与标准差；
- locked test 文件只允许 `closedset_6class_test.mat`。

## 3. 对比与指标

同一 seed 比较：

- `H2_Baseline`：六类全部使用冻结 H2；
- `RawSawCrossSeedFPR15`：只替换 SawFM 概率和阈值，其他五类逐元素保持不变。

报告：macro/micro AUPR、macro-F1、Hamming loss、exact match、逐标签 AUPR/recall/FPR/ECE、弱正召回、强正召回、组合指标、标签基数指标、逐 seed 结果和五 seed 均值/标准差。

弱分量定义保持开发阶段一致：复合样本中某正标签功率比其余最强正分量低至少 10 dB。

## 4. 预注册通过门槛

主要候选必须同时满足：

1. 五个 seed 均完整执行且冻结文件/checkpoint/校准/阈值哈希一致；
2. 其他五类的概率、阈值和二值预测与 H2 baseline 逐元素完全相同；
3. 五个 seed 的 SawFM locked-test FPR 均不高于 0.20；
4. 五 seed 平均 macro-AUPR 不低于 H2 baseline；
5. 五 seed 平均 macro-F1 不低于 H2 baseline；
6. 五 seed平均 exact match 相对 H2 baseline 下降不超过 0.01；
7. 五 seed平均 SawFM recall 相对 H2 baseline 提升至少 0.20；
8. 五 seed平均弱 SawFM recall 相对 H2 baseline 提升至少 0.20；
9. locked test 上没有任何拟合或选择操作；
10. 一次性 access marker、预测、指标、测试文件 SHA256 和最终 receipt 完整。

任一门槛失败都必须原样报告，不允许重新运行、改阈值或更换候选。

## 5. 一次性执行规则

1. 先实现 runner 和单元测试；
2. 创建带用户授权记录、候选包 SHA256、runner SHA256 和上述门槛的配置冻结文件；
3. 执行不读取 locked test 的 `--preflight-only`；
4. 确认正式输出目录不存在；
5. 使用 `--authorize-locked-test` 和冻结文件 SHA256 正式运行一次；
6. runner 在第一次读取测试文件前以独占方式写入 access marker；
7. access marker 一旦存在，runner 永久拒绝复用该输出目录。

