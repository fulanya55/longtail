# TAES/TIM 双论文技术设计分析与实现记录

日期：2026-07-22  
主要环境：`D:\anaconda3\envs\PY10\python.exe`  
主要数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`  
locked test：未授权、未读取

## 1. 方法思想的核心判断

该设计不是简单地为同一模型包装两篇论文，而是让两篇论文共享可信实验底座，同时用不同科学问题形成彼此隔离的证据链。

- TAPC-CJNet（论文 1）解决标签频次、组合频次和成分功率三轴不均衡。其关键不是 prototype 本身，而是使用同一 `receiver_noise_seed` 的功率视图提供有方向的物理监督：强视图的对应标签 logit 应高于弱视图，弱视图的标签条件表示向强视图蒸馏，未改变成分只允许有限置信度漂移。
- DAPS-CJNet-v2（论文 2）解决低标注条件下伪标签偏差和参数负迁移。关键不是 EMA Teacher 本身，而是逐标签 positive/negative/uncertain 决策、校准角色上的动态阈值、prototype margin、双弱视图稳定性、分布对齐和参数安全 mask 的联合可靠性。

两条路线都必须遵守 optimization、early-stop、calibration、threshold-selection、locked test 的角色边界。功率扫描数据与根训练集共享身份，因此只能继承根角色并作为配对视图，不能重新随机划分。

## 2. 现有底座与新增实现

原工程已经具备复数 I/Q backbone、四/六类严格筛选、五路 development 角色、CB-ASL、同身份功率索引、Student-A 原型分支、Student-B 半监督训练、校准、阈值和配置冻结。

本次新增或补全：

1. `cjlt/engine/combination_metrics.py`
   - 每组合 exact match、sample-F1、Hamming loss、漏检/虚警标签；
   - head/middle/tail、seen/unseen、Worst-5 汇总；
   - 按真实标签数的 subset accuracy、sample-F1、Hamming loss 和 cardinality MAE；
   - 组合—功率联合接口。
2. `cjlt/losses/paired_power.py`
   - 新增真正的 label-wise directional logit ordering；
   - 保留低功率到高功率 stop-gradient 表示蒸馏；
   - 新增 unchanged-component confidence stability。
3. `cjlt/data/longtail_manifest.py`
   - Power-LT 与 Mixed-LT；
   - 标签、组合、弱成分计数和 head/middle/tail 分组；
   - 样本 ID、样本 ID SHA-256、split SHA-256 和 manifest SHA-256。
4. `cjlt/models/student_a.py` 与 `cjlt/engine/student_a_trainer.py`
   - 可选 label-gated parameter heads；
   - T0-T5 损失开关；
   - 10% warmup、10%-30% ramp；
   - 分类、prototype、蒸馏、方向排序、稳定性、参数损失逐项写入历史；
   - NaN/Inf 立即中止并由运行记录器保存失败 receipt。
5. `cjlt/data/receiver_perturbation.py`
   - I/Q 增益失配、正交误差、相位噪声、小频偏、幅度缩放、样本平移和局部遮挡；
   - 明确这些仅是可控仿真域偏移，不代表实测数据。
6. `cjlt/analysis/decision_utility.py`
   - 识别结果到候选抗干扰措施的规则级映射；
   - action accuracy 和 cost-weighted decision loss；
   - 明确不是实际雷达闭环控制。
7. `experiments/run_tapc.py`
   - 独立 T0-T5 入口；
   - Balanced、Label-LT、Combo-LT、Power-LT、Mixed-LT；
   - 参数统计只在 optimization 上拟合；
   - calibration 只拟合温度，threshold-selection 只选择阈值；
   - 不提供任何绕过 locked test 授权的入口。

## 3. 实验留痕规范

每个 `run_tapc.py` 输出目录至少包含：

- `scientific_config.json`：全部科学参数与 config SHA-256；
- `environment.json`：Python、PyTorch、CUDA 和 GPU；
- `events.jsonl`：开始、数据清单、训练、完成或失败事件；
- `multi_axis_profile.json`、`longtail_manifest.json`：样本 ID、计数和哈希；
- `history.csv`：逐 epoch 各项损失、指标、耗时和辅助损失 ramp；
- `best_model.pth`、`prototype_diagnostics.json`；
- `calibration.json`、`thresholds.json`；
- `predictions.npz`、`metrics.json`；
- T5 的 `parameter_normalizer.json`、`parameter_metrics.json`；
- `run_receipt.json`，失败时另有 `failure.json`。

同一输出目录只允许相同配置哈希通过 `--resume` 跳过。参数变化必须使用新的输出目录，避免旧结果被覆盖。

## 4. 推荐实验顺序

1. 四类 Balanced 下完成 BCE、Focal、ASL、CB-ASL 的 T0 基线。
2. 固定 CB-ASL，完成 T0-T4：分别识别 prototype、directional ordering、联合和 stability 的增益。
3. 在 Label-LT、Combo-LT、Power-LT、Mixed-LT 下重复 T0 与 T4，先判断 tail/weak 指标是否达到预注册保留门槛。
4. 只有四类至少达到主要保留门槛，才运行 T5 参数接口和规则级 decision utility。
5. 再进行接收机扰动、参数 OOD、GPU/CPU latency。
6. 四类证据链成立后才扩展六类、5 seeds 和 63 组合。
7. 配置冻结并获得用户明确授权后，才执行一次 locked test。

## 5. 本次验证结果与边界

- `PY10` 全量单元测试：94 passed。
- T4/Mixed-LT 真实数据 dry-run 成功，生成 split 和 manifest SHA-256。
- T5/Power-LT 一轮小样本 GPU smoke 成功，输出目录：`outputs/tapc/smoke_t5_powerlt_20260722`。
- smoke 使用 64 个 optimization 样本、16 个 development 样本、1 epoch，只验证功率配对、方向排序、稳定性、参数头、校准、阈值和留痕链路可运行。
- smoke 的 macro-F1、weak recall 和参数 MAE 不可用于论文性能主张；正式结论仍需完整 development、5 seeds、配对统计和冻结后的单次 locked test。
