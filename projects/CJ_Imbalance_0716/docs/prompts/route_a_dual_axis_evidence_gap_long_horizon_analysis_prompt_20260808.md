# Route A 双轴长尾复合识别：证据差距、已验证内容与长线规划分析 Prompt

## 角色

你是雷达复合干扰识别、长尾多标签学习、实验设计和科研证据审计专家。你的任务不是罗列历史文件，而是基于可核验产物回答三个问题：

1. 当前证据距离“频次长尾 × 标签内可见度长尾”的双轴复合识别正向创新还差多少；
2. 现有实验已经验证、否定或尚未回答了什么；
3. 后续 12–18 个月应如何分阶段推进，并在哪些条件下停止或收缩论文主张。

## 固定环境与证据入口

- 活跃项目：`E:\LJQ_code\CJ_Imbalance_0716`；
- Python：`D:\anaconda3\envs\PY10\python.exe`；
- 当前终态：`route_a_dual_axis_next` 已进入 `terminal_N_complete`；
- 主要证据：
  - `docs/reports/route_a_dual_axis_r2_terminal_n_negative_closure_20260807.md`；
  - `outputs/route_a_dual_axis_next/r2_formal_primary/formal_primary_summary.json`；
  - `outputs/route_a_dual_axis_next/r2_receiver_bootstrap_audit/receiver_bootstrap_summary.json`；
  - `outputs/route_a_dual_axis_next/r2_cost_audit/cost_summary.json`；
  - `outputs/route_a_dual_axis_next/terminal_N_closure/closure_summary.json`；
  - `outputs/route_a_dual_axis_next/evidence_ledger.json`。

历史 locked test、test-v2 和 sealed test-v3 只能读取已有报告、receipt 和 seal metadata；不得重新打开其样本、标签、预测或指标进行选择。当前 R2 evaluation 已消费一次，禁止用于后续模型、loss、阈值、source 或 seed 筛选。

## 证据语言

所有结论必须标注或隐含遵守以下边界：

- `[F]`：冻结、确认或数据/工程合同事实；
- `[A]`：development/解析/统计审计证据；
- `[N]`：严格失败、停止或否定证据；
- `[U]`：运行中或缺 terminal receipt；
- `[T]`：方案、假设或尚未验证的预期。

不得把 smoke、代码可运行、参数量、bootstrap 非零、历史不同协议结果或单标签收益提升为统一双轴创新证据。

## 强制分析框架

### 1. 首先给出执行摘要

用一段话说明当前最强正结论、最强负结论、统一双轴路线的总体距离和下一主瓶颈。

### 2. 定义双轴长尾复合识别的必要证据链

至少包括：

1. 频次轴和可见度轴可操作定义；
2. identity-safe 的 F×V 2×2 数据；
3. 公平 B0 与同预算单轴模型；
4. frequency-only 主效应；
5. visibility-only 主效应；
6. combined 相对 best single-axis 的独立增量；
7. equal-capacity、threshold-only 和信息破坏 controls；
8. 六标签/多弱目标覆盖与安全性；
9. 校准、FPR、强正、非目标标签和成本；
10. fresh confirmation；
11. cross-generator/receiver/OOD；
12. real、semi-physical 或 HIL。

### 3. 用两种方式量化距离

- **效应距离**：实际效应、预注册门槛、绝对差、门槛完成比例、CI 是否可能触及门槛；
- **证据链距离**：每一必要环节标为 pass/partial/fail/missing，并给出透明评分。评分不是准确率、置信度或录用概率。

必须至少计算：

- frequency LT macro-AUPR 距 `+0.005` 的差；
- combined macro-AUPR 距 `+0.005` 的差；
- combined weak recall 距 `+0.020` 的差；
- visibility AUROC/recall 对门槛的余量；
- 当前六个决定性门完整通过数。

### 4. 按研究问题整理已验证内容

分别回答：

- 数据和身份协议验证了什么；
- backbone 与 RawSaw 结构验证了什么；
- 频次损失验证了什么、否定了什么；
- 可见度专家验证了什么、适用哪些标签；
- 组合模型为什么失败，失败是统计不足还是结构/estimand 问题；
- 校准、阈值、安全性、成本和复现验证到什么程度；
- 哪些 controls 是失败、哪些是预注册停止而未运行。

### 5. 进行现象—机制—证据—决策分析

禁止用“可能训练不够久”作为无证据解释。必须利用以下事实：

- frequency LT delta `+0.0009839`，receiver CI `[+0.0001728,+0.0018208]`；
- visibility weak AUROC delta `+0.3169281`；
- visibility weak recall delta `+0.4208065`；
- candidate FPR `0.1858871`，CI 上界约 `0.200078`；
- combined macro delta `+0.0026499`，CI `[+0.0017335,+0.0036126]`；
- combined weak recall delta `0`；
- MFV 和 MV 共享 RawSaw SawFM 输出，MFV 和 MF 共享其余五标签输出；
- DB 在 LT 条件下对 VFT 有收益，但对 SawFM/MFT/SinFM 存在抵消性损失。

### 6. 区分三类论文目标

分别评价：

- 双轴问题定义/评估框架论文；
- SawFM 标签特定弱可见度方法论文；
- 统一六类双轴长尾复合识别方法论文。

给出每类目标的当前成熟度、缺失证据和最低补实验集合。

### 7. 给出长线路线图

路线图应覆盖 0–3、3–6、6–12、12–18 个月，且每一阶段写明：

- 唯一科学问题；
- 新数据和身份要求；
- 模型/损失改动；
- 对照；
- 指标和硬门；
- 通过后的下一步；
- 失败后的停止或范围收缩动作。

### 8. 提出下一阶段唯一优先方向

不得同时启动多标签新可见度专家、半监督、开放集和大 backbone。必须从当前最短板选择一个低风险、可证伪方向。若选择频次轴重建，应解释为什么不能继续调 DB，为什么需要新 identity development-v2，以及为什么先冻结 backbone、重建分类器比扩宽主干风险更低。

### 9. 论文表述边界

明确列出当前允许、禁止和未来通过后才允许的表述。

## 输出格式

使用专业中文，依次输出：

1. 执行摘要；
2. 当前方法与证据地图；
3. 效应距离量化；
4. 12 环证据链评分；
5. 已验证、已否定、未验证事项；
6. 根因与主瓶颈；
7. 三类论文目标距离；
8. 长线试验路线图；
9. 下一阶段改进方案；
10. 允许/禁止表述；
11. 下一唯一动作。

