# 路线 A：六类 Base-DB 主基线与 DB 路径试验结果（2026-07-23）

## 1. 本轮结论

本轮完成了三个预注册方向：

1. 从随机初始化训练 DB 权重为 0.75 的全标签 `ASL-DB` 混合（Mix75）；
2. 从随机初始化训练 DB 权重为 0.90 的全标签 `ASL-DB` 混合（Mix90）；
3. 以 Base-DB 表征为起点，冻结全部表征和 BatchNorm 状态，仅用 ASL 训练分类头（DBRep-ASLHead）。

三个候选均未通过相对 Base-DB 的晋级门槛，因此不扩展到 VFT、ISRJ。Base-DB 保持为六类阶段的主要对比基线。

结果不是“全部无效”：Mix75/90 稳定地把 SawFM 尾类排序能力推高到了新的 Pareto 点；DBRep-ASLHead 则证明 DB 学到的表征明显优于 ASL 表征。但两种现象都不足以替代完整 Base-DB。

## 2. 固定协议

| 项目 | 设置 |
|---|---|
| 运行环境 | `D:\anaconda3\envs\PY10\python.exe` |
| 数据根目录 | `C:\LJQ_code\干扰数据\复合干扰数据0716` |
| 配置 | `configs/student_a/six_class.yaml` |
| 目标尾类 | SawFM |
| 随机种子 | 2041、2042、2043 |
| Label-LT | N=2400，tail=400，IF=4 |
| 训练轮数 | 20，patience=20 |
| 候选检查点选择 | SawFM AUPR（optimization-role validation） |
| locked test | 未使用 |

晋级要求为候选相对 Base-DB 同时满足：平均 macro-AUPR 不低于 -0.005、平均 SawFM AUPR 不低于 -0.005，且至少 2/3 seeds 的 SawFM AUPR 不低于 -0.005。

## 3. Base-DB 正式提升为六类主基线

现有六个尾类、每类三个种子，共 18 组 Base-DB 与 Base-ASL 的配对结果为：

| 指标 | Base-DB 相对 Base-ASL | 配对胜场 |
|---|---:|---:|
| macro-AUPR | +0.017870，95% bootstrap CI [0.015117, 0.020646] | 18/18 |
| tail-AUPR | +0.022616，95% bootstrap CI [0.011958, 0.033704] | 15/18 |
| macro-F1 | +0.001100 | — |
| Hamming loss | -0.005802 | — |
| tail-F1 | +0.011313 | — |

逐尾类 tail-AUPR 均值增量为：FMB +0.001483、SawFM +0.023688、SinFM -0.002942、MFT +0.016017、VFT +0.047717、ISRJ +0.049733。Base-DB 在总体和 5/6 尾类旋转上占优，因此后续六类试验必须以 Base-DB 为主要对比；Base-ASL 仅保留为传统损失参照。

## 4. SawFM 正式结果

三种候选的三种子均值如下。括号内是相对 Base-DB 的配对均值增量。

| 方法 | 初始化 / 可训练范围 | macro-AUPR | SawFM AUPR | 晋级 |
|---|---|---:|---:|---|
| Base-ASL | 随机初始化 / 全模型 | 0.801320 | 0.712712 | 旧参照 |
| **Base-DB** | 随机初始化 / 全模型 | **0.826986** | 0.736400 | **主基线** |
| Mix75 | 随机初始化 / 全模型，47,718 参数 | 0.777920（-0.049066） | **0.804705（+0.068305）** | 未通过 |
| Mix90 | 随机初始化 / 全模型，47,718 参数 | 0.779534（-0.047453） | **0.805551（+0.069150）** | 未通过 |
| DBRep-ASLHead | Base-DB / 仅分类头，390 参数 | 0.816429（-0.010558） | 0.732451（-0.003949） | 未通过 |

### 4.1 Mix75

- 相对 Base-DB：macro-AUPR -0.049066，SawFM AUPR +0.068305；
- 三个种子的 SawFM AUPR 全部提升，增量分别为 +0.098757、+0.057127、+0.049030；
- macro-F1 -0.116887，Hamming loss +0.064444，说明提升不是无代价的整体改进；
- 最佳轮次为 3、1、3，尾类优势主要在训练早期形成。

### 4.2 Mix90

- 相对 Base-DB：macro-AUPR -0.047453，SawFM AUPR +0.069150；
- 三个种子的 SawFM AUPR 全部提升，增量分别为 +0.099418、+0.059397、+0.048636；
- macro-F1 -0.119401，Hamming loss +0.055741；
- 最佳轮次同样为 3、1、3；
- 与 Mix75 几乎重合，DB 权重从 0.75 增至 0.90 没有解除宏观性能和尾类性能之间的冲突。

### 4.3 DBRep-ASLHead

- 相对 Base-DB：macro-AUPR -0.010558，SawFM AUPR -0.003949；
- 相对 Base-ASL：macro-AUPR +0.015109，SawFM AUPR +0.019739，且两个指标都是 3/3 seeds 提升；
- 三个种子的最佳轮次全部为第 1 轮；
- macro-F1 相对 Base-DB +0.027195，但 Hamming loss +0.079815，说明排序指标和固定阈值行为并未一致改善；
- 最终检查点仅分类器 weight/bias 两个张量改变，非分类器参数和 BatchNorm 缓冲区变化数均为 0，反向解耦实现有效。

## 5. 原因分析

### 5.1 混合损失找到的是尾类偏置点，不是全局更优点

Mix75/90 的 SawFM AUPR 在 3/3 seeds 上大幅提升，但宏 AUPR、宏 F1 和 Hamming loss 同时明显恶化。这说明混合目标能够强化 SawFM 的排序梯度，却破坏了其余标签的共同判别结构。0.75 与 0.90 的结果非常接近，继续在这一区间微调权重的边际价值很低。

同时，本轮候选按 SawFM AUPR 选点，而既有 Base-DB 按原主指标选点。宏观非劣门槛已经阻止候选被错误晋级，但“尾类提升来自混合损失”仍存在检查点选择混杂，不能直接作为论文机制结论。

### 5.2 DB 表征有效，但 ASL 头没有改善 DB 的决策边界

DBRep-ASLHead 明显优于从头训练的 Base-ASL，证明 DB 阶段确实学到了更适合长尾六类识别的表示；这与此前失败的“ASL 表征 + DB 头”形成方向相反的证据。

然而，DBRep-ASLHead 仍落后于完整 Base-DB，而且三个种子在 ASL 更新第 1 轮即达到最佳点。也就是说，问题不是头部训练轮数不足，而是 Base-DB 的分类边界已经与其表征协同适配；切换成 ASL 会立即产生目标错配。反向解耦适合作为“表示来源决定上限”的机制证据，不适合作为当前主方法。

## 6. 决策与下一步最小试验

### 6.1 当前决策

1. Base-DB 正式作为六类阶段主要基线；所有新方法首先报告相对 Base-DB 的配对增量和置信区间。
2. Mix75、Mix90 不晋级到其他尾类；保留为“尾类—总体 Pareto 冲突”的机制证据。
3. DBRep-ASLHead 不作为候选主模型；保留为 DB 表征优于 ASL 表征的反向解耦证据。

### 6.2 下一步优先级

1. **先解除选择指标混杂**：在 SawFM 上做 `{Base-DB, Mix90} × {macro-AUPR 选点, SawFM-AUPR 选点}` 的 2×2 配对试验。只有 Mix90 在相同选点规则下仍提供独立尾类增益，才能把增益归因于混合损失。
2. **若继续研究反向解耦，只做最小边界校正**：加入 epoch-0 Base-DB、较小头部学习率和到 Base-DB 分类头的 L2-SP 约束；目标不是超过尾类指标，而是验证 ASL 更新是否只能造成退化。
3. **主创新路线转向 Base-DB 内部结构增强**：在不改变 Base-DB 主目标的前提下，引入类别条件 margin、梯度冲突约束或尾类专用残差头，并坚持相对 Base-DB 的宏观非劣门槛。

## 7. 完整性与复现审计

- 正式运行：9/9 完成；每种方法 3 seeds；
- 每次运行均生成 `metrics.json`、`run_receipt.json`、`longtail_manifest.json`、`history.csv`、预测和最优检查点；
- 所有运行 `locked_test_used=false`；
- 每次训练样本数 2400，标签重建最大绝对误差 0；
- Mix75/90 均为随机初始化、全模型 47,718 个可训练参数；
- DBRep-ASLHead 均加载同 seed 的 Base-DB 检查点，仅训练 390 个分类器参数；
- 候选与对应 Base-DB 使用相同 manifest SHA256 和 split SHA256；
- 全量测试：123 passed，36 个现有 AMP 弃用警告；
- 新增批处理脚本通过 `py_compile`。

结果摘要 SHA256：

- DB 路径正式结果：`AED4BEB21B7A6B14488EF222B51A599AA4C3363F997DFBCD2CC04B43E255EB10`
- Base-DB 六类基线聚合：`803B71395FD2D658383C7FA4F0B51CFF6EF41E42AA6C7AEB7FCD95E866A126CD`

## 8. 产物位置

- 预注册技术方案：`docs/implementation/route_a_six_class_db_path_plan_20260723.md`
- Base-DB 聚合程序：`experiments/aggregate_route_a_six_class_basedb.py`
- 三方法批处理程序：`experiments/run_route_a_sawfm_db_path_screen.py`
- Base-DB 聚合结果：`outputs/tapc/route_a_six_class_basedb_baseline/route_a_six_class_basedb_baseline_summary.json`
- 本轮正式结果：`outputs/tapc/route_a_sawfm_db_path_screen/route_a_sawfm_db_path_screen_summary.json`
- 单次运行完整记录：`outputs/tapc/route_a_sawfm_db_path_screen/runs/SawFM/<method>/seed_<seed>/`
