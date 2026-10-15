# 路线 A：SawFM 损失函数 × 检查点选点 2×2 因子试验（2026-07-23）

## 1. 上一轮失败原因

上一轮 Mix75/90 相对 Base-DB 的 SawFM AUPR 提高约 0.068～0.069，但 macro-AUPR 下降约 0.047～0.049。该结果包含两个同步变化：

1. 损失从纯 DB 改为 ASL-DB 混合；
2. 检查点从 macro-AUPR 选点改为 SawFM-AUPR 选点。

因此，现有结果只能证明“混合损失 + 尾类选点”能找到尾类更强、总体更弱的 Pareto 点，不能证明尾类增益由混合损失本身造成。

反向解耦的三个最佳检查点全部出现在第 1 轮，且严格冻结表征后仍低于 Base-DB，说明 Base-DB 表征与 DB 分类头已经协同适配；切换 ASL 头会立即造成目标错配。本轮不继续扩大反向解耦，而先解决更基础的因果混杂。

## 2. 试验矩阵

| 单元 | 损失 | 检查点选择 | 状态 |
|---|---|---|---|
| DB-Macro | DB | macro-AUPR | 复用既有 Base-DB |
| DB-Tail | DB | SawFM-AUPR | 新增训练 |
| Mix90-Macro | 0.9 DB + 0.1 ASL | macro-AUPR | 新增训练 |
| Mix90-Tail | 0.9 DB + 0.1 ASL | SawFM-AUPR | 复用既有 Mix90 |

固定设置：

- 环境：`D:\anaconda3\envs\PY10\python.exe`；
- 数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`；
- 配置：`configs/student_a/six_class.yaml`；
- seeds：2041、2042、2043；
- Label-LT：N=2400，SawFM=400，IF=4；
- 训练：20 epochs，batch size 64，学习率 3e-4；
- 所有模型随机初始化并训练全部 47,718 个参数；
- locked test 禁止使用。

## 3. 需要估计的效应

对 macro-AUPR、SawFM AUPR、macro-F1、Hamming loss 和 SawFM F1 分别计算：

1. Macro 选点下的损失效应：`Mix90-Macro - DB-Macro`；
2. Tail 选点下的损失效应：`Mix90-Tail - DB-Tail`；
3. DB 下的选点效应：`DB-Tail - DB-Macro`；
4. Mix90 下的选点效应：`Mix90-Tail - Mix90-Macro`；
5. 损失与选点交互：`(Mix90-Tail - DB-Tail) - (Mix90-Macro - DB-Macro)`。

## 4. 判定规则

- 若 DB-Tail 已复现大部分尾类增益，而 Mix90-Tail 相对 DB-Tail 无明显优势，则上一轮提升主要来自检查点选择。
- 若 Mix90 在 Macro 和 Tail 两种选点下都稳定优于同选点 DB，则混合损失具有独立贡献。
- 若 Mix90 仅在 Tail 选点下有效且交互项很大，则混合损失主要改变早期优化轨迹，不能作为稳定主方法。
- 候选只有在相同选点规则下相对 DB 满足 macro-AUPR 不低于 -0.005，且 SawFM AUPR 至少提高 0.005，才进入后续尾类验证。

## 5. 执行边界

本轮新增 6 次训练，复用 6 次已有训练，共形成 12 个配对单元。结果首先用于失败归因，不读取 locked test，也不因单一最好种子扩大实验范围。
