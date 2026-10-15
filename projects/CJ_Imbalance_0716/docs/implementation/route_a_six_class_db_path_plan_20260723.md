# 路线 A：六类 DB 主基线、从头混合损失与反向解耦方案（2026-07-23）

## 1. 基线调整

六类阶段将 Base-DB 正式提升为主要对比基线。现有六尾类 × 3 seeds 结果显示，相对 Base-ASL：

- overall macro-AUPR：约 +0.01787；
- overall tail-AUPR：约 +0.02262；
- FMB、SawFM、MFT、VFT、ISRJ 的逐尾类 tail-AUPR 均值为正；
- SinFM 为唯一轻微负向类别，约 -0.00294。

后续候选不再只证明优于 Head-DB，而必须直接与 Base-DB 配对比较。

## 2. SawFM 候选试验

统一协议：

| 项目 | 设置 |
|---|---|
| 配置 | `configs/student_a/six_class.yaml` |
| 数据 | `C:\LJQ_code\干扰数据\复合干扰数据0716` |
| 环境 | `D:\anaconda3\envs\PY10\python.exe` |
| 尾类 | SawFM |
| seeds | 2041、2042、2043 |
| Label-LT | N=2400，tail=400，IF=4 |
| epochs | 20，patience=20 |
| 检查点选择 | optimization-role SawFM AUPR |
| locked test | 禁止使用 |

### 2.1 Mix75

- 随机初始化；
- 全模型训练；
- `asl_db_mix`，DB权重0.75，ASL权重0.25；
- 学习率沿用从头训练基线设置3e-4。

### 2.2 Mix90

- 与Mix75相同；
- DB权重0.90，ASL权重0.10。

两者用于判断：SawFM所需的DB表示能否从训练初期形成，同时用少量ASL抑制DB的潜在整体副作用。

### 2.3 DBRep-ASLHead

- 从对应 Base-DB 最优检查点初始化；
- 冻结全部表征与BatchNorm状态；
- 仅训练390个分类头参数；
- 使用ASL；
- 分类头学习率1e-3。

该方法与失败的“ASL表示 + DB头”构成严格反向对照，用于判断表示学习方向是否是关键因素。

## 3. 晋级规则

候选相对 Base-DB 同时满足以下条件才可进入VFT、ISRJ验证：

1. 平均 macro-AUPR 不低于 -0.005；
2. 平均 SawFM AUPR 不低于 -0.005；
3. 至少2/3 seeds 的 SawFM AUPR 相对Base-DB不低于 -0.005；
4. locked test未使用；
5. Mix方法保持完整从头训练，反向解耦保持严格分类头范围。

若多个方法通过，优先选择SawFM AUPR均值最高者；若所有方法均未超过Base-DB，则Base-DB继续作为六类主基线，候选只保留为机制证据。

## 4. 结果解释

- Mix75/90成功：问题主要是ASL预训练后的优化路径，而非ASL成分本身；
- DBRep-ASLHead成功：DB表示是关键，ASL更适合作为第二阶段边界校正；
- 三者均失败：Base-DB本身就是当前六类SawFM最优训练路径，应停止两阶段改造，创新点转向DB表示的结构增强或跨功率机制。

