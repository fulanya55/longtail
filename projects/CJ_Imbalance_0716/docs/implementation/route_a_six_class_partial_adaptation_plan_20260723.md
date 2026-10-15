# 路线 A 六类局部表征适配筛选方案（2026-07-23）

## 1. 研究问题

上一阶段表明，六类 Head-DB 能提高 macro-AUPR，但不能稳定恢复 SawFM、VFT、ISRJ 的尾类 AUPR。20-epoch 诊断只修复了 VFT，SawFM 和 ISRJ 仍失败；端到端 Base-DB 则在三类上明显更强。

本阶段检验：是否可以只适配最后一个时序块，在避免端到端 DB 表征破坏的同时，获得困难尾类所需的局部表征变化。

## 2. 固定协议

| 项目 | 设置 |
|---|---|
| 数据 | `C:\LJQ_code\干扰数据\复合干扰数据0716` |
| 环境 | `D:\anaconda3\envs\PY10\python.exe` |
| 配置 | `configs/student_a/six_class.yaml` |
| 协议 | controlled Label-LT，N=2400，tail=400，IF=4 |
| 尾类 | SawFM、VFT、ISRJ |
| seeds | 2041、2042、2043 |
| 初始化 | 对应尾类/seed 的六类 Base-ASL 最优检查点 |
| 训练轮数 | 20，patience=20 |
| 模型选择 | optimization role macro-AUPR |
| 分类头学习率 | 1e-3 |
| 最后时序块学习率 | 1e-4，即倍率 0.1 |
| 解冻范围 | `backbone.temporal_backbone.5` 与 `backbone.classifier` |
| locked test | 禁止使用 |

复用上一阶段 Base-ASL、Base-DB、Head-DB20 作为固定对照，不重新训练基线。

## 3. 三种方法

### A6-PU

- 最后时序块与分类头局部解冻；
- 使用 DB loss；
- 不增加保持约束。

目的：判断局部表征适配本身是否足够。

### A6-PU-SP

- 与 A6-PU 相同；
- 对最后时序块增加 L2-SP 起点保持项；
- 冒烟阶段原设 1e-3，但加权项仅约为分类损失的 0.013%；正式协议据此冻结为 0.1，使首轮约占分类损失 1%，分类头不受 L2-SP 约束。

目的：判断局部更新是否仍需要抑制表征遗忘。

### A6-TailMix

- 使用 A6-PU-SP 的解冻和保持设置；
- 当前注册尾类使用 DB 元素损失；
- 其余五个输出使用 ASL 元素损失。

目的：避免 DB 的宏指标收益被非目标、较容易标签吸收，把优化增益集中到当前尾类。

## 4. 运行规模

3 个尾类 × 3 seeds × 3 方法，共 27 个新增正式运行。

每个新增运行必须满足：

- split SHA 与对应 Base-ASL 一致；
- manifest SHA 与对应 Base-ASL 一致；
- 样本数为 2400，标签计数误差为 0；
- 可训练参数只来自最后时序块与分类头；
- 其余模型状态无漂移；
- locked test 为 false。

## 5. 晋级判据

单个候选方法只有同时满足以下条件才可进入六尾类扩展：

1. 相对 Base-ASL 的总体 tail-AUPR 均值大于 0；
2. 相对 Head-DB20 的总体 tail-AUPR 均值大于 0；
3. 相对 Base-ASL 的总体 macro-AUPR 不低于 -0.005；
4. 9 个配对中至少 6 个 macro-AUPR 获胜；
5. 9 个配对中至少 6 个 tail-AUPR 高于 Head-DB20；
6. SawFM、VFT、ISRJ 各自相对 Base-ASL 的平均 tail-AUPR 均大于 0；
7. 三个尾类各自相对 Base-ASL 的平均 macro-AUPR 均不低于 -0.005。

若多个方法通过，以相对 Base-ASL 的总体 tail-AUPR 最大者作为下一阶段候选。若全部失败，则停止扩种子并根据逐尾类结果诊断。

## 6. 预期解释边界

- A6-PU 成功：最后时序块适配已经足够；
- A6-PU 失败而 A6-PU-SP 成功：主要问题是局部适配遗忘；
- 只有 A6-TailMix 成功：主要问题是六标签总损失中的梯度分配；
- 三者均失败但 Base-DB 仍强：需要更深层适配或显式类间解耦，不能继续坚持近冻结方案。
