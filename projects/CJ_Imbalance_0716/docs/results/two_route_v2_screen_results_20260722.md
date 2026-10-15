# 两条路线 V2 改进实验结果与后续判断

日期：2026-07-22  
环境：`D:\anaconda3\envs\PY10\python.exe`  
数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`  
locked test：未读取

## 1. 总结

- 路线 A：**T2D（双视图分类 + low-to-high embedding distillation）通过本轮识别门槛**，macro-AUPR 小幅超过 T0P。单侧 ordering 成功避免旧 T2 的弱成分召回下降，但在权重 0.20 下仍明显损害整体排序质量，因此 ordering 暂不进入扩大实验。
- 路线 B：公平监督曝光已修复，但 B1/B2/B3 全部没有接受任何伪标签，均未通过机制门槛。当前问题不再是监督步数不公平，而是 EMA Teacher 预热不足、置信度阈值与控制器不匹配，以及零 coverage 时仍执行 student 强增强前向造成训练状态扰动。

## 2. 路线 A 结果

结果目录：`outputs/tapc/route_a_v2_screen/seed_2026`

| 变体 | macro-AUPR | macro-F1@0.5 | weak recall@0.5 | margin mean | direction satisfaction | 判断 |
|---|---:|---:|---:|---:|---:|---|
| T0P | 0.93628 | 0.80049 | 0.62791 | 0.23660 | 0.54984 | 主基线 |
| T2O_V2 | 0.90808 | 0.76961 | 0.62791 | 0.32101 | 0.66956 | 物理方向改善，AUPR 门槛失败 |
| **T2D** | **0.93995** | 0.79824 | 0.62791 | 0.22939 | 0.54063 | 识别门槛通过，保留 |
| T2OD_V2 | 0.90830 | 0.76886 | **0.65116** | 0.31566 | 0.66360 | 弱召回提高，但 AUPR 门槛失败 |

### 2.1 单侧排序修复了什么

旧双向 T2 的 weak recall 为 0.58140；T2O_V2 的 weak recall 恢复到与 T0P 相同的 0.62791。说明停止 weak logit 的排序梯度确实修复了“通过压低弱响应满足 margin”的问题。

### 2.2 单侧排序还没有解决什么

T2O_V2 相对 T0P：

- margin mean 提高 0.08442；
- direction satisfaction 提高 0.11972；
- macro-AUPR 下降 0.02820；
- Hamming loss 从 0.25977 增加到 0.30078。

因此排序损失即使不直接压低 weak logit，仍会通过共享 backbone 改变两个视图的分类表征。当前权重 0.20 对识别目标过强，或者 logit margin 本身与 CB-ASL 的最优排序空间不完全一致。

### 2.3 蒸馏分支的判断

T2D 相对 T0P：

- macro-AUPR：+0.00366；
- weak recall：保持不变；
- fixed macro-F1：-0.00225；
- Hamming loss：+0.00391。

这是小幅正向、尚未达到论文级结论的单 seed 结果，但已经满足“识别不退化”的机制筛选要求。T2D 可进入下一轮扩大验证；其贡献应表述为跨功率表示蒸馏，而不是物理单调排序。

### 2.4 路线 A 下一步

1. 保留 T0P 与 T2D，使用 1024 个训练样本、300 个开发评估样本、20 epochs，先跑 seeds 2026/2027/2028。
2. ordering 暂不与 T2D 联合。先单独测试更低权重 0.02、0.05、0.10，并考虑只作用于分类头输入的独立 residual，而不是直接扰动共享 backbone。
3. 只有低权重 ordering 同时提高方向指标且 macro-AUPR 下降不超过 0.01，才重新测试与 T2D 联合。
4. prototype、stability、参数头继续冻结。

## 3. 路线 B 结果

结果目录：`outputs/student_b/route_b_v2_screen/seed_2026`

| 变体 | macro-AUPR | macro-F1 | Hamming | 最大 pseudo coverage | 每轮 labeled 曝光 | 判断 |
|---|---:|---:|---:|---:|---:|---|
| **B0** | **0.88862** | **0.77132** | **0.25000** | 0 | 512 | 监督基线 |
| B1 | 0.87020 | 0.75809 | 0.27930 | 0 | 512 | 伪标签链路未激活 |
| B2 | 0.87020 | 0.75809 | 0.27930 | 0 | 512 | 与 B1 完全相同，feature consistency 未激活 |
| B3 | 0.87284 | 0.74718 | 0.29102 | 0 | 512 | 动态阈值未建立 coverage |

### 3.1 已成功修复的实验问题

B0、B1、B2、B3 每个 epoch 均处理 512 个 labeled 样本，原先 B0 使用 840 而 B5/B6 只使用 256 的不公平问题已经消除。因此本轮差异不再由监督样本曝光量造成。

### 3.2 Teacher 预热仍然不足

每轮只有 8 个优化 step，两轮 warm-up 共 16 step。EMA momentum 从约 0.99 开始，Teacher 在 warm-up 结束时仍大量保留随机初始化状态，未形成足够尖锐的正负概率。0.80/0.20 阈值下，第 3～8 epoch 的 accepted count 仍全部为 0。

下一轮不应继续简单增加在线 warm-up epoch，而应先完整训练 B0，再将 B0 best checkpoint 同时复制给 Student 和 Teacher，随后启动未标注训练。

### 3.3 零 coverage 仍改变了模型

B1 和 B2 的无监督分类、feature consistency 均为 0，却相对 B0 的 macro-AUPR 下降 0.01843。原因是当前代码在 accepted mask 全空时仍对 strong augmented unlabeled signal 执行 Student 前向：

- BatchNorm running mean/variance 会被强增强未标注数据更新；
- Dropout 前向会消耗随机状态，改变后续监督 batch 的优化轨迹。

所以“无监督 loss 为 0”不等于“未标注分支无影响”。下一轮必须在 accepted mask 全空时跳过 Student strong-view 前向，或者冻结未标注分支的 BatchNorm 统计。

### 3.4 动态阈值控制方向错误

B3 从 tau-pos=0.80 最终提高到约 0.92。当前控制器在正伪标签数为 0 时把 precision 记为 0，优先判定“precision 不足”，从而继续提高阈值；但真实问题是 coverage 为 0，应该降低阈值。

下一轮控制器顺序应修改为：

1. 若 coverage 低于下限，先降低正阈值、提高负阈值；
2. 只有 coverage 足够时，才依据 precision 调高阈值；
3. 正、负伪标签分别统计 precision 和 coverage；
4. 每次更新使用 EMA 平滑或 calibration 分位数，禁止全开/全关跳变。

### 3.5 路线 B 下一步

下一轮矩阵只保留 B0、B1、B2：

1. 完整训练 B0 作为初始化；
2. hard-copy B0 best checkpoint 到 Student 和 Teacher；
3. accepted mask 为空时跳过 Student 未标注前向；
4. 记录 Teacher 每标签概率 p10/p50/p90/p95、稳定性通过率，以及 confidence/stability 各自淘汰数量；
5. 根据 calibration 概率分布选择每标签阈值，使初始总 coverage 处于 5%～30%；
6. 首轮不启用 dynamic threshold，确认 B1/B2 产生稳定非零 coverage 后再单独测试新控制器。

prototype gate、distribution alignment 和参数头继续冻结，直到 B1/B2 在至少三个连续 epoch 中具有非零、非全覆盖伪标签，并且 macro-AUPR 不低于 B0 超过 0.01。

## 4. 实验边界

- 当前均为单 seed、512 样本、8 epochs 的开发阶段实验。
- 所有 checkpoint 按 macro-AUPR 选择。
- 未读取 locked test。
- 路线 A 的 T2D 只获得进入扩大验证的资格；路线 B 尚未获得扩大实验资格。
