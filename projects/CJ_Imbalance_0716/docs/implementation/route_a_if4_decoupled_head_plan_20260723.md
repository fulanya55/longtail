# 路线 A：IF=4 表征—分类头解耦实验方案（2026-07-23）

## 1. 实验动机

端到端 Mix-0.25 已把 MFT、ISRJ 的 tail-AUPR 恢复到 ASL 水平附近，但 MFT macro-AUPR 为 -0.00170，未通过严格正增益门槛。Mix-0.50 又在 ISRJ 上退化，说明 DB 梯度强度会影响共享表征。

本阶段检验：DB 的 IF=4 失败主要来自“表征被重平衡梯度破坏”，还是来自“最终分类边界本身不合适”。

## 2. 两阶段结构

第一阶段直接复用每个 tail/seed 对应的 20-epoch ASL 最优检查点。第二阶段：

- 冻结 complex stem、temporal backbone、BatchNorm 统计和训练辅助分支；
- 保留 ASL 已训练分类头初始化，不随机重置；
- 只训练 `Dropout + Linear(64,4)` 分类头，共 260 个参数；
- 第二阶段模型选择仍使用 macro-AUPR；
- 学习率 `1e-3`，训练 10 epochs，patience 10。

第二阶段损失：

- Head-ASL：继续使用 ASL，控制“额外头训练”效应；
- Head-Mix25：使用 Mix-0.25，候选方法；
- Head-DB：使用原始 DB，用于判断冻结表征能否消除 DB 尾类伤害。

## 3. 正式矩阵

- IF：名义 4，实际 3.8；
- tails：FMB、MFT、ISRJ；
- seeds：2036、2037、2038；
- 固定 `N=1600`、tail positives=400；
- 新训练：`3 heads × 3 tails × 3 seeds = 27`；
- 复用基线：Base-ASL、Base-DB、端到端 Mix-0.25；
- locked test：禁止使用。

每个第二阶段运行必须记录初始 ASL 检查点路径及 SHA256，并验证所有比较共享相同 split 和 manifest。

## 4. 预注册门槛

Head-Mix25 对 MFT、ISRJ 必须同时满足：

1. tail-AUPR 高于 Head-ASL；
2. tail-AUPR 相对 Base-ASL 不低于 -0.005；
3. macro-AUPR 相对 Base-ASL 不低于 -0.005；
4. tail-AUPR 相对端到端 Mix-0.25 不低于 -0.005；
5. per-label F1 模式下 macro-F1 相对 Base-ASL 不低于 -0.005；
6. Hamming loss 相对 Base-ASL 恶化不超过 +0.005。

FMB sanity check 要求 macro-AUPR 和 tail-AUPR 相对 Base-ASL 均不低于 -0.005。

机制诊断：若 Head-DB 相对 Base-DB 的 MFT、ISRJ tail-AUPR 均提高，则支持“原始 DB 主要破坏共享表征”；若仍下降，则主要是分类边界或共现权重问题。

## 5. 产物

- `experiments/run_tapc.py`：初始检查点与 classifier-only 模式；
- `cjlt/models/student_a.py`：冻结参数和冻结状态管理；
- `experiments/run_route_a_if4_decoupled_head.py`：正式 runner；
- 输出：`outputs/tapc/route_a_if4_decoupled_head/`。
