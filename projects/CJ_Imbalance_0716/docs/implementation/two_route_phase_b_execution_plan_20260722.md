# 两条路线 Phase-B 执行计划

日期：2026-07-22  
状态：已按冻结配置执行完成  
环境：`D:\anaconda3\envs\PY10\python.exe`  
数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`  
locked test：禁止读取

## 1. 路线 A

### 1.1 T0P 与 T2D 扩大验证

| 项目 | 设置 |
|---|---|
| 变体 | T0P、T2D |
| seeds | 2026、2027、2028 |
| train / development eval | 1024 / 300 |
| epochs / patience | 20 / 20 |
| batch size | 64 |
| protocol | Power-LT |
| power levels | 25、30、35 dB |
| distillation weight | 0.20 |
| checkpoint | macro-AUPR |

T2D 保留条件：三 seed 平均 macro-AUPR 不低于 T0P 超过 0.01，weak recall 不出现系统性下降。

### 1.2 低权重 ordering 筛选

固定 seed 2026，独立运行 T2O_V2，ordering weight 为 0.02、0.05、0.10。其他协议与扩大验证一致。

保留条件：相对同 seed T0P，macro-AUPR 下降不超过 0.01，同时方向 margin mean 和 satisfaction rate 均提高，weak recall 不下降。

执行入口：

```powershell
D:\anaconda3\envs\PY10\python.exe experiments\run_route_a_phase_b.py
```

## 2. 路线 B

### 2.1 训练阶段

1. 使用完整 20% nested labeled subset 训练 B0_PRETRAIN 20 epochs。
2. 将 B0_PRETRAIN best model 同时作为后续 Student 和 EMA Teacher 的初始状态。
3. 在 calibration 角色上按每标签 p90/p10 构造正负阈值，候选总覆盖目标约为 20%。
4. 从相同预训练权重分别继续训练：
   - B0_CONT：纯监督续训；
   - B1：伪标签分类；
   - B2：伪标签分类 + feature consistency。

### 2.2 训练机制

- B1/B2 continuation 为 12 epochs；
- 每轮完整遍历 labeled loader；
- unlabeled cap 为 1024，development eval 为 300；
- accepted mask 为空时跳过 Student strong-view 前向，避免 BatchNorm 和 Dropout 的隐式扰动；
- 保存 Teacher 概率 p10/p50/p90/p95、confidence candidate、stability pass/reject、正负伪标签数量；
- checkpoint 使用 macro-AUPR。

### 2.3 判断门槛

B1/B2 只有同时满足以下条件才保留：

1. 至少三个连续 epoch 出现非零且非全覆盖伪标签；
2. 平均有效 coverage 优先位于 5%～30%；
3. 相对 B0_CONT，macro-AUPR 下降不超过 0.01；
4. B2 必须相对 B1 产生可观察的非零 feature consistency，并且识别指标不进一步恶化。

执行入口：

```powershell
D:\anaconda3\envs\PY10\python.exe experiments\run_route_b_phase_b.py
```

## 3. 边界

- 不启用 prototype gate、distribution alignment、参数头或六类实验。
- 本阶段仍为开发阶段，不读取 locked test，不报告最终论文泛化性能。

## 4. 执行结果索引

- 路线 A 汇总：`outputs/tapc/route_a_phase_b/route_a_phase_b_summary.json`
- 路线 B 汇总：`outputs/student_b/route_b_phase_b/seed_2026/route_b_phase_b_summary.json`
- 分析报告：`docs/results/two_route_phase_b_results_20260722.md`
- locked test：未读取。
