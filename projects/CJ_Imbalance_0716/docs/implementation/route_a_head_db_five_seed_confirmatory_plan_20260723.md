# 路线 A：Head-DB 四尾类五随机种子确认方案（2026-07-23）

## 1. 目标

三随机种子诊断表明，`ASL representation + frozen backbone + DB head` 在 FMB、MFT、ISRJ 上相对 Base-ASL 同时获得正向 macro-AUPR 和非劣 tail-AUPR，并大幅修复端到端 DB 的 MFT/ISRJ 尾类损失。

本阶段不再调整结构或超参数，只补充独立随机种子和 SinFM，验证该结果是否可重复。

## 2. 冻结方法

方法固定为两阶段：

1. Base-ASL：20 epochs 端到端训练，按 macro-AUPR 选择最优检查点；
2. Head-DB：加载 Base-ASL 检查点，冻结除 `Dropout + Linear(64,4)` 外全部参数和 BatchNorm 状态；
3. 仅训练260个分类头参数，DB loss，10 epochs，学习率 `1e-3`；
4. Base-DB：20 epochs 端到端 DB，对照表征损伤。

数据协议继续固定为：`N=1600`、tail positives=400、标签容差5%、名义 IF=4（实际3.8）、max eval samples=300。

## 3. 运行矩阵

- tails：FMB、SinFM、MFT、ISRJ；
- seeds：2036、2037、2038、2039、2040；
- 复用：前三个 seeds 的既有 Base-ASL/Base-DB，以及 FMB/MFT/ISRJ 的 Head-DB；
- 新增：
  - seeds 2039/2040 的四尾类 Base-ASL、Base-DB、Head-DB：24次；
  - SinFM seeds 2036/2037/2038 的 Head-DB：3次；
- 正式新训练合计：27次。

## 4. 确认门槛

对每个尾类，Head-DB 相对 Base-ASL 必须满足：

1. 平均 macro-AUPR > 0；
2. macro-AUPR 至少3/5 seeds 为正；
3. 平均 tail-AUPR >= -0.005；
4. tail-AUPR 至少2/5 seeds 为正；
5. per-label F1 下 macro-F1 >= -0.005；
6. Hamming loss 恶化不超过 +0.005。

机制门槛：Head-DB 相对 Base-DB 的 MFT、ISRJ tail-AUPR 均必须为正。

全部满足后，Head-DB 才可升级为路线 A 的四类确认方法；否则保留为条件性诊断结果。locked test 继续禁止使用。

## 5. 输出

- runner：`experiments/run_route_a_head_db_confirmatory.py`；
- 输出根目录：`outputs/tapc/route_a_if4_head_db_confirmatory/`；
- 汇总：`route_a_head_db_confirmatory_summary.json`。
