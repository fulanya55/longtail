# 六类纯标签长尾（阶段 A）

本目录把“标签边际长尾”和“组合频率长尾”显式分开。规划器只求解
63 个非空组合的整数计数，不读取或改写波形；MATLAB 入口再用这些计数复用
现有六类物理生成器。

本次新版本固定标签顺序为 `[FMB,AMA,SawFM,SinFM,MFT,ISRJ]`；入口也兼容
之前的 `[FMB,SawFM,SinFM,MFT,VFT,ISRJ]` 版本。计数向量按 bit-mask
`1:63` 排列，bit 1--6 对应上述六类。

## 指标定义

令 `A` 为 63×6 的组合 multi-hot 矩阵，`n` 为组合计数：

* `IF = max(A*n) / min(A*n)`：六类标签边际频次比；
* `CF = max(n_active) / min(n_active)`：保留组合的频次比；
* `JS`：`n/sum(n)` 与 63 组合均匀 Anchor 之间的 Jensen--Shannon 散度。

所有值均从实际整数计数重算，名义 IF/CF 不能替代 achieved 值。若 JS 或任一
硬约束失败，规划记录为 `js_infeasible`/`infeasible`，不会静默放宽约束。

## 规划器

```powershell
& D:\anaconda3\envs\PY10\python.exe .\solve_pure_label_longtail.py `
  --labels FMB,AMA,SawFM,SinFM,MFT,ISRJ `
  --n 8000 --target-if 3,5,6,10 --cf-cap 8,16,32 `
  --min-per-pattern 8 --js-threshold 0.10 `
  --output .\planning_receipt_n8000_js010.json
```

求解器使用 SciPy/HiGHS MILP；先最大化保留组合数，再在固定覆盖下优化标签
频次展度并用 L1 项靠近 Anchor。JS 是精确后验约束，receipt 保留求解状态、
计数、实际 IF/CF/JS、组合 mask 和 SHA-256。

## MATLAB 生成

```matlab
addpath(genpath('E:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'));
c = [...  % 从 planning_receipt_n8000.json 选取一个 status=feasible 的 counts
];
out = 'E:\论文相关\干扰数据\复合干扰数据0716\six_class_pure_label_longtail_v1_IF3_CF8_N8000';
r = generate_closedset_6class_pure_label_longtail(c, c, true, 'both', 'resume', out);
```

入口使用现有 `generate_dataset_from_manifest`、参数 schema、`*.partial.mat`
原子提交和 `verify_saved_closedset_split`。纯标签 manifest 的组合计数可以不同，
旧的平衡六类和九挡功率扫描入口不受影响。train/test 的 sample seed 分别使用
0 与 3,000,000 偏移，且不共享样本身份。

## 当前已生成示例

`six_class_pure_label_longtail_v2_IF3_CF8_N8000_AMA` 已完成 train/test，各 8,000
条、63 个组合；实际组合计数为 41--322，标签 IF=3.0、CF≈7.854。两个 MAT
文件均通过 partial→final 回载校验；文件 SHA-256 见同目录运行收据或外部审计记录。
该示例只证明数据与协议完整性，不是模型性能结论。
