# 路线 A 受控 Label-LT 下一阶段技术与实现方案（2026-07-23）

## 1. 阶段目标

上一轮 IF=2/4 试验中，不同 IF 同时改变了头尾比例、训练集总样本数和尾类绝对正例数。因此，跨 IF 的性能变化不能唯一归因于长尾强度。本阶段只回答一个问题：

> 在训练集总样本数和尾类绝对正例数固定时，DB Loss 相对 ASL 的收益是否随 IF 增大而稳定存在，且 IF=4、SinFM 尾类的失败是否仍然出现？

路线 B 已达到预注册止损条件。本阶段不追加路线 B 训练，只保留既有负结果、实现和审计材料，避免用线性探针提升替代最终识别性能证据。

## 2. 受控数据协议

- 权威数据源：`C:\LJQ_code\干扰数据\复合干扰数据0716`
- 训练环境：`D:\anaconda3\envs\PY10\python.exe`
- 数据协议：四类、optimization role、`label_lt`
- 锁定测试集：全阶段禁止使用
- 固定训练样本数：`N_train = 1600`
- 固定尾类目标正例数：`n_tail = 400`
- IF：`2, 3, 4`
- 尾类轮换：`FMB, SinFM, MFT, ISRJ`
- 标签边际容差：5%
- 总样本数容差：0（必须精确为 1600）

对给定头尾顺序，标签目标为

\[
n_r = \operatorname{round}\left(n_{tail}\,IF^{1-r/(C-1)}\right),
\quad r=0,\ldots,C-1,
\]

其中 `C=4`，最后一位固定为尾类。对应目标为：

| IF | 头类 | 中间类 1 | 中间类 2 | 尾类 |
|---:|---:|---:|---:|---:|
| 2 | 800 | 635 | 504 | 400 |
| 3 | 1200 | 832 | 577 | 400 |
| 4 | 1600 | 1008 | 635 | 400 |

MILP 同时求解组合样本数、标签边际误差和精确总样本数。可行性搜索已覆盖 seeds 2036-2040、3 个 IF 和 4 个尾类；`N=1600, n_tail=400, tolerance=5%` 的 60 个 manifest 条件全部可行。每个正式运行必须保存 manifest 哈希、实际标签计数、实际 IF 和样本数。

## 3. 模型与对照

本阶段不引入新网络模块，避免把数据控制问题与结构变化混在一起。

- 网络：T0 四类基线网络
- 对照损失：ASL
- 候选损失：Distribution-Balanced Loss（DB）
- 训练轮数：20
- patience：20
- batch size：64
- AMP：开启
- 验证样本上限：300
- 模型选择指标：macro-AUPR
- 主阈值模式：`per_label_f1`
- 稳健性阈值模式：`shrinkage_per_label_f1`

第一阶段运行 seeds `2036, 2037, 2038`，共 `3 IF × 4 tail × 2 loss × 3 seed = 72` 次训练。通过阶段门槛后，再扩展 seeds `2039, 2040`，累计 120 次训练。

## 4. 证据与判定门槛

所有差值均采用同一 IF、尾类、seed 下的 `DB - ASL` 配对差。

主证据：

1. 每个 IF 的平均 macro-AUPR 差值为正；
2. 每个 IF 的平均 tail-AUPR 差值为正；
3. `per_label_f1` 下 macro-F1 不下降超过 0.002；
4. `per_label_f1` 下 Hamming loss 不恶化超过 0.002；
5. IF=4、SinFM 尾类的 macro-AUPR 平均差值不低于 -0.005，且至少 2/3 seeds 为正。

阶段结论：

- 全部门槛满足：扩展至 5 seeds，并把 DB 作为路线 A 候选创新组件；
- 总体有益但 SinFM 仍失败：路线 A 降级为条件性方法，下一步做标签语义/共现条件建模；
- IF=2/3/4 均不稳定：停止继续调损失，转向采样或解耦分类器；
- 只有阈值后 F1 改善而 AUPR 不改善：视为校准收益，不能宣称表征或排序能力提升。

## 5. 实现与产物

代码改动：

- `cjlt/data/longtail_manifest.py`：增加精确总样本数 MILP 约束和审计字段；
- `experiments/run_tapc.py`：增加固定尾类锚点、固定总样本数参数；
- `experiments/run_route_a_controlled_lt.py`：冻结并执行本阶段矩阵，生成配对汇总和门槛判定；
- `tests/`：覆盖固定样本预算、固定尾类目标和不可行输入。

正式输出目录：

`outputs/tapc/route_a_controlled_lt/`

核心产物：

- 每次运行的 `longtail_manifest.json`
- `run_receipt.json`
- `metrics_threshold_modes.json`
- `route_a_controlled_lt_summary.json`
- `subprocess_output.txt`

锁定测试集仅在方法、随机种子、阈值策略和统计分析均冻结后，另行授权使用一次。
