# P3o-M：分辨率感知稀疏延迟事件 MFT count ordinal 筛查

## 判定

P3o-M **未通过预注册晋级门，停止该稀疏峰 token MFT count 专家路线**。不生成 `1590000000` 或 `1610000000` 确认身份，不调峰规则、token 数、网络宽度、ordinal 损失、训练日程、选择准则，也不将其接入分类或 predicted-label-gated 全 12 参数系统。

该结论不是“稀疏峰完全无信息”：候选在整体与复合样本上相对严格等参数、等 token 数固定分段控制均有三种子稳定收益。但 P3o-M 的预注册规则要求 ID_control 和 low_count 都在三种子达到 Pearson `>=0.30`；low_count 的两颗种子未达标，故不能把该机制表述为可靠专家。

## 研究问题和严格对照

P3c 已证明正确参考下的精确 MFT 分量可恢复 target count；P3d 的稠密反卷积序列 ordinal CNN 则在复合波形中失败，尤其是 ultra-close。P3o-M 不再令网络从 751 点延迟响应隐式计数，而是检验：**显式选择稀疏局部延迟事件是否比等信息量的固定分段读取提供增量信息**。

- `C0_fixed_bin`：将同一 35--110 us 反卷积窗口固定分为 16 个 bin；每 bin 取一个最大 contrast 点；
- `C1_sparse_peak`：在同一四通道窗口内保留正 contrast 的严格局部极大值，按 contrast 取前 16 个，再恢复时间顺序并零填充；
- 两者均使用完全相同的 146,599 参数 `OrdinalTokenNet`、7 个累计 ordinal logits、ordinal BCE + `0.05` 单调性项、AdamW (`lr=3e-4`、`wd=1e-4`)、最多 20 epochs、patience=5、batch=64；
- 仅 MFT-positive 且第 28 列 count 参数 finite/mask 有效的样本参与，**分类损失关闭，结果是 oracle-positive 参数开发证据**。

## 数据与执行可靠性

- 新训练/开发 identity seed：`1570000000` / `1580000000`；各 2,079 个 identity；五个工况、三档功率、每组合 33 身份；
- 预训练审计检查 373 个历史身份文件，训练--开发交叠和与历史最大交叠均为 0；跨功率配对、标签和 parameter mask 合同均通过；
- 15 个训练与 15 个开发视图均已原子提交，未遗留 partial；未生成、读取或使用 test/locked test。

批量三功率 MATLAB 会话在一个工况后出现 HDF5 生命周期中断，且不是空间不足（C/D 当时仍有约 398/50 GB）。随后将生成执行限定为“一个工况 × 一个功率 × 一个 MATLAB 进程”，仅用精确同名 partial resume；所有 P3o-M 正式视图均在该模式下完成。该工程修复不改变波形、种子、样本或方法规则。

首次重分块曾错误使用紧凑的 level 编号，失败于打开源文件之前，仅留下无 MAT 的空目录；其已在 `docs/p3om_rechunk_attempt_freeze_20260803.md` 冻结。正式缓存使用 `_rechunked_v2` 与源目录实际的 `level_04/06/09` 编号。

## 三种子 formal-development 结果

| seed | C1 相对 C0 整体 MAE 降幅 | C1 相对 C0 复合 MAE 降幅 | 整体 Pearson 差 | 复合 Pearson 差 |
|---:|---:|---:|---:|---:|
| 2201 | 16.07% | 15.38% | +0.1162 | +0.1133 |
| 2202 | 14.30% | 13.58% | +0.1141 | +0.1116 |
| 2203 | 16.94% | 16.22% | +0.1234 | +0.1210 |

候选相对固定控制的整体和复合 MAE/Pearson 四项核心对照门均为 3/3 通过；候选对常数基线的 ID_control 与 low_count MAE 降幅也均超过 10%。没有任何工况在 3/3 中相对 C0 的 MAE 退化超过 0.02。

但 low_count 的候选 Pearson 分别为 `0.3057 / 0.2973 / 0.2967`。由于后两颗种子低于预注册 `0.30`，以下门失败：

`each_seed_C1_ID_control_and_low_count_pearson_at_least_0p30=false`。

因此 `development_passed=false`，而非“接近通过”。

## 失败归因与证据边界

1. 稀疏事件选择确实改进了复合 count 估计；它不是单纯容量效应，因为 C0/C1 网络、token 数与参数量完全相同。
2. 低计数场景的 MAE 改善很大，但 1--2 目标的样本内排序在两颗种子仍不稳定；这与 P3d 中的低 count“可区分低/高但类内排序不足”边界一致。
3. ultra-close 仍接近常数基线（候选 MAE 降幅 `2.67% / 0.95% / 3.83%`）。它没有触发相对 C0 的三种子系统性伤害门，但也不能被解释为已解决的可分辨区间。
4. 该轮没有分类或 predicted-gate 结果，因此不得把 oracle-positive 改善外推为端到端 12 参数系统收益。

下一条允许路线必须使用新身份，且机制应显式建模低计数/超近间隔的可分辨性，例如基于可观测峰间距的区间预测或拒识，并与匹配固定分段控制报告 coverage-risk；不得将 P3o-M 的峰规则微调后重跑。

## 工件

- 预注册：[p3om_sparse_peak_ordinal_preregister_20260803.json](../configs/p3om_sparse_peak_ordinal_preregister_20260803.json)
- 方法冻结：[p3om_sparse_peak_ordinal_method_freeze_20260803.json](../configs/p3om_sparse_peak_ordinal_method_freeze_20260803.json)
- 结果冻结：[p3om_sparse_peak_ordinal_outcome_freeze_20260803.json](../configs/p3om_sparse_peak_ordinal_outcome_freeze_20260803.json)
- 身份/合同审计：`outputs/p3om_sparse_peak_ordinal_preflight_20260803/data_audit.json`
- 完整逐 seed 收据与预测：`outputs/p3om_sparse_peak_ordinal_development_20260803/`
