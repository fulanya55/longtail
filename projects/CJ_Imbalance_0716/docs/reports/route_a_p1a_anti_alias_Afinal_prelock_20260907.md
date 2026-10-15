# Route A P1a：A-final-prelock 开发候选（2026-09-07）

## 决策

在新六类任务（FMB、AMA、SawFM、SinFM、MFT、ISRJ）和固定 F2 输入/分类器/训练协议下，`F2_anti_alias` 是当前最强的可复现开发候选。它只被登记为 `A-final-prelock`，尚未进行 locked test，也不构成最终泛化结论。

## 证据边界

- 两个独立数据划分（split seed 260827、260907），每个划分 5 个训练种子（2039--2043），均为 anti-alias 与 F2_full 的配对比较。
- 合并 10 个配对观测：30 dB 组合准确率差 `+0.00683594`（约 `+0.684` 个百分点），Macro-F1 差 `+0.001223`，Macro-AUPR 差 `+0.00005558`。
- 合并 Wilcoxon 数值为探索性 post-hoc 汇总（组合准确率 `p=0.00763`、Macro-F1 `p=0.00391`、Macro-AUPR `p=0.00586`）；不能替代预注册的确认性检验，单划分 p 值仍处于边界区间（约 0.0625--0.125）。
- 功率维度的优势主要集中在 25/40 dB 组合准确率和 40--50 dB AUPR；15--20 dB 未显示稳定收益，低功率可观测性仍是未解决问题。
- 参数量两者均为 257,511；当前计时差异噪声较大，不能宣称确定的速度开销。

## 冻结内容

`frozen/route_a_anti_alias_Afinal_prelock_20260907/manifest.json` 固定了候选结构、六类标签、`F2_DFMS_Full/dfms_iq` 输入契约、阈值、源 checkpoint 哈希及 `locked_test_used=false`。必要消融为同协议的 `F2_full`；两者均不得再调参。为避免测试后挑选 seed，预注册将十个候选 checkpoint 的 logits 做等权平均，并与对应的十个 F2_full checkpoint 配对；同时保留每个 checkpoint 的分布用于审计。

## 下一步门控

只有获得明确授权后，才可按 manifest 一次性读取 locked test。授权后不得修改模型、阈值、数据划分或报告口径；若未授权，继续开展的工作只能是开发集上的预注册补充或失败原因分析。

## 收据

- `outputs/route_a_p1a_anti_alias_confirm_20260906/p1a_anti_alias_confirm_statistics.json`
- `outputs/route_a_p1a_anti_alias_independent_20260907/p1a_anti_alias_independent_statistics.json`
- `outputs/route_a_p1a_anti_alias_Afinal_prelock_20260907/combined_replication_statistics.json`
- `frozen/route_a_anti_alias_Afinal_prelock_20260907/manifest.json`
- `frozen/route_a_anti_alias_Afinal_prelock_20260907/locked_test_preregistration.json`
- `outputs/route_a_p1a_anti_alias_Afinal_prelock_20260907/prelock_audit_receipt.json`

## 预注册状态

locked-test 的一次性读取协议已经预注册，包含固定候选、F2_full 消融、阈值、指标、原始结果优先输出和读后禁止修改约束。当前 `read_count=0`、`test_files_opened=false`；在用户明确授权前不得打开 sealed manifest 对应的测试文件。
## 数据角色复核补充（2026-09-08）

路径审计发现，历史 C 盘 Route-A v3 测试源不属于本次 E 盘九类分支；九类 `power_sweep_9level_v1` 的 test 文件已被 C1 基线作为开发评估读取，不能再次声明为独立 locked test；`full_power_10_30dB_v1` 只覆盖 10--30 dB，不能支撑完整 10--50 dB 结论。因此预注册暂时降级为“等待绑定新的 E 盘身份不相交 sealed source”，详见 `outputs/route_a_p1a_anti_alias_Afinal_prelock_20260907/locked_test_data_role_audit.json`。在完成绑定前，即使得到授权，也不能读取上述不合格文件。
