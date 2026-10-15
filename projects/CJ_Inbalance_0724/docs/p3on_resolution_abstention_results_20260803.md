# P3o-N：分辨率感知拒识的冻结开发结果

## 结论

P3o-N 未通过预注册的三种子开发门槛，结论为 `stop_no_confirmation_data`。不得在本轮身份上调整分数公式、间距尺度、覆盖率、token、网络、损失或训练过程；不得生成确认身份、接回系统或使用 locked test。

## 证据边界

- 仅 MFT oracle-positive 参数估计；分类损失关闭。
- 新训练/开发 receiver-noise seed 分别为 1720000000 / 1730000000；各 2,079 个身份。
- 身份审计：跨角色重叠 0；与 383 份历史文件的最大重叠 0；三功率配对与 MFT 参数契约通过。
- C0 和 C1 共用同一个每种子点估计器、checkpoint 和逐样本预测（收据中的 prediction SHA-256 相同）。它们仅以 selection 身份固定的 80% 覆盖阈值改变接受集。
- 不生成/不读取测试集，`locked_test_used=false`。

## 结果

| seed | C1 相对 C0 总体 MAE 改善 | 复合 MAE 改善 | 总体 Pearson 变化 | low-count C1 Pearson | ultra-close 覆盖变化 | ultra-close MAE 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2301 | -8.83% | -8.90% | -0.0594 | 0.2850 | -0.0280 | +0.0666 |
| 2302 | -8.47% | -8.61% | -0.0486 | 0.3076 | -0.0415 | +0.0907 |
| 2303 | -8.57% | -8.60% | -0.0516 | 0.2951 | -0.0359 | +0.0811 |

所有方法的开发总体覆盖率均在预注册的 0.70--0.90 区间，且每种子 selection 覆盖率恰为 0.80；因此失败不是覆盖率不匹配造成的。

## 失败机制

C1 的局部峰强度与最小间距并非“点估计已正确时的可靠性”代理：它在 25 dB 样本上接受更多困难例、在 50 dB 样本上接受更少样本，从而把覆盖率由功率可见性而非真实计数风险主导。虽有较少 ultra-close 样本被接受的趋势，但所留下的样本 MAE 反而增大，说明该可观测分辨率分数无法在固定覆盖下区分可安全估计与不可安全估计样本。

## 实现与生成器修复记录

MATLAB 长生命周期的多视图 HDF5 写入此前可能在后续事务发生访问冲突。本轮以一个 `角色×制度×功率` 视图对应一个 MATLAB 进程生成 30/30 个视图，全部原子提交、无 `.partial.mat` 遗留。两次 `evalc` 静默包装报输出参数错误，但都发生在目标 MAT 已提交之后；后续已移除该包装，未影响数据内容。

关键工件：

- `outputs/p3on_resolution_abstention_preflight_20260803/data_audit.json`
- `outputs/p3on_resolution_abstention_development_20260803_contractfix/summary.json`
- `configs/p3on_resolution_abstention_method_freeze_20260803.json`
- `configs/p3on_resolution_abstention_outcome_freeze_20260803.json`
