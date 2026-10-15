# P3o-H：finite-valid 组合 CVaR 的不完整执行冻结

## 结论

P3o-H 证明了 P3o-G 的 NaN 实现缺陷已被修复，但没有完成预注册的三种子 formal development，故不构成关于组合 CVaR 的科学证据。1270M/1280M 与当前运行已冻结，不能续跑或补跑。

## 已验证内容

- 独立 finite-valid 单测：inactive NaN 不进入 Smooth-L1；loss、梯度与一次 AdamW step 均 finite；valid target 为 NaN 时显式报错。
- 全量回归：265 passed，9 条既有合成边界 warning。
- 新训练/开发身份各 2,079 个，机制和三功率配对，训练/开发交集与 333 个历史身份文件的最大交集均为 0。
- seed 2071 完成 F/H0/H1，三份预测 payload 已保存；可见 H0/H1 首 epoch checkpoint 均 finite。
- 从未生成、打开或使用 test/locked test。

## 执行失败

seed 2072 的 F-base checkpoint 1--20 和 H0 checkpoint 1--2 已写入，随后 Python 进程退出；没有 `summary.json`、异常收据或 stderr。由于正式 3-seed 证据不完整，不能读取或报告 seed 2071 的数值结果作为方法成果。

## 冻结规则与下一步

不得在 1270M/1280M 上续跑、重跑、重校准、调 loss/threshold/checkpoint，也不得生成 1290M/1320M confirmation。后继 P3o-I 可以保留完全相同的有限有效损失和科学比较，但必须使用全新身份，并把每个 seed 在独立进程中执行、保存独立 result/exception receipt、捕获 stderr，并在 seed 间释放 CUDA 内存。只有三种子均完成后才能读取 development 并判断风险 gate。

## 产物

- [P3o-H 预注册](../configs/p3oh_finite_masked_combo_cvar_preregister_20260802.json)
- [P3o-H 输出冻结](../configs/p3oh_finite_masked_combo_cvar_outcome_freeze_20260802.json)
- [不完整输出目录](../outputs/p3oh_finite_masked_combo_cvar_development_20260802/)
