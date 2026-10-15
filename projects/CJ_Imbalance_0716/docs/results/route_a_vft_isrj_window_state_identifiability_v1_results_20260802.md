# VFT--ISRJ window-state 可识别性屏幕 v1：失败并停止

新鲜 seed `179000000` 的四制度 train-only 数据通过身份、标签、参数和 test-v3 密封审计；主要 hard-negative `VFT=0 AND ISRJ=1` 每制度 320 条。使用预注册的、无任何拟合/校准/阈值选择的相位曲率事件间隔统计，窄宽度与宽宽度的区分结果为：原始相位统计 AUROC `0.46780`，相位置乱控制 AUROC `0.53231`，差值 `-0.06451`；与真窗口宽度的 Spearman 为 `-0.10041`。

三个主要门槛全部失败（AUROC `>=0.65`、相对置乱增益 `>=0.05`、Spearman `>=0.30`）。这否证的是“可由该固定低维 phase-reset 统计从复合 I/Q 恢复窗口状态”，不否证上一轮已证实的生成器宽度干预会增加冻结 Q4 的 VFT FPR。

决定：停止该 phase-reset window-state 特征路线；不得调阈值、事件分段、时频网格或用本数据训练专家。后续若要修复 VFT--ISRJ OOD，必须提出不同且可证伪的高维源分离机制，并在新的身份隔离 development 数据上预注册验证。`locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。

产物：`experiments/run_route_a_vft_isrj_window_state_identifiability_v1.py`、`outputs/tapc/route_a_vft_isrj_window_state_identifiability_v1/summary.json`。
