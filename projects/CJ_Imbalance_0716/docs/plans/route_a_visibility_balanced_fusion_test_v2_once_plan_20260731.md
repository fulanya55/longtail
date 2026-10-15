# Route A：可见性分层均衡融合 test-v2 一次性验证计划

## 冻结候选

development-v2 已完成全部选择并通过全部门槛。冻结对象包括五个种子的：

- F2_DFMS_Full checkpoint；
- C6_PosBCE checkpoint 与 global feature mean/std；
- T_FinalTemporal VTSE checkpoint 与 feature mean/std；
- 可见性分层均衡融合器、expert score center/scale/orientation；
- development calibration temperature 与五标签 thresholds；
- 网络参数、数据配置、计划、执行脚本和 SHA-256 manifest。

冻结后禁止修改、重新训练、重新校准或重新选择阈值。

## test-v2 一次性规则

- 数据：`closedset_6class_test.mat`，预封存 SHA-256
  `902e636911e76d6915fe04976f59cc46a588b3d98926bb9a48e2200628617947`。
- 执行器在读取 test-v2 前创建不可覆盖的 started guard。
- 若 started/completed guard 已存在，拒绝再次运行。
- 即使进程失败或中断，也禁止重跑。
- test-v2 不训练、不校准、不选阈值；仅使用冻结参数前向推理。
- 同时评估五个预注册种子，报告配对均值和置信区间。

## 独立复现门

候选相对冻结 C6：

1. 五标签 macro-AUPR 差值 >= -0.005；
2. macro-F1 差值 > 0；
3. MFT/VFT/ISRJ 平均弱召回差值 > 0；
4. 目标弱正—负 AUROC 差值 > 0；
5. 目标平均 FPR <= 0.20；
6. 强正召回差值 >= -0.010；
7. 至少 4/5 种子弱召回为正增益；
8. SawFM-present 五标签 macro-F1 差值 >= -0.005。

test-v2 仅验证方向复现，不允许因未达到 development 效应量而调参。

