# P8a：三轴长尾数据协议图谱（train-only 预备阶段）

## 结论

P8a **通过数据协议有效性门**，但它不是模型训练或长尾方法结果。它首次冻结了可复现的六类 VFT 目标三轴长尾合同：标签频次尾部、组合频次尾部和低可见度功率尾部。此结论只能证明后续长尾实验可在不把九个功率视图视作独立样本的前提下开展；不能表述为任何网络、损失或路由改善了长尾性能。

## 冻结边界

- 预注册：`configs/p8a_multi_axis_longtail_atlas_preregister_20260802.json`
- 方法冻结：`configs/p8a_multi_axis_longtail_atlas_method_freeze_20260802.json`
- 实现：`cj0724/p8a_multi_axis_longtail_atlas.py`
- 单元测试：`tests/test_p8a_multi_axis_longtail_atlas.py`，结果 `2 passed`。
- P8a 仅读取 `C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1` 的九个 `train_path` 文件中身份和标签元数据；没有读取 `closedset_6class_test`、任何 checkpoint、预测或历史 test。
- 九档功率的身份顺序和标签对齐审计哈希为 `6568ea1c9a1b5650d039fb47c0fa12d6d5e8f0c5fb8aa5622b09c0adf6a775e4`。
- 按 63 个组合划分的身份角色哈希为 `a334e09a9127a0b2259b4eeec35e0b53ad53fc9fd0fa756243420ae482660f34`；每组合的 `optimization/early/calibration/development` 身份严格隔离。
- 所有 P8a 选中身份仅属于协议预备角色，后续 P8b 或确认实验不得用它们训练、选模、校准、评价或确认。

## 三份哈希 manifest

| Manifest | 六类标签计数 `[FMB,SawFM,SinFM,MFT,VFT,ISRJ]` | 组合计数范围 | 可见度功率分配 |
|---|---:|---:|---|
| balanced | `[1217,1217,1217,1217,1217,1217]` | 38–39 | 30 dB：2400 |
| combo-LT | `[1200,1200,1200,1200,1200,1200]` | 8–67 | 30 dB：2400 |
| mixed-LT | `[1200,1200,1200,1200,400,1200]` | 1–280 | 10/15/20/30 dB：127/133/116/2024 |

组合尾部以非空偶数基数组合为低频组。`combo-LT` 的偶数/奇数组合平均计数为 `9.935/65.375`，同时保持六类边际都为 1200。`mixed-LT` 在此组合尾部之上，将 VFT 正例收缩至 400；其中仅 `VFT=1` 且至少有一个固定分量共存的 376 个身份被确定性分配到 10/15/20 dB，其余身份均使用 30 dB。该定义避免把纯 VFT 的全局低功率误当作“相对固定分量下的弱可见度”。

三个 manifest 的 SHA-256 分别为：

- balanced：`0b5252df41572150210e91f8a8e0cb64b7b4f91eccc23d1b706f3ff57b0227e8`
- combo-LT：`93a244748275d82f700d5889ff0773cb90b2737d535a7ed3add0abd4ea33b42f`
- mixed-LT：`e221a38b38f4eef40b4a5b2b102a7829ebb0884474a6fd718b368ffec021d4b1`

结果文件：`outputs/p8a_multi_axis_longtail_atlas_20260802/result.json`。

## 有效性门

balanced、combo-LT 和 mixed-LT 均通过：2400 个不重复身份、63 个组合均保留、精确标签边际、功率值仅取 `{10,15,20,30}`。mixed-LT 还通过“所有且仅有 VFT 阳性并与固定分量共存的行被分配为弱功率”的审计。

`test_generated=false`、`test_opened=false`、`locked_test_used=false`。

## 下一轮约束

P8b 必须使用全新、与 P8a 不重叠的生成身份，并在读取任何 P8b 开发结果前冻结一个单一的长尾机制。至少应包含等计算 BCE 强基线、相同优化步数、三模型种子、逐类 AUPR/Recall/FPR、VFT 困难负例分离度、组合/功率最坏切片，以及禁止以 Recall 增加掩盖 AUPR 或 FPR 恶化的门。P5d 的 LSCO 失败不得作为阈值、关系损失或组合原型的事后救火依据。
