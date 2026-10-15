# Route-A P3：双轴长尾损失开发筛查（2026-08-28）

## 1. 实验状态

**completed（开发筛查）；晋级门控未通过，未冻结候选。** 本轮只读取 30 dB `train` 和各功率 `train` 的未用身份作为开发集，`test` 与 `locked test` 均未打开。

目标六类固定为 `FMB、AMA、SawFM、SinFM、MFT、ISRJ`，SN、VFT、RangeFakeTarget 被排除；不将排除类投影为六类负样本。骨干固定为 `F2_DFMS_Full / dfms_iq`，输入 `[B,2,3000]`。

## 2. 协议

- 30 dB train 纯六类池按固定 split-seed=260828 划分：4,096 个拟合候选、2,048 个验证样本；验证身份与拟合身份不交叉。
- 对 4,096 个拟合候选，按 `label_imbalance_factor=5` 和 `combo_imbalance_factor=5` 生成每个 seed 的同一套 mixed-long-tail 清单，并保留 `weak_component_mask`（阈值 10 dB）。实际保留量是清单记录的权威值，不宣称理想 IF。
- 每个清单同时训练三种损失：普通 BCE、平方根逆频率的 class-balanced BCE、类别权重乘以有界弱分量权重（弱分量最多额外 1.5 倍）的 dual-axis BCE。
- seeds：2039、2040、2041；2 epochs；batch=16；AdamW，学习率 `1e-4`，weight decay `1e-4`；梯度范数裁剪 5.0；从 C1 F2 checkpoint 加载共享特征参数，分类头重新训练。
- 各功率开发集为 train-only、排除拟合和验证身份，每个功率最多 256 个样本。没有阈值调节、普通 test 选择或 locked-test 读取。

## 3. 长尾清单审计

| seed | 实际训练样本数 | 说明 |
|---:|---:|---|
| 2039 | 1,331 | 组合先削减、再进行保尾标签保留 |
| 2040 | 1,371 | 同协议、不同确定性随机清单 |
| 2041 | 1,288 | 同协议、不同确定性随机清单 |

清单文件分别为 `longtail_manifest_seed_2039/2040/2041.json`，其中包含组合计数、弱标签计数、样本 SHA256 和源拟合集 SHA256。由于 4,096 候选中组合频率很不均衡，IF=5 的组合削减较强，导致有效训练量约为候选池的 31%–33%；这是本轮结果的重要边界。

## 4. 结果（3 seed 均值）

| 变体 | 参数量 | 验证 Macro-AUPR | 验证 Macro-F1 | 验证组合完全匹配 | 10 dB AUPR | 20 dB | 40 dB | 45 dB | 50 dB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F2+BCE | 257,511 | **0.999983** | **0.996848** | **0.980632** | 0.856291 | **0.897105** | **0.969023** | **0.878352** | 0.827399 |
| F2+class-balanced BCE | 257,511 | 0.999955 | 0.994304 | 0.963704 | 0.856970 | 0.896529 | 0.968458 | 0.878060 | 0.828450 |
| F2+dual-axis BCE | 257,511 | 0.999956 | 0.994562 | 0.965495 | **0.857115** | 0.896276 | 0.967743 | 0.878139 | **0.828674** |

验证组合率的 seed 波动较大：普通 BCE 的单 seed 为 0.9995、0.9844、0.9580；class-balanced 为 0.9995、0.9771、0.9146；dual-axis 为 0.9995、0.9775、0.9194。AUPR 接近饱和，但组合级指标和误报更能区分稳定性。

## 5. 逐类与失败图谱

三种变体都没有出现 VFT 式全标签塌缩，AMA/FMB 的 recall 基本为 1。但 seed=2041 时，ISRJ 的验证 FPR 为：BCE 0.0768、class-balanced 0.1652、dual-axis 0.1555；seed=2040 也出现 ISRJ FPR 约 0.015。class-balanced 和 dual-axis 还使 SinFM 在部分 seed 的 FPR 升高，导致组合完全匹配率下降。也就是说，权重确实提高了弱/尾标签响应，却没有稳定改善困难负样本的正负可分性。

## 6. 门控判定与根因

本轮 **不晋级**。相对于同一长尾清单，普通 BCE 在验证组合率和大多数功率层优于两种加权方案；dual-axis 仅在 10 dB、50 dB AUPR 有极小增益，未达到稳定、跨 seed、跨组合的改善门槛。

根因判断为：

1. 组合 IF=5 使训练样本量骤减，模型主要继承 C1 checkpoint 的已学表示，2 epoch 不足以重新建立尾组合边界；
2. 当前权重按标签频率和弱分量状态增加正/负梯度幅度，但没有提供“同功率、相近背景、只差一个成分”的反事实约束；
3. ISRJ 等标签在困难负样本上形成系统性误报，说明问题是可分性与组合边界，而非简单的正样本响应不足；
4. 因此不能把 AUPR 的微小变化解释为双轴长尾创新成立。

## 7. 工件

- [执行脚本](../../experiments/run_route_a_p3_dual_axis_longtail.py)
- [汇总 JSON](../../outputs/route_a_p3_dual_axis_longtail_20260828/p3_dual_axis_summary.json)
- [角色清单](../../outputs/route_a_p3_dual_axis_longtail_20260828/role_manifest.json)
- [多轴分布审计](../../outputs/route_a_p3_dual_axis_longtail_20260828/multi_axis_profile.json)
- [终端回执](../../outputs/route_a_p3_dual_axis_longtail_20260828/terminal_receipt.json)

## 8. 下一步

进入 P4，保持 F2 和数据角色不变，先做匹配困难负样本的最小实验：同功率/同背景的成分去除反事实对，比较 BCE 与有界 matched-hard-negative loss；先验证 ISRJ、MFT、SawFM 的 FPR 是否下降且 recall 不降，再决定是否进入受约束专家模块。P3 的两种加权损失保留为失败/未晋级证据，不再盲目扩大规模。

