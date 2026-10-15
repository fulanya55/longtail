# 六类组合干扰九挡功率扫描设计规格

## 1. 目标

在现有六类闭集组合数据基础上生成 `power_sweep_9level_v1` 数据集。六类顺序固定为：

`FMB, SawFM, SinFM, MFT, VFT, ISRJ`

FMB、SawFM、SinFM 的 JNR 始终固定为 30 dB；MFT、VFT、ISRJ 在每个实验挡位中同时固定为一个 JNR，挡位为：

`10, 15, 20, 25, 30, 35, 40, 45, 50 dB`

每个挡位分别生成训练集和测试集。每个划分覆盖六类的 63 个非空组合，每个组合 400 条，共 25,200 条。

## 2. 配对种子设计

九个挡位复用同一套 train/test manifest 顺序和 `sample_seed`。同一 split、同一 `sample_id` 在九个挡位中具有相同的：

- 活动干扰组合；
- 各分量随机子流种子；
- 接收机噪声种子；
- 除 JNR 外的物理参数随机抽样。

挡位之间仅改变指定三类的有效 JNR。train 与 test 继续使用现有固定 split 偏移，因此二者种子集合互不相交。该设计用于配对比较功率变化对识别和参数提取的影响，不应把九挡数据简单视为相互独立的新样本。

## 3. 功率实现

所有六类生成器的公共标称 JNR 固定为 `[30, 30]`。manifest 对活动分量写入相对功率偏置：

- FMB、SawFM、SinFM：`0 dB`；
- MFT、VFT、ISRJ：`sweep_JNR_dB - 30 dB`。

组合层已经实现：

`effective_JNR_dB = nominal_JNR_dB + component_power_offset_dB`

因此最终参数真值、目标功率和实测功率均以有效 JNR 为准，无需修改六个物理生成器。`weak_component_mask` 按每条组合内的相对功率计算：低于该组合最高 JNR 的活动分量标为弱分量；同功率组合全部为 false。现有 `imbalance_flags` 保持 false，因为本实验不改变组合或成分采样分布，而是一个确定性类型功率剖面。

## 4. 输出布局

输出根目录：

`G:\论文相关\干扰数据\复合干扰数据0716\power_sweep_9level_v1`

每挡目录：

```text
level_01__jnr_10dB
level_02__jnr_15dB
...
level_09__jnr_50dB
```

每个目录包含：

- `closedset_6class_train.mat`
- `closedset_6class_test.mat`

根目录还包含：

- `功率不均衡九挡配对数据集说明.md`
- `power_sweep_manifest.csv`

旧的 `closedset_6class_train.mat` 和 `closedset_6class_test.mat` 保持原位置，不覆盖。

## 5. 运行接口

新增公共入口：

```matlab
result = generate_closedset_6class_power_sweep( ...
    executeGeneration, levels_dB, splitSelection, existingPolicy)
```

默认参数：

- `executeGeneration=true`
- `levels_dB=10:5:50`
- `splitSelection='both'`
- `existingPolicy='resume'`

`resume` 对每挡每个 split 单独做增强回载；已验证 final 直接复用，损坏或配置不匹配立即报错。新文件仍先写精确同名 `.partial.mat`，验证后再原子提交。

## 6. 数据契约与验证

每份 MAT 继续保存原有顶层变量，包括 `iq_complex`、`data_legacy`、标签、参数真值/掩码、manifest、配置和分量功率元数据。

增强回载除原有尺寸和 manifest/config/schema 全等核对外，还检查：

- 每个活动代码的有效 JNR 是否等于 30 dB 或当前挡位；
- manifest 偏置是否等于代码对应的期望偏置；
- 目标功率减去噪声功率是否等于有效 JNR；
- 实测功率是否在 `cfg.output.power_tolerance_dB` 内匹配目标功率；
- 参数矩阵中六类 `JNR_dB` 真值是否与有效 JNR 一致；
- 九挡 train 种子完全相同、九挡 test 种子完全相同、train/test 种子无交集；
- 每份文件 25,200 条、63 个组合且每组合 400 条。

## 7. 容量和执行约束

预计新增 18 个 MAT 文件，逻辑容量约 42.73 GiB。串行执行顺序为每挡 train 后 test，再进入下一挡；禁止并发生成同一挡位。中断时最多留下当前 split 的 partial，已完成挡位可通过 `resume` 复用。
