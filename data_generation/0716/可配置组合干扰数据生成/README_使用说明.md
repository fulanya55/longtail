# 可配置组合干扰数据生成程序——快速使用说明

程序位置：

```text
G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成
```

本目录是独立的新实现，不覆盖 `Gen_com.m`、`SP_gen_comjam.m`、
`Gen_openjamset.m`、`长尾组合干扰生成` 等历史程序。

## 1. 三个入口程序

| 入口 | 用途 | 默认输出文件 |
|---|---|---|
| `generate_compound_dataset.m` | 普通组合干扰识别与参数提取 | `compound_dataset_balanced.mat` |
| `generate_imbalanced_compound_dataset.m` | 成分、组合、功率及联合不均衡 | `compound_dataset_imbalanced.mat` |
| `generate_openset_compound_dataset.m` | 开放集已知/未知组合干扰 | 4 个 `openset_*.mat` 文件 |

在 MATLAB 中执行：

```matlab
root = 'G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
addpath(root);

cfg = create_default_config();
result = generate_compound_dataset(cfg);
```

入口函数会自动加入 `core` 和 `generators` 子目录，无需手动逐个添加。

## 2. 十类基础干扰

程序固定注册以下十类，输出标签始终按该顺序排列：

```text
1 FMB              阻塞式噪声调频
2 AMA              噪声调幅瞄准
3 SawFM            锯齿波扫频
4 SinFM            正弦调频
5 SN               智能噪声
6 MFT              多假目标
7 VFT              速度假目标
8 ISRJ             间歇采样转发
9 RangeFakeTarget  距离假目标
10 Traction        拖引干扰
```

## 3. 自主选择类型、已知类和未知类

### 3.1 普通或不均衡数据

```matlab
cfg = create_default_config();
cfg.types.enabled = {'FMB','AMA','MFT','ISRJ','Traction'};
cfg.types.known = cfg.types.enabled;
cfg.types.unknown = {};
cfg.dataset.num_samples = 2000;
cfg.dataset.min_components = 1;
cfg.dataset.max_components = 3;
```

### 3.2 开放集数据

已知和未知类别可以直接修改，但两者必须互不重叠，且并集必须等于
`enabled`：

```matlab
cfg = create_default_config();
cfg.types.known = {'FMB','AMA','MFT','ISRJ'};
cfg.types.unknown = {'SawFM','SinFM','SN','VFT', ...
    'RangeFakeTarget','Traction'};
cfg.types.enabled = [cfg.types.known, cfg.types.unknown];

cfg.openset.train_known = 1000;
cfg.openset.calibration_known = 200;
cfg.openset.validation_mixed = 500;
cfg.openset.test_mixed = 1000;

result = generate_openset_compound_dataset(cfg);
```

`mixed_ratios` 的四项依次表示：纯已知、纯未知、已知加一个未知、已知加
多个未知。

## 4. 自主修改参数

所有范围均使用 `[下界, 上界]`：

```matlab
cfg.params.common.JNR_dB = [15, 35];
cfg.params.common.arrival_delay_s = [0, 40e-6];

cfg.params.FMB.bandwidth_factor = [2, 4];
cfg.params.AMA.modulation_index = [0.2, 0.8];
cfg.params.SinFM.center_frequency_Hz = [0.5e6, 1.2e6];
cfg.params.ISRJ.sample_width_s = [5e-6, 15e-6];
cfg.params.ISRJ.num_replicas = [1, 4];
cfg.params.MFT.num_targets = [3, 6];
cfg.params.Traction.velocity_mps = [20, 180];
```

配置校验会拒绝 Nyquist 混叠、假目标时延槽不足、副本不完整、拖引轨迹
越出 PRI 等不可实现参数。

## 5. 不均衡数据配置

### 5.1 成分不均衡

权重与 `cfg.types.enabled` 一一对应：

```matlab
cfg.imbalance.component.enable = true;
cfg.imbalance.component.weights = [10, 6, 3, 1, 1];
```

### 5.2 组合阶数不均衡

```matlab
cfg.imbalance.combination.enable = true;
cfg.imbalance.combination.cardinality_weights = [10, 4, 1];
```

也可使用 `pattern_weights` 对具体组合模式赋权。

### 5.3 功率不均衡

```matlab
cfg.imbalance.power.enable = true;
cfg.imbalance.power.weak_component_policy = 'rotate';
cfg.imbalance.power.relative_power_dB = [-20, -8];
```

`weak_component_policy` 可取：

- `random`：每条样本随机选择弱分量；
- `rotate`：按样本编号轮换弱分量；
- `fixed`：指定固定类型，需同时设置 `fixed_component`。

三个不均衡开关可以单独使用，也可以同时开启：

```matlab
result = generate_imbalanced_compound_dataset(cfg);
```

## 6. 功率字段的关键区别

manifest 中的：

```text
component_target_power_dB
```

表示相对功率偏置。普通/开放集为 0；功率不均衡的弱分量为负值。

最终 bundle 中：

```text
component_power_offset_dB     原始相对偏置
component_target_power_dB     绝对目标平均功率 dB
component_measured_power_dB   波形实测平均功率 dB
relative_power_dB             相对本样本最强分量的最终功率
```

前九类的有效 JNR 为“采样 JNR + 相对偏置”，参数监督保存有效 JNR。

Traction 不另采 JNR，也不做独立绝对功率归一化。其 `amplitude` 是唯一的
全局幅度监督；功率偏置换算成线性增益后并入最终 amplitude 真值。

## 7. 参数真值与 mask

```text
parameter_values   N×79 double
parameter_mask     N×79 logical
```

只有活动类型且可观测、有限的参数位置为 `mask=true`。非活动类型和未使用的
定长槽保存 `NaN/false`。

MFT 和 RangeFakeTarget 的 `delay_XX_s` 保存“目标相对时延 + 公共到达时延”
量化后的最终可观测总时延；公共 `arrival_delay_s` 已被吸收，因此该槽保存
`NaN/false`，避免同一波形对应多组不可辨识标签。

## 8. 输出数据

主要 MAT 顶层字段：

```text
iq_complex
data_legacy                  可选，[real(iq), imag(iq)]
label_all
label_selected
known_label
unknown_present
unknown_component_count
parameter_values
parameter_mask
component_*                 分量功率、JNR、幅度和种子元数据
target_*                    可选真实目标元数据
manifest
metadata
config
registry_snapshot
parameter_schema
```

输出目录：

```matlab
cfg.output_dir = 'D:\your_output_directory';
cfg.output.save_v73 = true;     % 大文件建议开启
cfg.output.save_legacy = true;  % 是否保存实部/虚部拼接格式
```

## 9. 运行测试

```matlab
root = 'G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成';
addpath(root, fullfile(root,'core'), ...
    fullfile(root,'generators'), fullfile(root,'tests'));
results = runtests(fullfile(root,'tests'));
table(results)
```

更完整的设计、旧程序分析、字段定义和科学边界见：

```text
程序说明_组合干扰数据生成系统.md
```
