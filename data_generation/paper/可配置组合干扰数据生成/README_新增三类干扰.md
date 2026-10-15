# 梳状谱、杂乱脉冲与频谱弥散干扰使用说明

本文说明 v2 注册表新增的 `CombSpectrum`、`ClutterPulse` 和
`SpectralDispersion`（标准 SMSP）三类干扰。三类生成器均使用 10 MHz
复基带采样、显式标量 `RandStream`、零边界到达时延和固定参数真值模式。
修改配置不会自动增加数据集样本数；样本量仍由 `cfg.dataset` 或
`cfg.openset` 中的计数字段决定。

## 1. v1与v2注册表

```matlab
cfgV2 = create_default_config();  % 等价于 create_default_config(2)
cfgV1 = create_default_config(1);
```

| 版本 | 类型数 | `label_all`列数 | 参数列数 | 兼容性 |
|---|---:|---:|---:|---|
| v1 | 10 | 10 | 79 | 不含 `cfg.registry`，保持历史结构 |
| v2 | 13 | 13 | 295 | `cfg.registry.version=2`，追加 ID 11--13 |

v2 的稳定顺序为：

```text
1 FMB, 2 AMA, 3 SawFM, 4 SinFM, 5 SN,
6 MFT, 7 VFT, 8 ISRJ, 9 RangeFakeTarget, 10 Traction,
11 CombSpectrum, 12 ClutterPulse, 13 SpectralDispersion
```

缺少 `cfg.registry` 的旧配置由 `build_jamming_registry(cfg)` 按 v1 解释。
`create_closedset_4class_config`、`create_closedset_5class_config`、
`create_closedset_6class_config` 及功率扫描包装器继续固定使用 v1。

## 2. 已知类、未知类和启用类型的修改方法

只生成三个新增类型：

```matlab
cfg = create_default_config(2);
cfg.types.enabled = ...
    {'CombSpectrum','ClutterPulse','SpectralDispersion'};
cfg.types.known = cfg.types.enabled;
cfg.types.unknown = {};
cfg.dataset.min_components = 1;
cfg.dataset.max_components = 3;
```

只启用其中一种时，三个列表应保持完整分区：

```matlab
cfg.types.enabled = {'CombSpectrum'};
cfg.types.known = {'CombSpectrum'};
cfg.types.unknown = {};
cfg.dataset.min_components = 1;
cfg.dataset.max_components = 1;
```

开放集示例，把 SMSP 设为未知类：

```matlab
cfg = create_default_config(2);
cfg.types.enabled = ...
    {'FMB','SawFM','CombSpectrum','ClutterPulse','SpectralDispersion'};
cfg.types.known = {'FMB','SawFM','CombSpectrum','ClutterPulse'};
cfg.types.unknown = {'SpectralDispersion'};

% 只有一个未知类型，不能生成“已知+多个未知”，故第四项必须为0。
cfg.openset.mixed_ratios = [0.25, 0.25, 0.50, 0];
```

`known` 与 `unknown` 必须互不重叠，二者并集必须恰好等于 `enabled`。

## 3. CombSpectrum数学模型与参数表

先把截获参考 LFM 搬移为零中心复基带信号 `s0(t)`。若齿数为 `K`、齿间隔
为 `Delta_f`、梳状谱中心为 `f_c`，第 `k` 个齿的频率为：

```text
f_k = f_c + [k - (K+1)/2] * Delta_f,  k=1,...,K
```

梳状谱干扰为：

```text
j(t) = sum_k a_k s0(t) exp(j[2*pi*f_k*t + phi_k])
```

所有齿共享同一参考 LFM 包络，但具有独立初相。生成器会根据离散参考支撑
自动收缩可行的 `spacing_Hz` 和 `center_offset_Hz`，并保证每个平移后的完整
支撑严格位于 Nyquist guard 内。

| 配置字段 | 单位/取值 | 默认值 | 含义 |
|---|---|---|---|
| `num_teeth` | 整数 | `[3,8]` | 活动齿数 `K` |
| `spacing_Hz` | Hz | `[0.2e6,0.6e6]` | 相邻齿间隔 |
| `center_offset_Hz` | Hz | `[-0.3e6,0.3e6]` | 梳状谱中心偏移 |
| `amplitude` | 无量纲 | `[0.5,1.5]` | random 模式逐齿幅度范围 |
| `amplitude_mode` | 文本 | `'equal'` | `'equal'` 或 `'random'` |
| `max_teeth` | 固定整数 | `8` | 真值容量，不应修改 |

两种幅度模式：

```matlab
cfg.params.CombSpectrum.amplitude_mode = 'equal';  % 所有活动齿幅度为1
cfg.params.CombSpectrum.amplitude_mode = 'random'; % 逐齿独立抽样
cfg.params.CombSpectrum.amplitude = [0.7, 1.3];
```

该类型占 30 个参数列：6 个汇总字段，加 8 个频率、8 个幅度和 8 个相位槽。

## 4. ClutterPulse数学模型、两种模式与32槽语义

杂乱脉冲由随机门控的零中心参考 LFM 片段相干叠加：

```text
j(t) = sum_p a_p g_p(t-t_p) s0(t-t_p)
       * exp(j[2*pi*f_p*(t-t_p) + phi_p])
```

`g_p` 是宽度为 `w_p` 的矩形门。脉冲允许重叠；波形按复数相加，占空比按
所有活动脉冲占用样点的并集计算，因此重叠区只计一次。

| 配置字段 | 单位/取值 | 默认值 | 含义 |
|---|---|---|---|
| `pulses_per_pri` | 整数 | `[3,8]` | 每个 PRI 的脉冲数 |
| `pulse_width_s` | s | `[2e-6,20e-6]` | 单脉冲量化前宽度 |
| `relative_amplitude` | 无量纲 | `[0.5,1.5]` | 单脉冲相对幅度 |
| `frequency_offset_Hz` | Hz | `[-1e6,1e6]` | 单脉冲频偏 |
| `mode` | 文本 | `'independent'` | `'independent'` 或 `'repeated'` |
| `max_pulses_per_pri` | 固定整数 | `8` | 每 PRI 真值容量 |
| `max_total_pulses` | 固定整数 | `32` | 每条记录总真值容量 |

模式含义：

- `independent`：每个 PRI 独立抽样脉冲数、起点、宽度、幅度、频偏和相位；
- `repeated`：只抽样首 PRI 布局，后续 PRI 精确复用脉内参数。

```matlab
cfg.params.ClutterPulse.mode = 'independent';
cfg.params.ClutterPulse.mode = 'repeated';
```

固定 32 槽按“PRI 主序、PRI 内起点升序”排列。每槽保存 `start_XX_s`、
`width_XX_s`、`amplitude_XX`、`frequency_XX_Hz` 和 `phase_XX_rad`；未用槽
为 `NaN/false`。加上 9 个汇总字段，总计 169 个参数列。默认 4 个 PRI 时，
每 PRI 8 槽恰好对应总容量 32；若提高 `radar.num_pulses`，必须同时保证
`num_pulses * pulses_per_pri(2) <= 32`。

## 5. SMSP标准整数子脉冲模型

本实现使用标准 DRFM 频谱弥散模型，不再使用独立连续压缩比。
从整数范围 2--8 抽取子脉冲数 `M`。设原参考脉冲有效样点数为 `N`、脉宽为
`T`、带宽为 `B`、采样率为 `fs`：

```text
N_s = round(N/M)
T_s = N_s/fs
mu_s = B/T_s
chirp_rate_multiplier = T/T_s approximately M
```

一个局部子脉冲为：

```text
q(t) = exp(j*pi*mu_s*(t-T_s/2)^2 + j*phi_0)
```

输出在一个 PRI 内相干重复同一个复子脉冲 `M` 次：

```text
j(t) = sum_(m=0)^(M-1) q(t-m*T_s)
```

每个子脉冲仍扫描原带宽 `B=5 MHz`，不是 `B/M`，也不是 `M*B`；缩短的是
子脉冲时长，增大的是调频斜率。离散量化后总时长为 `M*N_s/fs`，可能与
原始 `T` 存在不超过若干样点的差异。

| 配置字段 | 单位/取值 | 默认值 | 含义 |
|---|---|---|---|
| `num_subpulses` | 整数 | `[2,8]` | 子脉冲数、重复数及理想斜率倍数 |
| `max_subpulses` | 固定整数 | `8` | 真值容量，不应修改 |

固定四子脉冲示例：

```matlab
cfg.params.SpectralDispersion.num_subpulses = [4,4];
```

该类型占 17 个参数列：9 个汇总字段和 8 个 `subpulse_start_XX_s` 槽。

## 6. 10 MHz采样与零中心参考LFM

三类新增干扰都调用 `build_centered_reference_signal`。该函数根据相邻样点
共轭乘积的相位差估计离散瞬时频率支撑，再用支撑实际中点完成搬移：

```text
f[n] = angle(s[n+1] * conj(s[n])) * fs / (2*pi)
f_mid = (min(f) + max(f))/2
s0[n] = s[n] * exp(-j*2*pi*f_mid*n/fs)
```

使用离散实际中点而不是名义 `B/2`，可以补偿有限样点端点误差。搬移只改
相位，不改包络，也不消耗随机数。当前默认 `fs=10 MHz`、参考带宽
`B=5 MHz`；频移后的完整支撑必须严格位于 `(-fs/2, fs/2)` 的 guard 内。

公共 `arrival_delay_s` 先量化为整数样点，再使用前端补零、尾部截断的
非循环时延。任何生成器都不会用 `circshift` 把尾部复制到记录前端。

## 7. 295维参数真值和NaN掩码

v2 参数模式为：

```text
旧十类 79 列
+ CombSpectrum 30 列
+ ClutterPulse 169 列
+ SpectralDispersion 17 列
= 295 列
```

MAT 文件中的主要监督数组：

```text
label_all        N×13 logical
label_selected   N×启用类型数 logical
known_label      N×已知类型数 logical
parameter_values N×295 double
parameter_mask   N×295 logical
```

活动类型的有限可观测字段保存数值并置 `mask=true`。非活动类型、未使用的
固定槽以及不可辨识字段保存 `NaN/false`。新增类型只允许以下固定槽使用
NaN，不允许汇总字段出现 NaN：

- CombSpectrum：频率、幅度、相位的 8 槽；
- ClutterPulse：起点、宽度、幅度、频率、相位的 32 槽；
- SpectralDispersion：8 个子脉冲起点槽。

## 8. 平衡、不平衡和开放集示例

平衡组合：

```matlab
cfg = create_default_config(2);
cfg.types.enabled = ...
    {'CombSpectrum','ClutterPulse','SpectralDispersion'};
cfg.types.known = cfg.types.enabled;
cfg.types.unknown = {};
cfg.dataset.num_samples = 120;
result = generate_compound_dataset(cfg);
```

组合、成分、功率联合不均衡：

```matlab
cfg.imbalance.component.enable = true;
% 可按完整 v2 注册表给13维权重，规范化后自动映射到 enabled 顺序。
cfg.imbalance.component.weights = [zeros(1,10), 1, 2, 3];
cfg.imbalance.combination.enable = true;
cfg.imbalance.combination.cardinality_weights = [3, 2, 1];
cfg.imbalance.power.enable = true;
cfg.imbalance.power.relative_power_dB = [-12, -6];
result = generate_imbalanced_compound_dataset(cfg);
```

开放集：

```matlab
cfg.types.known = {'CombSpectrum','ClutterPulse'};
cfg.types.unknown = {'SpectralDispersion'};
cfg.types.enabled = [cfg.types.known, cfg.types.unknown];
cfg.openset.mixed_ratios = [0.25, 0.25, 0.50, 0];
result = generate_openset_compound_dataset(cfg);
```

上述示例只改变配置；若要正式生成大规模数据，还需自行设置输出目录、样本数
和文件策略。增加注册表类型不会自动增加 `num_samples`。

## 9. 常见配置错误及稳定错误标识

| 错误标识 | 常见原因 |
|---|---|
| `JammingConfig:InvalidRegistryVersion` | 配置版本不是数值 1 或 2 |
| `JammingConfig:InvalidIntegerRange` | 齿数、脉冲数、子脉冲数或固定容量非法 |
| `JammingConfig:InvalidPhysicalParameter` | Nyquist 支撑、PRI 时长或固定槽容量不可实现 |
| `JammingConfig:UnsupportedMode` | 配置层收到不支持的幅度/布局模式 |
| `JammingConfig:InvalidImbalanceWeights` | 成分权重长度、符号或正权重数量不合法 |
| `JammingOpenSet:InvalidComposition` | 当前 known/unknown 数量与 mixed 比例无法构造 |
| `JammingGenerator:InvalidStream` | 未传入标量 `RandStream` |
| `JammingGenerator:InvalidPhysicalParameter` | 生成时抽样结果越出物理或记录边界 |
| `JammingGenerator:UnsupportedMode` | 生成器收到非法模式文本 |

建议先执行：

```matlab
cfg = normalize_and_validate_config(cfg, 'balanced');  % 或 imbalanced/openset
registry = build_jamming_registry(cfg);
schema = build_parameter_schema(registry);
```

确认 v2 时 `numel(registry)==13` 且 `height(schema)==295` 后，再启动正式仿真。
