# 六类组合干扰九挡功率扫描使用说明

## 1. 实验用途

本程序生成 FMB、SawFM、SinFM、MFT、VFT、ISRJ 六类基础干扰全部63种非空组合的配对功率扫描数据，用于研究功率不均衡条件下的组合干扰识别和参数提取。

FMB、SawFM、SinFM 始终固定为30 dB；MFT、VFT、ISRJ在每个挡位中同时固定为：

`10、15、20、25、30、35、40、45、50 dB`

每个挡位分别生成训练集和测试集，每个具体组合400条，因此每份MAT为25,200条、每挡共50,400条，九挡合计453,600条。

## 2. 配对种子含义

九个挡位的同一split、同一sample_id共享同一sample_seed。由sample_seed派生的活动分量随机子流、接收机噪声子流和除JNR外的物理参数抽样均相同，只有类型功率发生变化。

这种设计适合做配对性能曲线和敏感性分析。九挡文件不应直接解释为九批完全独立的新样本。train与test仍使用不同split偏移，二者sample_seed没有交集。

## 3. 功率实现和真值

六个物理生成器的公共标称JNR固定为30 dB。manifest按活动类型写入相对偏置：

| 类型 | manifest偏置 | 最终有效JNR |
|---|---:|---:|
| FMB | 0 dB | 30 dB |
| SawFM | 0 dB | 30 dB |
| SinFM | 0 dB | 30 dB |
| MFT | 当前挡位−30 dB | 当前挡位 |
| VFT | 当前挡位−30 dB | 当前挡位 |
| ISRJ | 当前挡位−30 dB | 当前挡位 |

组合层按以下公式完成绝对功率标定：

```text
effective_JNR_dB = nominal_JNR_dB + component_power_offset_dB
target_power_dB = 10*log10(noise_power) + effective_JNR_dB
```

`parameter_values`中各活动类型的`JNR_dB`保存最终有效JNR，而不是公共标称值。`component_effective_JNR_dB`、`component_target_power_dB`和`component_measured_power_dB`提供分量级交叉核对。

`weak_component_mask`标记当前组合内低于最高JNR的活动分量。10–25 dB挡位中，混合组合里的MFT/VFT/ISRJ为弱分量；35–50 dB挡位中，混合组合里的FMB/SawFM/SinFM为弱分量；30 dB挡位没有弱分量。

## 4. 其他物理参数

除功率外全部沿用0716六类闭集配置：

- 采样率：10 MHz；
- 脉宽：100 μs；
- PRI：300 μs；
- 雷达带宽：5 MHz；
- 载频：10 GHz；
- 单条复数I/Q长度：3000；
- 公共到达时延：10–50 μs；
- FMB带宽因子请求范围：1.5–3；
- SawFM扫频带宽：2–5 MHz，周期20–80 μs；
- SinFM中心频率0.5–1.5 MHz，频偏1–3 MHz，调制频率10–20 kHz；
- MFT假目标数3–8，相对时延30–50 μs；
- VFT多普勒1–7 kHz；
- ISRJ采样宽度5–20 μs，转发1–5次。

FMB请求带宽因子超过约2时会受到10 MHz采样率的Nyquist支持限制。程序把实际离散实现带宽因子写入参数真值，不把被裁剪前的请求值伪装为实现值。

## 5. 输出结构

```text
G:\论文相关\干扰数据\复合干扰数据0716\power_sweep_9level_v1\
├─ power_sweep_manifest.csv
├─ 功率不均衡九挡配对数据集说明.md
├─ level_01__jnr_10dB\
│  ├─ closedset_6class_train.mat
│  └─ closedset_6class_test.mat
├─ level_02__jnr_15dB\
...
└─ level_09__jnr_50dB\
```

每个split文件继续保存：

- `iq_complex`：25200×3000 complex double；
- `data_legacy`：25200×6000 double，格式为`[real(IQ), imag(IQ)]`；
- `label_all`：25200×10 logical；
- `label_selected`和`known_label`：25200×6 logical；
- `parameter_values`：25200×79 double；
- `parameter_mask`：25200×79 logical；
- `manifest`、`config`、`parameter_schema`和`metadata`；
- 分量级JNR、目标功率、实测功率、相对功率、弱分量和随机种子元数据。

## 6. 推荐运行命令

先干运行，不写文件：

```matlab
result = generate_closedset_6class_power_sweep(false);
```

正式生成全部九挡：

```matlab
result = generate_closedset_6class_power_sweep( ...
    true, 10:5:50, 'both', 'resume');
```

只处理指定挡位：

```matlab
result = generate_closedset_6class_power_sweep( ...
    true, [20, 25, 30], 'both', 'resume');
```

只处理训练集或测试集：

```matlab
generate_closedset_6class_power_sweep(true, 10:5:50, 'train', 'resume');
generate_closedset_6class_power_sweep(true, 10:5:50, 'test', 'resume');
```

## 7. 已有文件策略

- `resume`：最终文件存在时执行完整增强回载；匹配则复用，不匹配立即报错；
- `error`：只要最终文件存在就拒绝；
- `overwrite`：显式允许用本次结果原子替换最终文件。

程序按挡位串行执行，每挡先train后test。每个split先写`*.partial.mat`，释放大型波形内存并完成增强回载后才移动为正式文件。中断后已完成文件可由`resume`复用；当前未完成split会从头重新生成。

## 8. 自动验证

每个最终文件都会检查：

- 25,200条样本、63个组合、每组合400条；
- 波形、标签和79维参数矩阵尺寸；
- 保存manifest、config、parameter_schema和metadata与期望完全相同；
- 每个活动类型的manifest偏置、有效JNR、目标功率、实测功率和参数JNR真值；
- train/test种子无交集，九挡同split种子完全一致；
- 原子提交完成后不存在该split的partial文件。

## 9. 四类与五类子集兼容性

新增的四类、五类入口复用相同的manifest、功率剖面、原子保存和增强回载
内核，但通过可选`cfg.output.dataset_stem`生成类别化文件名。旧六类配置
不新增该字段；路径解析器在字段缺失时自动回退为`closedset_6class`，所以
原六类入口、旧文件名以及已生成MAT内保存的config结构均保持不变。

四类与五类的类型、目录、种子和公共入口详见：

```text
README_四类五类闭集数据.md
```
