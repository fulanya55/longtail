# 四类与五类闭集组合干扰数据生成说明

## 1. 数据目标

本程序在已经验证的六类闭集程序上增加“类型子集”能力，正式生成两组
闭集组合干扰数据。每组均包含一套10–30 dB连续全功率覆盖数据，以及一
套10–50 dB九挡配对功率扫描数据。

四类类型：

```text
FMB、SawFM、MFT、ISRJ
```

五类类型：

```text
FMB、SinFM、MFT、VFT、ISRJ
```

全部启用类型同时属于已知类，未知类为空，因此训练集和测试集均为严格
闭集。程序不修改或覆盖现有六类数据。

## 2. 数据规模

| 数据 | 非空组合数 | 每组合 | 每个split | train+test | 九挡总量 |
|---|---:|---:|---:|---:|---:|
| 四类 | 15 | 400 | 6,000 | 12,000 | 108,000 |
| 五类 | 31 | 400 | 12,400 | 24,800 | 223,200 |

四类和五类的全功率、九挡合计：

```text
40个MAT文件，368,000条记录
```

## 3. 输出目录

```text
G:\论文相关\干扰数据\复合干扰数据0716\
├─ 4类\
│  ├─ full_power_10_30dB_v1\
│  │  ├─ closedset_4class_train.mat
│  │  └─ closedset_4class_test.mat
│  └─ power_sweep_9level_v1\
│     ├─ power_sweep_manifest.csv
│     ├─ level_01__jnr_10dB\
│     │  ├─ closedset_4class_train.mat
│     │  └─ closedset_4class_test.mat
│     ├─ ...
│     └─ level_09__jnr_50dB\
└─ 5类\
   ├─ full_power_10_30dB_v1\
   │  ├─ closedset_5class_train.mat
   │  └─ closedset_5class_test.mat
   └─ power_sweep_9level_v1\
      ├─ power_sweep_manifest.csv
      ├─ level_01__jnr_10dB\
      │  ├─ closedset_5class_train.mat
      │  └─ closedset_5class_test.mat
      ├─ ...
      └─ level_09__jnr_50dB\
```

## 4. 两种功率语义

### 4.1 全功率覆盖

`full_power_10_30dB_v1` 中，每个活动干扰分量独立从10–30 dB连续均匀
抽取有效JNR。一个组合内的多个分量不要求具有相同功率。这套数据与九挡
数据使用独立种子域，不应解释为逐样本配对数据。

回载验证会逐活动分量核对：

- 有效JNR位于10–30 dB；
- 绝对目标功率等于背景噪声功率dB加有效JNR；
- 实测干净分量功率位于允许容差内；
- `parameter_values`中对应类型的`JNR_dB`与分量真值一致；
- 对应参数掩码为有效。

### 4.2 四类九挡

固定30 dB：

```text
FMB、SawFM
```

同步扫描10、15、20、25、30、35、40、45、50 dB：

```text
MFT、ISRJ
```

### 4.3 五类九挡

固定30 dB：

```text
FMB、SinFM
```

同步扫描10、15、20、25、30、35、40、45、50 dB：

```text
MFT、VFT、ISRJ
```

九挡使用同一基础种子。同一类别数、同一split、同一sample_id在九挡中
具有相同组合、分量随机子流、接收机噪声子流和非JNR参数；只改变功率
剖面。train和test种子无交集。

## 5. 四个独立种子域

| 实验 | 基础种子 |
|---|---:|
| 四类全功率 | 100260722 |
| 四类九挡 | 200260722 |
| 五类全功率 | 300260722 |
| 五类九挡 | 400260722 |

总入口会重新构建四个代表配置的train和test manifest，对8个完整
`sample_seed`向量执行两两交集检查。任一交集都会停止执行。

## 6. 雷达与类型参数

雷达体制和单类参数继承六类正式程序：

| 参数 | 设置 |
|---|---|
| 采样率 | 10 MHz |
| 脉宽 | 100 μs |
| PRI | 300 μs |
| 雷达带宽 | 5 MHz |
| 载频 | 10 GHz |
| 单条I/Q长度 | 3000 |
| 背景噪声功率 | 1 |
| 真实目标回波 | 不叠加 |
| 公共到达时延 | 10–50 μs |

类型参数：

- FMB：请求带宽因子1.5–3，中心频率0；
- SawFM：扫频带宽2–5 MHz，周期20–80 μs；
- SinFM：中心频率0.5–1.5 MHz，频偏1–3 MHz，调制频率10–20 kHz；
- MFT：假目标数3–8，相对时延30–50 μs，相对幅度0.5–1.5；
- VFT：多普勒频移1–7 kHz；
- ISRJ：采样宽度5–20 μs，直接转发1–5次。

FMB请求带宽因子超过当前10 MHz采样率可支持范围时，生成器会按Nyquist
支持裁剪离散实现，并把实际带宽和实际带宽因子写入参数真值。请求值不能
直接当作最终实现带宽。

## 7. 正式入口

### 7.1 只处理四类

```matlab
result = generate_closedset_4class_datasets( ...
    true, 'all', 10:5:50, 'both', 'resume');
```

### 7.2 只处理五类

```matlab
result = generate_closedset_5class_datasets( ...
    true, 'all', 10:5:50, 'both', 'resume');
```

### 7.3 四类、五类全部串行生成

```matlab
result = generate_closedset_4class_5class_all( ...
    true, 'all', 10:5:50, 'both', 'resume');
```

总入口的固定顺序为：

1. 四类全功率train、test；
2. 四类九挡，每挡train、test；
3. 五类全功率train、test；
4. 五类九挡，每挡train、test；
5. 四个实验的8个split种子域审计。

正式生成只允许启动一个MATLAB进程。

## 8. dry-run与选择参数

不写文件的全规模预检：

```matlab
result = generate_closedset_4class_5class_all( ...
    false, 'all', 10:5:50, 'both', 'resume');
```

`datasetSelection`允许：

- `all`：全功率和九挡；
- `full_power`：只处理全功率；
- `power_sweep`：只处理九挡。

`splitSelection`允许`both`、`train`、`test`。`levels_dB`允许
`10:5:50`中的非空无重复子集。

## 9. 已有文件策略与原子保存

- `resume`：完整回载匹配的final并复用；语义不匹配立即报错；
- `error`：只要final存在就拒绝；
- `overwrite`：显式允许用本次新结果替换final。

每个split先写精确同名的`*.partial.mat`。保存结束后释放大型内存数据，
验证partial，通过才移动为final，然后再次验证正式路径。中断最多留下
partial，不会把未完成文件伪装成final。

## 10. MAT存储字段

主要顶层变量：

- `iq_complex`：复数I/Q；
- `data_legacy`：实部与虚部拼接；
- `label_all`：该四/五类包装器固定使用 v1 十类注册表标签；
- `label_selected`：四类文件4列，五类文件5列；
- `known_label`：四类文件4列，五类文件5列；
- `parameter_values`与`parameter_mask`：参数提取监督；
- `manifest`：组合、sample_id、split、种子和功率偏置；
- `config`、`parameter_schema`、`metadata`；
- 分量有效JNR、绝对目标功率、实测功率等cell真值。

`label_all`和全量参数模式保持 v1 十类注册表稳定格式，从而继续兼容已有六类
和其他场景的数据读取代码。

## 11. 最终验收

正式完成后必须满足：

- 四类20个MAT、五类20个MAT，共40个；
- 0个`*.partial.mat`；
- 四类每个split 6000条、15个组合、每组合400条；
- 五类每个split 12400条、31个组合、每组合400条；
- 40个文件增强回载通过；
- 两组九挡分别完成配对字段全等核对；
- 8个split种子域两两无交集；
- 生成四类20行、五类20行和总计40行SHA-256清单；
- 保存四类、五类和总说明三个中文文件。
