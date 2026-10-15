# 四类与五类闭集组合干扰数据集设计

日期：2026-07-22
状态：用户已完成书面复核，进入实施计划

## 1. 目标

在现有六类闭集与 `power_sweep_9level_v1` 程序基础上，增加一个可复用的“类型子集”生成能力，并正式生成四类、五类两组闭集组合干扰数据。每组都包含：

1. 一套分量 JNR 在 10–30 dB 内独立连续随机抽取的全功率覆盖数据；
2. 一套 10–50 dB、步长 5 dB 的九挡配对功率扫描数据。

不得覆盖现有六类数据。正式输出直接放在：

```text
G:\论文相关\干扰数据\复合干扰数据0716\
├─ 4类\
└─ 5类\
```

## 2. 类型选择

### 2.1 四类实验

启用并全部设为已知类：

```text
FMB、SawFM、MFT、ISRJ
```

组合模式为 `all`，覆盖一至四阶全部非空组合，共 `2^4-1=15` 种。

### 2.2 五类实验

启用并全部设为已知类：

```text
FMB、SinFM、MFT、VFT、ISRJ
```

组合模式为 `all`，覆盖一至五阶全部非空组合，共 `2^5-1=31` 种。

两组实验均为严格闭集，`unknown={}`。

## 3. 输出目录

```text
G:\论文相关\干扰数据\复合干扰数据0716\
├─ 4类\
│  ├─ full_power_10_30dB_v1\
│  │  ├─ closedset_4class_train.mat
│  │  └─ closedset_4class_test.mat
│  ├─ power_sweep_9level_v1\
│  │  ├─ power_sweep_manifest.csv
│  │  ├─ level_01__jnr_10dB\
│  │  ├─ ...
│  │  └─ level_09__jnr_50dB\
│  ├─ 四类组合干扰数据集说明.md
│  └─ 四类文件SHA256清单.csv
├─ 5类\
│  ├─ full_power_10_30dB_v1\
│  │  ├─ closedset_5class_train.mat
│  │  └─ closedset_5class_test.mat
│  ├─ power_sweep_9level_v1\
│  │  ├─ power_sweep_manifest.csv
│  │  ├─ level_01__jnr_10dB\
│  │  ├─ ...
│  │  └─ level_09__jnr_50dB\
│  ├─ 五类组合干扰数据集说明.md
│  └─ 五类文件SHA256清单.csv
├─ 四类五类数据集总说明.md
└─ 四类五类文件SHA256总清单.csv
```

每个九挡目录中的单挡均保存 `train` 和 `test` 两个 MAT 文件，命名分别使用对应类别数：

```text
closedset_4class_train.mat
closedset_4class_test.mat
closedset_5class_train.mat
closedset_5class_test.mat
```

## 4. 样本规模

每种具体组合生成 400 条样本，train 和 test 分别独立生成。

| 数据 | 组合数 | 每个 split | MAT 数 | 总记录数 |
|---|---:|---:|---:|---:|
| 四类全功率 | 15 | 6,000 | 2 | 12,000 |
| 四类九挡 | 15 | 6,000 | 18 | 108,000 |
| 五类全功率 | 31 | 12,400 | 2 | 24,800 |
| 五类九挡 | 31 | 12,400 | 18 | 223,200 |
| 合计 | — | — | 40 | 368,000 |

按现有六类正式文件的字节规模线性估计，新增数据约 35 GiB。正式生成前必须重新检查可用空间，并为当前 `.partial.mat` 额外预留至少一个单文件容量。

## 5. 功率设置

### 5.1 全功率覆盖数据

`full_power_10_30dB_v1` 中，每个活动干扰分量的有效 JNR 在 `[10,30] dB` 内独立连续均匀抽取。不同活动分量不要求相同 JNR，也不使用九挡相对偏置。

公共配置：

```matlab
cfg.params.common.JNR_dB = [10, 30];
```

最终有效 JNR、目标功率、实测功率及参数掩码必须写入现有真值字段。
全功率文件的增强回载不能只检查尺寸：还必须确认每个活动分量的有效
JNR 位于 10–30 dB、目标功率与有效 JNR 一致、实测功率位于允许容差
内，并且 `parameter_values` 中对应类型的 `JNR_dB` 与分量真值一致。

### 5.2 四类九挡扫描

固定 30 dB：

```text
FMB、SawFM
```

同步扫描 `10:5:50 dB`：

```text
MFT、ISRJ
```

### 5.3 五类九挡扫描

固定 30 dB：

```text
FMB、SinFM
```

同步扫描 `10:5:50 dB`：

```text
MFT、VFT、ISRJ
```

九挡实现继续采用已经验证的方式：所有物理生成器的公共标称 JNR 固定为 30 dB，再由 manifest 为扫描组写入 `当前挡位-30 dB` 的确定性相对偏置。

## 6. 雷达和类型参数

全部雷达参数和干扰类型参数继承已经验收的六类配置：

- 采样率：10 MHz；
- 脉宽：100 μs；
- PRI：300 μs；
- 雷达带宽：5 MHz；
- 载频：10 GHz；
- 单条复数 I/Q 长度：3000；
- 背景噪声功率：1；
- 不叠加真实目标；
- 公共到达时延：10–50 μs；
- FMB 带宽因子请求范围：1.5–3，并保存 Nyquist 裁剪后的实际真值；
- SawFM、SinFM、MFT、VFT、ISRJ 的参数范围与六类正式程序一致。

## 7. 通用程序结构

采用通用子集配置器，不复制物理生成器或数据保存核心。

### 7.1 配置层

新增通用闭集子集配置函数，输入至少包括：

- `selected_types`；
- `class_tag`；
- `seed`；
- `output_dir`；
- `samples_per_pattern`。

配置中新增 `cfg.output.dataset_stem`，四类取 `closedset_4class`，五类取
`closedset_5class`。`run_closedset_splits` 和 `run_power_sweep_levels` 必须
改为根据该字段生成 final/partial 文件名；字段缺失时回退到
`closedset_6class`，保证现有六类入口及已经生成的数据完全兼容。

新增通用功率扫描配置函数，在闭集子集配置上增加：

- `fixed_codes`；
- `swept_codes`；
- `fixed_JNR_dB`；
- `sweep_JNR_dB`；
- `standard_levels_dB`；
- `paired_seeds`。

配置器必须验证：

- selected、fixed 和 swept 类型均存在于注册表；
- fixed 与 swept 无交集；
- fixed 与 swept 的并集严格等于 selected；
- known 严格等于 selected；
- unknown 为空；
- `min_components=1`，`max_components=类别数`；
- `combination_mode='all'`。

### 7.2 执行层

继续复用：

- `run_closedset_splits`；
- `run_power_sweep_levels`；
- `build_closedset_manifest`；
- `verify_saved_closedset_split`；
- 原子 `.partial.mat` 保存和 `resume` 机制。

现有两个 runner 中硬编码的 `closedset_6class_train/test` 文件名属于本次
必须消除的类别数耦合；除文件命名外，不改变它们已经验证的保存、恢复
和核对语义。

正式公开入口：

```matlab
generate_closedset_4class_datasets(...)
generate_closedset_5class_datasets(...)
generate_closedset_4class_5class_all(...)
```

总入口严格串行执行：

1. 四类全功率 train/test；
2. 四类九挡 train/test；
3. 五类全功率 train/test；
4. 五类九挡 train/test。

任何时候只允许一个正式 MATLAB 生成进程。

## 8. 种子设计

四个实验使用独立种子域：

| 实验 | `cfg.seed` |
|---|---:|
| 四类全功率 | 100260722 |
| 四类九挡 | 200260722 |
| 五类全功率 | 300260722 |
| 五类九挡 | 400260722 |

要求：

- 九挡内部同一 split、同一 sample_id 的 sample_seed、组合、分量子流、接收机噪声子流和非 JNR 参数抽样保持配对；
- 全功率与九挡之间不声明为配对数据；
- 四类与五类之间不声明为配对数据；
- 四个实验的 train/test sample_seed 集合必须全部两两无交集；
- 最终说明文件记录实际首尾种子和交集检查结果。

## 9. MAT 存储契约

沿用现有 v7.3 格式和变量：

- `iq_complex`；
- `data_legacy`；
- `label_all`；
- `label_selected`；
- `known_label`；
- `parameter_values`；
- `parameter_mask`；
- `manifest`；
- `config`；
- `parameter_schema`；
- `metadata`；
- 分量级有效 JNR、目标功率和实测功率 cell 真值。

差异仅为：

- 四类 `label_selected` 和 `known_label` 为 4 列；
- 五类为 5 列；
- `label_all` 继续保持固定十类注册表列，保证与六类及其他已有数据兼容；
- 参数模式继续保持注册表全量参数列，未活动类型通过 `parameter_mask=false` 表示无效。

## 10. 错误处理与恢复

- 正式文件先写入同目录的 `*.partial.mat`；
- 临时文件完成后执行增强回载；
- 验证通过才原子改名为正式文件；
- 改名后再次执行正式路径回载；
- `resume` 只复用 config、manifest、schema、metadata 和功率真值全部匹配的文件；
- 任一已存在正式文件语义不匹配时立即停止，不静默覆盖；
- 当前 split 的旧 partial 可精确删除后从该 split 开头重建；
- 不删除或修改现有六类数据。

## 11. 测试与验收

### 11.1 实施前测试

采用测试驱动方式增加以下测试：

- 四类类型顺序、已知/未知集合和组合数；
- 五类类型顺序、已知/未知集合和组合数；
- 每组合 400 条和 split 总量；
- 全功率独立 JNR 范围；
- 全功率有效 JNR、目标功率、实测功率及参数矩阵 JNR 的内部一致性；
- 四类与五类固定组/扫描组完整性；
- 九挡目录和文件命名；
- 四个种子域两两无交集；
- 九挡配对字段全等；
- label_selected、known_label 尺寸；
- 功率真值和参数掩码回载。

### 11.2 正式生成前验证

- 完整 MATLAB 测试套件全通过；
- Code Analyzer 无问题；
- 四类和五类分别执行全规模 dry-run，确认计数和种子但不写 MAT；
- 使用缩小样本数的端到端夹具验证四种实验路径；
- 检查 G 盘容量和目标目录不存在冲突文件。

### 11.3 正式生成后验收

- 40/40 个正式 MAT 存在；
- 0 个 `.partial.mat`；
- 四类每个 split 6,000 条、15 种组合、每组合 400 条；
- 五类每个 split 12,400 条、31 种组合、每组合 400 条；
- 18 个四类九挡文件和 18 个五类九挡文件的功率真值全部通过；
- 两组九挡各自完成完整配对字段核对；
- 四个实验所有 train/test 种子域两两无交集；
- 四类、五类文件夹分别生成 20 行 SHA-256 清单，并在上级目录合并生成
  40 行总清单；
- 数据说明记录正式文件数、总字节数、验证结论和 FMB Nyquist 裁剪语义。

## 12. 不在本次范围内

- 不增加未知类或开放集样本；
- 不改变已有物理干扰生成器；
- 不改变雷达体制和类型参数范围；
- 不生成类别不均衡或组合数量不均衡样本；
- 不并行启动多个正式 MATLAB 生成进程；
- 不覆盖或重新生成已经完成的六类数据。
