# 历史组合干扰程序可观测真值强化设计

日期：2026-07-27

状态：已由用户确认

目标分支：`codex/legacy-observable-truth-hardening`

## 1. 背景与目标

当前“可配置组合干扰数据生成”程序已经形成 v1（10 类/79 维）与 v2（13 类/295 维）两套显式参数注册表，并统一了功率控制语义；但仓库中的历史入口程序仍直接调用 `Jam9/Data_gen_*`。这些旧函数存在带宽、时延和幅度语义不完全可观测的问题，且旧数据文件只保存波形与类别标签，无法可靠支持组合干扰参数提取。

本次改造包含两条并行但保持兼容的工作线：

1. 强化当前可配置程序：增加科学语义版本标记、关键物理真值不变量检查和回归测试。
2. 迁移历史核心链路：修正 FMB、AMA、MFT、RangeFakeTarget、Traction 的物理/标签问题，并让三个历史入口在不破坏旧数据格式的前提下额外保存参数真值旁车文件。

最终原则是：监督标签只表达由最终生成波形唯一决定、且在离散采样条件下可观测的量。随机请求值、不可辨识分解量和未迁移类型不得伪装成有效真值。

## 2. 范围

### 2.1 当前可配置程序

计划修改：

- `可配置组合干扰数据生成/core/validate_generated_dataset.m`
- `可配置组合干扰数据生成/core/save_dataset_bundle.m`
- `可配置组合干扰数据生成/core/compose_jamming_sample.m`
- `可配置组合干扰数据生成/core/generate_dataset_from_manifest.m`
- `可配置组合干扰数据生成/tests/TestDatasetGeneration.m`
- `可配置组合干扰数据生成/tests/TestSingleJammingGenerators.m`

### 2.2 历史核心生成函数

计划修改：

- `Jam9/Data_gen_NoiseFM_Block.m`
- `Jam9/Data_gen_NoiseAM_Spot.m`
- `Jam9/Data_gen_Multi_fake_target.m`
- `Jam9/Data_gen_Range_fake_target.m`
- `Jam9/Data_gen_Tuoyin.m`

计划新增共享辅助函数：

- `Jam9/legacy_realized_noise_support.m`
- `Jam9/legacy_sample_unique_delays.m`
- `Jam9/legacy_zero_boundary_delay.m`
- `Jam9/legacy_initialize_truth.m`
- `Jam9/build_legacy_truth_bundle.m`
- `Jam9/save_legacy_truth_sidecar.m`

### 2.3 历史入口程序

计划修改：

- `组合干扰生成/Gen_com.m`
- `组合干扰生成/SP_gen_comjam.m`
- `Gen_openjamset.m`

计划新增测试：

- `可配置组合干扰数据生成/tests/TestLegacyCompatibility.m`

### 2.4 明确不在本次范围内

- 不给仓库中所有长尾旧脚本逐一增加旁车真值保存逻辑。
- 不改变三个历史入口原有的类别顺序、波形变量名、类别标签变量名和主文件命名规则。
- 不追求修正前后波形逐比特一致；物理错误被修正后，随机波形可以变化。
- 不扩展 v1 的 79 维或 v2 的 295 维参数注册表，不重新定义现有干扰类别编码。
- 不正式生成或覆盖训练集、测试集及已有论文数据。
- 不在设计和单元测试阶段向正式数据目录写入大规模数据。

## 3. 总体架构

```text
当前可配置程序
  ├─ v1：10 类干扰与 79 维真值
  ├─ v2：13 类干扰与 295 维真值
  ├─ 增加科学语义版本标记
  └─ 增加可观测真值不变量验证

历史入口
  ├─ Gen_com.m
  ├─ SP_gen_comjam.m
  └─ Gen_openjamset.m
       │
       ├─ 保持原调用参数、原主输出文件和原类别标签
       ├─ 调用已修正的 Jam9/Data_gen_* 核心函数
       └─ 额外写出同名 *_truth.mat 旁车文件
              ├─ parameter_values
              ├─ parameter_mask
              ├─ parameter_schema
              ├─ atomic_truth
              └─ metadata
```

历史长尾脚本仍可按原方式只请求一个输出，因此不会被迫迁移；只有三个选定入口负责聚合并保存组合级真值。

## 4. 兼容性设计

### 4.1 函数接口

现有输入参数及顺序全部保持不变。第一个输出继续表示旧程序原有的波形矩阵。

FMB、AMA、MFT 和 RangeFakeTarget 增加可选第二输出：

```matlab
[out_sig, truth] = Data_gen_NoiseFM_Block(...)
[out_sig, truth] = Data_gen_NoiseAM_Spot(...)
[out_sig, truth] = Data_gen_Multi_fake_target(...)
[out_sig, truth] = Data_gen_Range_fake_target(...)
```

Traction 保持原有两个输出，并增加可选第三输出：

```matlab
[out_sig, out_sig_pc, truth] = Data_gen_Tuoyin(...)
```

只请求旧输出数量时，函数不构建或返回真值结构，旧调用代码可以继续运行。

### 4.2 结构兼容与科学兼容

本设计保证结构兼容：输入参数数量、原输出数量、波形矩阵形状和历史主文件结构保持不变。

本设计不保证修正前后的波形逐比特兼容，因为以下错误必须改变生成结果：

- FMB/AMA 非共轭对称的离散频谱掩膜；
- MFT 假目标数量恒为上限加一；
- MFT 公共时延额外除以 `1e6`；
- MFT/Range 延迟碰撞和循环移位回绕；
- Traction 初始距离未进入实际传播时延。

因此，修正后产生的数据必须带科学语义版本，禁止把无版本旧 MAT 文件默认当作已修正数据。

### 4.3 真值输出的重复次数限制

旧程序的一个真值行必须与一个波形行一一对应。为避免 `nRepetition` 的历史复制语义造成标签歧义：

- 只请求旧输出时，保留原 `nRepetition` 行为；
- 请求 `truth` 时，要求 `nRepetition == 1`；
- 否则抛出 `LegacyJamming:TruthRequiresSingleRepetition`。

三个选定入口当前均使用 `nRepetition == 1`，因此不会影响其正常迁移。

## 5. 真值语义

### 5.1 通用规则

- `truth` 每个字段的第一维与 `out_sig` 的行数一致。
- 每个样本的有效参数必须由该样本的最终波形确定。
- 固定容量的多目标字段使用 NaN 补齐；对应掩码为 `false`。
- 未迁移类型在组合级 `parameter_values` 中保持 NaN，在 `parameter_mask` 中保持 `false`。
- 不允许用配置请求值、默认值或零值填充不可观测参数。
- 历史旁车参数单位严格使用 v1 的 79 维参数注册表定义。

### 5.2 FMB：阻塞式噪声调频干扰

每个样本返回：

- `JNR_dB`
- `bandwidth_factor`：离散频谱掩膜实际覆盖宽度与雷达带宽之比
- `interference_bandwidth_Hz`：离散 FFT 支撑对应的实际干扰带宽
- `center_frequency_Hz`
- `arrival_delay_s`

若保留连续随机请求的带宽因子，仅放入诊断元数据，例如 `requested_bandwidth_factor`；它不得进入监督参数列。

离散频谱支撑必须围绕目标中心频率构成共轭对称掩膜。实际带宽由最终选中的 FFT bin 数计算，而不是直接复制连续请求值。DC-only 掩膜不构成有效噪声带宽，必须拒绝。

### 5.3 AMA：瞄准式噪声调幅干扰

每个样本返回：

- `JNR_dB`
- `bandwidth_factor`
- `interference_bandwidth_Hz`
- `center_frequency_Hz`
- `arrival_delay_s`
- `modulation_index`

带宽真值与 FMB 使用相同的离散可观测定义。连续请求值仅作为诊断元数据，不作为监督真值。

### 5.4 MFT：多假目标干扰

每个样本返回：

- `JNR_dB = NaN`：旧 MFT 没有独立 JNR 控制量，对应掩码为 `false`
- `num_targets`
- `mean_delay_s`
- `mean_spacing_s`：排序后相邻最终总时延间隔的均值；单目标时为 NaN
- `delay_01_s ... delay_K_s`：最终施加到每个假目标上的总时延
- `amplitude_01 ... amplitude_K`
- `arrival_delay_s = NaN`

设计约束：

- `num_targets` 必须在 `[n_fake_tar_min, n_fake_tar_max]` 内均匀抽取，不能恒为上限加一。
- 公共干扰时延从秒转换为采样点时只乘 `fs`，不得再除以 `1e6`。
- 每个活动目标的总时延必须在离散采样网格上唯一。
- 延迟与幅度成对排序；排序后第 k 个延迟仍对应第 k 个幅度。
- `mean_delay_s` 基于最终总时延计算。
- 公共时延与目标相对时延的不同分解，只要最终总时延相同，就必须产生相同监督标签。
- `arrival_delay_s` 不单独监督，因为它已与目标相对时延合并且不可唯一辨识。

### 5.5 RangeFakeTarget：距离假目标干扰

每个样本返回：

- `JNR_dB = NaN`：旧 RangeFakeTarget 没有独立 JNR 控制量，对应掩码为 `false`
- `num_targets`
- `mean_delay_s`
- `delay_01_s ... delay_K_s`
- `amplitude_01 ... amplitude_K`
- `range_01_m ... range_K_m`
- `arrival_delay_s = NaN`

其中 `range_XX_m = c * delay_XX_s / 2`，由最终总时延确定。延迟唯一性、延迟—幅度配对、NaN 补齐以及不可辨识到达时延规则与 MFT 相同。

### 5.6 Traction：拖引干扰

每个样本返回：

- `initial_range_m`
- `velocity_mps`
- `nominal_amplitude`
- `effective_amplitude`
- `range_change_per_stage_m`
- `delay_change_per_stage_s`
- `stage_time_s`

初始距离与速度必须共同进入每个阶段的传播时延：

```text
range(stage) = initial_range + velocity * elapsed_stage_time
delay(stage) = 2 * range(stage) / c
```

延迟操作使用零边界移位或等效的非循环延迟，禁止 `circshift` 将末端样本回绕到开头。

Traction 的幅度参数是唯一的信号尺度来源。若组合场景设置功率偏置，偏置必须折算到 `effective_amplitude`：

```text
effective_amplitude = nominal_amplitude * 10^(power_offset_dB / 20)
```

不得在 Traction 内部再执行一套独立 JNR 归一化，也不得同时把幅度和 JNR 作为两个独立可调监督量。

## 6. 历史旁车文件格式

### 6.1 文件命名

历史主文件保持不变，例如：

```text
Jam4_zh0_tttt.mat
```

新增同目录旁车文件：

```text
Jam4_zh0_tttt_truth.mat
```

旁车文件的 stem 必须与主文件完全一致，只追加 `_truth`。

### 6.2 顶层变量

每个旁车文件至少包含：

- `parameter_values`：`N × 79` double
- `parameter_mask`：`N × 79` logical
- `parameter_schema`：当前 v1（10 类/79 维）参数注册表快照
- `label`：与历史主文件一致的类别标签及类别顺序
- `legacy_class_codes`：历史列索引到当前干扰代码的显式映射
- `truth_coverage`：每类已迁移、未迁移及缺失原因
- `atomic_truth`：各原子干扰的原始真值结构
- `source_sample_index`：每个组合输出行对应的原子样本行号
- `metadata`：生成配置、版本、入口、时间及兼容性信息

`metadata` 至少包括：

- `scientific_semantics_version`
- `parameter_registry_version = "v1"`
- `legacy_entrypoint`
- `legacy_main_file`
- `truth_file`
- `sample_count`
- `truth_width`
- `nRepetition`
- `corrected_legacy_physics = true`
- `unmigrated_types`

### 6.3 组合级聚合

`atomic_truth` 保存每类干扰单独生成时的真值，并与该类原子波形的 `1...nBatch` 行严格对齐。历史入口按组合块重复使用同一批原子样本，因此必须同时构建 `source_sample_index`：每个组合块都显式写入 `1...nBatch`，不能使用组合文件的全局行号直接索引原子真值。

组合输出第 i 行启用某类干扰时，`build_legacy_truth_bundle.m` 使用 `source_sample_index(i)` 读取该类 `atomic_truth` 的对应原子行，再写入 v1 的 79 维模式。这样单干扰块、二元组合块及更高阶组合块都与实际相加的原子波形行严格一致。

聚合规则：

1. 活动且已迁移的类型写入真实参数值并将掩码置为 `true`。
2. 非活动类型保持 NaN/`false`。
3. 活动但未迁移的类型也保持 NaN/`false`，并在 `truth_coverage` 中明确标记。
4. 同一参数列出现重复所有权、映射缺失、行数不一致或标签映射不合法时，停止保存。
5. 旁车 `label` 必须与主文件逐行、逐列一致。
6. `source_sample_index` 必须是有限正整数，且不得超过任一被引用原子真值的行数。

## 7. 共享辅助函数职责

### 7.1 `legacy_realized_noise_support.m`

职责：

- 将连续目标带宽投影到离散 FFT bin；
- 构造围绕中心频率的共轭对称支撑；
- 返回最终掩膜、实际带宽 Hz、实际带宽因子及诊断信息；
- 拒绝空支撑、DC-only 支撑、越过 Nyquist 或无法构成有效带宽的配置。

### 7.2 `legacy_sample_unique_delays.m`

职责：

- 在给定秒范围和采样率下构造可用的离散延迟槽；
- 无放回抽取活动目标延迟；
- 合并公共时延与相对时延；
- 检查容量与最终唯一性；
- 返回按最终时延排序的索引。

当可用离散延迟槽不足以容纳目标数时，抛出 `LegacyJamming:InsufficientDelaySlots`，而不是静默制造重复标签。

### 7.3 `legacy_zero_boundary_delay.m`

职责：

- 对行向量或矩阵执行非循环延迟；
- 正时延左侧补零，超出记录长度的尾部被截断；
- 支持 Traction 阶段延迟和 MFT/Range 目标延迟；
- 明确处理零延迟、全越界和非法非有限输入。

### 7.4 `legacy_initialize_truth.m`

职责：

- 建立固定行数的真值结构；
- 将未使用的数值槽初始化为 NaN；
- 建立对应有效掩码或活动计数；
- 保存诊断元数据但不将诊断字段自动注册为监督参数。

### 7.5 `build_legacy_truth_bundle.m`

职责：

- 显式加载当前 v1（10 类/79 维）干扰注册表和参数模式；
- 验证历史类别列与当前代码之间的显式映射；
- 合并组合标签与原子真值；
- 使用显式 `source_sample_index` 将组合输出行映射回原子样本行；
- 生成 `parameter_values`、`parameter_mask`、`truth_coverage` 和 `metadata`；
- 写文件前完成全部形状、NaN/掩码和所有权检查。

该函数只构建内存 bundle，便于单元测试不接触正式路径。

### 7.6 `save_legacy_truth_sidecar.m`

职责：

- 根据历史主文件路径生成同 stem 的 `_truth.mat` 路径；
- 再次核对 bundle 必需字段、行数和 79 维宽度；
- 使用 `save -struct` 写出旁车顶层变量；
- 保存失败时使用稳定错误，不改变或覆盖历史主文件。

## 8. 当前可配置程序强化

### 8.1 科学语义版本

当前可配置程序的 v1 与 v2 数据在新保存时都在顶层元数据中增加：

```matlab
metadata.scientific_semantics_version
```

版本字符串必须稳定且可比较，用于区分：

- 无版本历史数据；
- 修正前的可配置数据；
- 本次可观测真值强化后的数据。

载入或验证旧 MAT 文件时：

- 缺失版本只能判定为 legacy/unknown；
- 不得自动写入或假定当前版本；
- 若调用者要求已修正语义，必须显式失败。

### 8.2 新增数据不变量

`validate_generated_dataset.m` 增加三组逐样本检查：

1. FMB/AMA：保存的 `interference_bandwidth_Hz` 与实际离散频谱支撑一致，`bandwidth_factor` 与前者/雷达带宽一致。
2. MFT/RangeFakeTarget：活动延迟槽等于最终施加的总延迟；活动槽唯一；未使用槽为 NaN/`false`；独立 `arrival_delay_s` 为 NaN/`false`。
3. Traction：`effective_amplitude / nominal_amplitude` 与功率偏置满足 `20*log10` 关系；不存在第二套独立 JNR 缩放。

`compose_jamming_sample.m` 保留生成器返回的可观测诊断量，供最终数据校验使用；诊断量不自动扩展监督目标维度。

`generate_dataset_from_manifest.m` 创建 `metadata.scientific_semantics_version`，并将逐样本校验需要的分量级语义信息汇总进 bundle。

`save_dataset_bundle.m` 保存科学语义版本及与当前参数模式一致的元数据副本。

## 9. 稳定错误标识

本次新增或统一使用以下错误标识：

- `LegacyJamming:InvalidPhysicalParameter`：采样率、带宽、幅度、时延或其他物理参数非法。
- `LegacyJamming:InvalidTruthShape`：波形、原子真值、标签或组合真值的行数/容量不一致。
- `LegacyJamming:TruthRequiresSingleRepetition`：请求真值但 `nRepetition ~= 1`。
- `LegacyJamming:InsufficientDelaySlots`：离散延迟槽容量不足。
- `LegacyJamming:InvalidLegacyLabelMap`：历史类别列到当前代码的映射缺失、重复或非法。
- `LegacyJamming:SidecarSaveFailed`：旁车路径有效但 MAT 文件未能成功创建。

所有入口在写出旁车文件前必须完成验证；验证失败时不允许留下看似完整的 `_truth.mat`。

## 10. 三个历史入口的集成方式

每个入口按以下顺序工作：

1. 保留原随机数设置、信号参数、样本数量、组合构造和主文件保存流程。
2. 对五个已迁移核心函数请求 `truth` 输出。
3. 对未迁移类型只保留原波形输出。
4. 建立显式 `legacy_class_codes`，禁止依赖文件名或隐式列顺序猜测类别。
5. 按原组合规则构建 `label`。
6. 与每个组合块同步构建 `source_sample_index = (1:nBatch).'`，并按输出块顺序拼接。
7. 调用 `build_legacy_truth_bundle.m` 聚合真值。
8. 调用 `save_legacy_truth_sidecar.m` 将主文件 stem 对应的 `_truth.mat` 写入同一目录。

路径处理必须基于入口脚本位置或已解析的绝对路径，不依赖用户当前工作目录碰巧正确。新增辅助路径不能永久污染 MATLAB path。

## 11. 测试设计

### 11.1 当前程序回归

继续运行现有完整测试集，并新增：

- FMB/AMA 请求带宽被量化后，真值等于实际掩膜带宽；
- 不同连续请求值映射到相同离散掩膜时，监督带宽相同；
- DC-only 支撑被拒绝；
- MFT/Range 的最终时延、唯一性、NaN 补齐和掩码一致；
- 等价时延分解得到相同最终延迟真值；
- Traction 功率偏置只改变 `effective_amplitude`；
- 保存和回载科学语义版本；
- 无版本旧 MAT 不被误判为当前语义。

### 11.2 历史兼容性测试

`TestLegacyCompatibility.m` 使用极小 `nBatch`、短波形和临时目录，覆盖：

- 五个核心函数按旧输出数量调用时，输出数量与矩阵形状不变；
- 可选真值输出与波形逐行对齐；
- MFT 数量在配置范围内，且不再恒为上限加一；
- MFT 公共时延单位正确；
- MFT/Range 活动总时延唯一且延迟—幅度配对不被打乱；
- Traction 初始距离与速度都实际影响波形延迟；
- `nRepetition ~= 1` 请求真值时稳定失败；
- 非法标签映射、真值行数错误和延迟槽不足使用稳定错误标识；
- 组合块通过 `source_sample_index` 精确复用对应原子样本真值；
- 三个入口的集成逻辑可在临时目录产生旧主结构和新旁车结构；
- 测试不覆盖或生成正式数据集。

### 11.3 验收标准

实现完成必须同时满足：

1. 不请求真值时，旧输入参数、旧输出数量和波形矩阵尺寸不变。
2. FMB/AMA 相同离散掩膜对应相同实际带宽真值。
3. DC-only 噪声支撑被拒绝。
4. MFT 假目标数始终位于配置范围内。
5. MFT 公共时延不再错误除以 `1e6`。
6. MFT/Range 活动延迟唯一，延迟与幅度成对排序。
7. 等价时延分解得到相同最终延迟真值。
8. Traction 初始距离和速度均影响最终波形延迟。
9. Traction 功率偏置只通过有效幅度生效。
10. 三个入口保留原数据/标签文件，同时生成一一对应的真值旁车文件。
11. 当前基线 231 项测试全部继续通过。
12. 新增历史兼容性和真值语义测试全部通过。
13. 自动测试只使用临时目录和小样本，不覆盖正式数据。

## 12. 风险与控制

### 12.1 旧结果可重复性变化

物理错误修正会改变旧随机种子下的具体波形。控制方式是保留结构兼容，并用 `scientific_semantics_version` 明确区分数据代际；不得混合使用不同语义版本的数据而不记录来源。

### 12.2 历史类别映射错误

历史脚本中类别列含义可能依赖局部变量顺序。控制方式是每个入口显式声明 `legacy_class_codes`，聚合器验证映射长度、唯一性和代码合法性。

### 12.3 标签看似完整但不可观测

未迁移类型若填入默认参数，会制造虚假监督。控制方式是强制 NaN/`false`，并通过 `truth_coverage` 报告覆盖范围。

### 12.4 正式数据被测试污染

控制方式是入口集成拆分为可测试的内存构建逻辑，所有测试使用临时目录、小样本和显式输出根目录。

## 13. 实施检查点

实现阶段不开子智能体，在当前任务内分批执行：

1. 检查点 A：先写失败测试和共享辅助函数，验证五类原子干扰真值。
2. 检查点 B：迁移五个 `Data_gen_*` 并复跑旧接口/新真值测试。
3. 检查点 C：强化当前可配置程序的版本与不变量验证。
4. 检查点 D：迁移三个历史入口并验证旁车文件。
5. 检查点 E：运行完整 MATLAB 测试、差异检查和代码复核。

每个检查点只在相关测试通过后进入下一批；正式数据生成不属于本次实现验收。
