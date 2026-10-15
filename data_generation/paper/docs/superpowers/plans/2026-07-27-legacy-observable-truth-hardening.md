# Legacy Observable Truth Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The user selected inline execution with checkpoints and explicitly disabled subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修正五个历史干扰核心函数的带宽、目标时延和拖引轨迹语义，为三个历史组合入口增加与原数据逐行对齐的 v1（10 类/79 维）真值旁车，同时强化当前可配置程序的科学语义版本和可观测真值校验。

**Architecture:** 历史 `Data_gen_*` 保持原输入与原第一输出，按需返回逐波形行真值；共享辅助函数统一离散带宽、非循环时延、唯一目标槽和真值初始化。三个历史入口继续保存原 `data`/`label` 文件，并通过显式 `source_sample_index` 将每个组合块映射回原子样本，再写出 `_truth.mat`。当前可配置程序继续使用 v1/v2 注册表，在数据生成、合成、验证和保存链路中加入稳定科学语义版本及 FMB/AMA、MFT/RangeFakeTarget、Traction 不变量。

**Tech Stack:** MATLAB R202x、`matlab.unittest`、MAT v7/v7.3、Git、PowerShell；测试不生成正式数据集。

---

## 0. 文件结构与职责

### 新增文件

- `Jam9/legacy_initialize_truth.m`：建立逐样本 NaN 真值结构。
- `Jam9/legacy_zero_boundary_delay.m`：整数采样点的零边界非循环延迟。
- `Jam9/legacy_realized_noise_support.m`：构造共轭对称 FFT 掩膜并返回实际离散带宽。
- `Jam9/legacy_sample_unique_delays.m`：无放回抽取可辨识目标时延槽并排序。
- `Jam9/build_legacy_truth_bundle.m`：将历史组合标签、原子真值和原子行索引编码为 v1 79 维 bundle。
- `Jam9/save_legacy_truth_sidecar.m`：验证并保存 `_truth.mat` 顶层变量。
- `可配置组合干扰数据生成/core/current_scientific_semantics_version.m`：返回唯一稳定科学语义版本字符串。
- `可配置组合干扰数据生成/tests/TestLegacyCompatibility.m`：历史接口、物理真值、旁车和入口接线测试。
- `历史组合干扰程序真值旁车说明.md`：最终用户说明。

### 修改文件

- `Jam9/Data_gen_NoiseFM_Block.m:1`：可选真值、离散带宽、非循环到达时延。
- `Jam9/Data_gen_NoiseAM_Spot.m:1`：可选真值、离散带宽、固定调制指数、非循环到达时延。
- `Jam9/Data_gen_Multi_fake_target.m:1`：正确目标数、正确公共时延单位、唯一总时延、配对排序。
- `Jam9/Data_gen_Range_fake_target.m:1`：最终总时延、距离真值、非循环延迟。
- `Jam9/Data_gen_Tuoyin.m:1`：初始距离进入传播时延、速度轨迹、可选第三输出。
- `可配置组合干扰数据生成/core/compose_jamming_sample.m:1`：合成时立即检查分量可观测真值和 Traction 单一幅度尺度。
- `可配置组合干扰数据生成/core/generate_dataset_from_manifest.m:1`：写入科学语义版本。
- `可配置组合干扰数据生成/core/validate_generated_dataset.m:1`：数据集级可观测真值不变量。
- `可配置组合干扰数据生成/core/save_dataset_bundle.m:1`：拒绝缺失/未知科学语义版本。
- `可配置组合干扰数据生成/tests/TestSingleJammingGenerators.m:1780`：补强当前生成器离散真值回归。
- `可配置组合干扰数据生成/tests/TestDatasetGeneration.m:333`：版本保存和篡改检测。
- `组合干扰生成/Gen_com.m:76`：获取 FMB/AMA/MFT 原子真值并为 4 类组合保存旁车。
- `组合干扰生成/SP_gen_comjam.m:60`：为 7 类组合保存部分覆盖旁车。
- `Gen_openjamset.m:55`：为五组开放集组合分别保存部分覆盖旁车。

### 分批检查点

- 检查点 A：任务 1--2，共享基础设施与 FMB/AMA。
- 检查点 B：任务 3--4，MFT/RangeFakeTarget 与 Traction。
- 检查点 C：任务 5，当前可配置程序强化。
- 检查点 D：任务 6--7，79 维旁车与三个历史入口。
- 检查点 E：任务 8，完整回归、说明文档和最终复核。

## 1. 统一测试命令

以下 PowerShell 命令在工作树根目录运行。后续步骤中的 `TEST_FILE` 和 `TEST_NAME` 替换为任务给出的精确值。

```powershell
matlab -batch "repoRoot=pwd; appRoot=fullfile(repoRoot,'可配置组合干扰数据生成'); addpath(repoRoot,fullfile(repoRoot,'Jam9'),appRoot,fullfile(appRoot,'core'),fullfile(appRoot,'generators'),fullfile(appRoot,'tests')); results=runtests(TEST_FILE,'Name',TEST_NAME); assertSuccess(results);"
```

完整回归命令：

```powershell
matlab -batch "repoRoot=pwd; appRoot=fullfile(repoRoot,'可配置组合干扰数据生成'); addpath(repoRoot,fullfile(repoRoot,'Jam9'),appRoot,fullfile(appRoot,'core'),fullfile(appRoot,'generators'),fullfile(appRoot,'tests')); results=runtests(fullfile(appRoot,'tests')); disp(table(results)); assertSuccess(results);"
```

## Task 1: 建立历史测试夹具、NaN 真值初始化和零边界延迟

**Files:**

- Create: `可配置组合干扰数据生成/tests/TestLegacyCompatibility.m`
- Create: `Jam9/legacy_initialize_truth.m`
- Create: `Jam9/legacy_zero_boundary_delay.m`

- [ ] **Step 1: 创建历史兼容测试类并写两个失败测试**

创建测试类，固定恢复 MATLAB path，验证 NaN 初始化和非循环延迟：

```matlab
classdef TestLegacyCompatibility < matlab.unittest.TestCase
    methods (TestClassSetup)
        function addRepositoryPaths(testCase)
            testsDir = fileparts(mfilename('fullpath'));
            appRoot = fileparts(testsDir);
            repoRoot = fileparts(appRoot);
            originalPath = path;
            addpath(repoRoot, fullfile(repoRoot, 'Jam9'), appRoot, ...
                fullfile(appRoot, 'core'), fullfile(appRoot, 'generators'));
            testCase.addTeardown(@() path(originalPath));
        end
    end

    methods (Test)
        function truthInitializerUsesColumnNaNPadding(testCase)
            truth = legacy_initialize_truth(3, {'alpha', 'beta'});
            testCase.verifyEqual(fieldnames(truth), {'alpha'; 'beta'});
            testCase.verifySize(truth.alpha, [3, 1]);
            testCase.verifyTrue(all(isnan(truth.alpha)));
            testCase.verifyTrue(all(isnan(truth.beta)));
        end

        function zeroBoundaryDelayNeverWrapsTailToHead(testCase)
            signal = [1, 2, 3, 4, 5];
            shifted = legacy_zero_boundary_delay(signal, 2);
            testCase.verifyEqual(shifted, [0, 0, 1, 2, 3]);
            testCase.verifyEqual(legacy_zero_boundary_delay(signal, 5), ...
                zeros(1, 5));
        end
    end
end
```

- [ ] **Step 2: 运行测试并确认缺少函数而失败**

```powershell
matlab -batch "repoRoot=pwd; appRoot=fullfile(repoRoot,'可配置组合干扰数据生成'); addpath(repoRoot,fullfile(repoRoot,'Jam9'),appRoot,fullfile(appRoot,'core'),fullfile(appRoot,'generators'),fullfile(appRoot,'tests')); results=runtests(fullfile(appRoot,'tests','TestLegacyCompatibility.m')); assertSuccess(results);"
```

Expected: FAIL，MATLAB 报告 `legacy_initialize_truth` 或 `legacy_zero_boundary_delay` 未定义。

- [ ] **Step 3: 实现 `legacy_initialize_truth.m`**

```matlab
function truth = legacy_initialize_truth(sampleCount, fieldNames)
%LEGACY_INITIALIZE_TRUTH 创建逐样本 NaN 真值结构。
%   每个字段均为 SAMPLECOUNT×1 double。未被最终波形唯一确定的量保持
%   NaN，后续编码层据此保持 parameter_mask=false。

    if ~isnumeric(sampleCount) || ~isreal(sampleCount) || ...
            ~isscalar(sampleCount) || ~isfinite(sampleCount) || ...
            sampleCount < 1 || sampleCount ~= fix(sampleCount) || ...
            ~iscellstr(fieldNames) || isempty(fieldNames) || ...
            numel(unique(fieldNames)) ~= numel(fieldNames) || ...
            any(~cellfun(@isvarname, fieldNames))
        error('LegacyJamming:InvalidTruthShape', ...
            'sampleCount 必须是正整数，fieldNames 必须是唯一合法字段名单。');
    end
    truth = struct();
    for fieldIndex = 1:numel(fieldNames)
        truth.(fieldNames{fieldIndex}) = nan(double(sampleCount), 1);
    end
end
```

- [ ] **Step 4: 实现 `legacy_zero_boundary_delay.m`**

```matlab
function shifted = legacy_zero_boundary_delay(signal, delaySamples)
%LEGACY_ZERO_BOUNDARY_DELAY 执行整数采样点的正向非循环延迟。
%   输入按列方向独立处理；左侧补零、右侧截断，禁止 circshift 回绕。

    if ~isnumeric(signal) || ndims(signal) > 2 || isempty(signal) || ...
            ~all(isfinite(real(signal)), 'all') || ...
            ~all(isfinite(imag(signal)), 'all') || ...
            ~isnumeric(delaySamples) || ~isreal(delaySamples) || ...
            ~isscalar(delaySamples) || ~isfinite(delaySamples) || ...
            delaySamples < 0 || delaySamples ~= fix(delaySamples)
        error('LegacyJamming:InvalidPhysicalParameter', ...
            'signal 必须有限，delaySamples 必须是非负整数标量。');
    end
    shifted = zeros(size(signal), 'like', signal);
    if delaySamples < size(signal, 2)
        shifted(:, delaySamples + 1:end) = ...
            signal(:, 1:end - delaySamples);
    end
end
```

- [ ] **Step 5: 运行历史测试并确认通过**

Run: Task 1 Step 2 的命令。

Expected: 2 tests passed。

- [ ] **Step 6: 提交检查点 A 的基础设施**

```powershell
git add -- "Jam9/legacy_initialize_truth.m" "Jam9/legacy_zero_boundary_delay.m" "可配置组合干扰数据生成/tests/TestLegacyCompatibility.m"
git commit -m "test: add legacy truth compatibility harness"
```

## Task 2: 修正 FMB/AMA 离散带宽并增加可选真值

**Files:**

- Create: `Jam9/legacy_realized_noise_support.m`
- Modify: `Jam9/Data_gen_NoiseFM_Block.m:1`
- Modify: `Jam9/Data_gen_NoiseAM_Spot.m:1`
- Modify: `可配置组合干扰数据生成/tests/TestLegacyCompatibility.m`

- [ ] **Step 1: 写离散掩膜和旧接口兼容失败测试**

在 `TestLegacyCompatibility` 中加入：

```matlab
function realizedNoiseSupportIsSymmetricAndQuantized(testCase)
    [maskA, bandwidthA, infoA] = ...
        legacy_realized_noise_support(16, 16, 4.1);
    [maskB, bandwidthB, infoB] = ...
        legacy_realized_noise_support(16, 16, 4.9);
    testCase.verifyEqual(maskA, maskB);
    testCase.verifyEqual(bandwidthA, 4);
    testCase.verifyEqual(bandwidthB, 4);
    testCase.verifyEqual(infoA.positive_bin_count, 2);
    testCase.verifyEqual(infoB.support_bin_count, 5);
    testCase.verifyEqual(maskA(2:3), true(1, 2));
    testCase.verifyEqual(maskA(end-1:end), true(1, 2));
end

function realizedNoiseSupportRejectsDcOnly(testCase)
    testCase.verifyError(@() legacy_realized_noise_support(16, 16, 1), ...
        'LegacyJamming:InvalidPhysicalParameter');
end

function fmbAmaKeepOldWaveformOutputAndReturnTruth(testCase)
    fs = 1e6;
    sampleCount = 4096;
    reference = complex(ones(sampleCount, 1));
    rng(21);
    oldFmb = Data_gen_NoiseFM_Block(reference, 1, 30, 30, ...
        2, 2, 10e3, 0, fs, sampleCount, 2, 1, 0, 0, 0);
    rng(21);
    [newFmb, fmbTruth] = Data_gen_NoiseFM_Block(reference, 1, 30, 30, ...
        2, 2, 10e3, 0, fs, sampleCount, 2, 1, 0, 0, 0);
    testCase.verifyEqual(oldFmb, newFmb);
    testCase.verifySize(newFmb, [2, sampleCount]);
    testCase.verifyEqual(fmbTruth.bandwidth_factor, ...
        fmbTruth.interference_bandwidth_Hz / 10e3, 'AbsTol', 1e-12);

    rng(22);
    oldAma = Data_gen_NoiseAM_Spot(reference, 1, 30, 30, ...
        2, 2, 10e3, 0, fs, sampleCount, 2, 1, 0, 0, 0);
    rng(22);
    [ama, amaTruth] = Data_gen_NoiseAM_Spot(reference, 1, 30, 30, ...
        2, 2, 10e3, 0, fs, sampleCount, 2, 1, 0, 0, 0);
    testCase.verifyEqual(oldAma, ama);
    testCase.verifySize(ama, [2, sampleCount]);
    testCase.verifyEqual(amaTruth.modulation_index, repmat(1/3, 2, 1));
    testCase.verifyEqual(amaTruth.bandwidth_factor, ...
        amaTruth.interference_bandwidth_Hz / 10e3, 'AbsTol', 1e-12);
end

function legacyTruthRejectsRepeatedCopies(testCase)
    fs = 1e6;
    reference = complex(ones(4096, 1));
    testCase.verifyError(@() request_fmb_truth(reference, fs), ...
        'LegacyJamming:TruthRequiresSingleRepetition');
end
```

并在测试文件末尾加入本地函数：

```matlab
function request_fmb_truth(reference, fs)
    [~, ~] = Data_gen_NoiseFM_Block(reference, 1, 30, 30, ...
        2, 2, 10e3, 0, fs, numel(reference), 1, 2, 0, 0, 0);
end
```

- [ ] **Step 2: 运行新增测试并确认失败**

```powershell
matlab -batch "repoRoot=pwd; appRoot=fullfile(repoRoot,'可配置组合干扰数据生成'); addpath(repoRoot,fullfile(repoRoot,'Jam9'),appRoot,fullfile(appRoot,'core'),fullfile(appRoot,'generators'),fullfile(appRoot,'tests')); results=runtests(fullfile(appRoot,'tests','TestLegacyCompatibility.m'),'Name',{'TestLegacyCompatibility/realizedNoiseSupportIsSymmetricAndQuantized','TestLegacyCompatibility/realizedNoiseSupportRejectsDcOnly','TestLegacyCompatibility/fmbAmaKeepOldWaveformOutputAndReturnTruth','TestLegacyCompatibility/legacyTruthRejectsRepeatedCopies'}); assertSuccess(results);"
```

Expected: FAIL，辅助函数缺失且旧 FMB/AMA 不支持第二输出。

- [ ] **Step 3: 实现共轭对称离散支撑**

`legacy_realized_noise_support.m` 使用未移位 FFT 索引，排除无配对的 Nyquist 单点：

```matlab
function [mask, realizedBandwidth_Hz, info] = ...
        legacy_realized_noise_support(sampleLength, fs_Hz, requestedBandwidth_Hz)
%LEGACY_REALIZED_NOISE_SUPPORT 构造实噪声所需的共轭对称离散低通支撑。

    validInteger = isnumeric(sampleLength) && isreal(sampleLength) && ...
        isscalar(sampleLength) && isfinite(sampleLength) && ...
        sampleLength >= 3 && sampleLength == fix(sampleLength);
    validRate = isnumeric(fs_Hz) && isreal(fs_Hz) && isscalar(fs_Hz) && ...
        isfinite(fs_Hz) && fs_Hz > 0;
    validBandwidth = isnumeric(requestedBandwidth_Hz) && ...
        isreal(requestedBandwidth_Hz) && isscalar(requestedBandwidth_Hz) && ...
        isfinite(requestedBandwidth_Hz) && requestedBandwidth_Hz > 0;
    if ~(validInteger && validRate && validBandwidth)
        error('LegacyJamming:InvalidPhysicalParameter', ...
            '采样点数、采样率和请求带宽必须为有限有效值。');
    end
    binSpacing_Hz = fs_Hz / sampleLength;
    maxPositiveBins = floor((sampleLength - 1) / 2);
    positiveBinCount = min(floor(requestedBandwidth_Hz / ...
        (2 * binSpacing_Hz)), maxPositiveBins);
    if positiveBinCount < 1
        error('LegacyJamming:InvalidPhysicalParameter', ...
            '离散带宽必须至少包含一对非零共轭 FFT 频点。');
    end
    mask = false(1, sampleLength);
    mask(1) = true;
    mask(2:positiveBinCount + 1) = true;
    mask(end - positiveBinCount + 1:end) = true;
    realizedBandwidth_Hz = 2 * positiveBinCount * binSpacing_Hz;
    info = struct('bin_spacing_Hz', binSpacing_Hz, ...
        'positive_bin_count', positiveBinCount, ...
        'support_bin_count', 2 * positiveBinCount + 1, ...
        'requested_bandwidth_Hz', requestedBandwidth_Hz);
end
```

- [ ] **Step 4: 迁移 FMB 输出和真值语义**

将函数签名改为：

```matlab
function [out_sig, truth] = Data_gen_NoiseFM_Block(sig_LOS, nPulse, ...
    JNR_min, JNR_max, L_min, L_max, sig_bandwidth, freq0, fs, nPP, ...
    nBatch, nRepetition, flag_addTar, jam_delay_min, jam_delay_max)
```

在分配输出后执行：

```matlab
sampleCount = nBatch * nPulse;
truth = struct();
if nargout > 1
    if nRepetition ~= 1
        error('LegacyJamming:TruthRequiresSingleRepetition', ...
            '请求逐波形真值时 nRepetition 必须等于 1。');
    end
    truth = legacy_initialize_truth(sampleCount, {'JNR_dB', ...
        'bandwidth_factor', 'interference_bandwidth_Hz', ...
        'center_frequency_Hz', 'arrival_delay_s', ...
        'requested_bandwidth_factor'});
end
```

每次重复中使用以下离散化，不再手写非对称 `f_filter`：

```matlab
requestedFactor = L_min + (L_max - L_min) * rand();
requestedInterferenceBandwidth_Hz = requestedFactor * sig_bandwidth;
fdeRequested_Hz = requestedInterferenceBandwidth_Hz / ...
    (2 * sqrt(2 * log(2)));
noiseBandwidthRequested_Hz = fdeRequested_Hz / mfe;
[f_filter, realizedNoiseBandwidth_Hz] = ...
    legacy_realized_noise_support(nSample, fs, noiseBandwidthRequested_Hz);
realizedInterferenceBandwidth_Hz = realizedNoiseBandwidth_Hz * ...
    mfe * 2 * sqrt(2 * log(2));
realizedFactor = realizedInterferenceBandwidth_Hz / sig_bandwidth;
color_noise = real(ifft(fft(randn(1, nSample)) .* f_filter));
color_noise = color_noise - mean(color_noise);
sigma_n = std(color_noise);
if ~isfinite(sigma_n) || sigma_n <= 0
    error('LegacyJamming:InvalidPhysicalParameter', ...
        'FMB 离散带限噪声必须具有有限正标准差。');
end
KFM = (realizedInterferenceBandwidth_Hz / ...
    (2 * sqrt(2 * log(2)))) / sigma_n;
```

到达时延改为：

```matlab
jamDelaySamples = round((jam_delay_min + ...
    (jam_delay_max - jam_delay_min) * rand()) * fs);
out_sig(iBatch, :) = legacy_zero_boundary_delay( ...
    out_sig(iBatch, :), jamDelaySamples);
```

当 `nargout > 1` 时写入：

```matlab
truth.JNR_dB(iBatch) = JNR;
truth.bandwidth_factor(iBatch) = realizedFactor;
truth.interference_bandwidth_Hz(iBatch) = ...
    realizedInterferenceBandwidth_Hz;
truth.center_frequency_Hz(iBatch) = freq0;
truth.arrival_delay_s(iBatch) = jamDelaySamples / fs;
truth.requested_bandwidth_factor(iBatch) = requestedFactor;
```

- [ ] **Step 5: 迁移 AMA 输出和真值语义**

将签名改为可选第二输出，并使用与 FMB 相同的重复次数守卫。AMA 的请求带宽直接进入离散支撑：

```matlab
[f_filter, realizedInterferenceBandwidth_Hz] = ...
    legacy_realized_noise_support(nSample, fs, ...
    requestedFactor * sig_bandwidth);
realizedFactor = realizedInterferenceBandwidth_Hz / sig_bandwidth;
color_noise = real(ifft(fft(randn(1, nSample)) .* f_filter));
color_noise = real(sqrt(fs / realizedInterferenceBandwidth_Hz) * color_noise);
modulationIndex = 1 / U0;
```

AMA 仅在 `nargout > 1` 时初始化以下真值，并在同一条件内逐行赋值：

```matlab
truth = legacy_initialize_truth(sampleCount, {'JNR_dB', ...
    'bandwidth_factor', 'interference_bandwidth_Hz', ...
    'center_frequency_Hz', 'modulation_index', 'arrival_delay_s', ...
    'requested_bandwidth_factor'});
truth.JNR_dB(iBatch) = JNR;
truth.bandwidth_factor(iBatch) = realizedFactor;
truth.interference_bandwidth_Hz(iBatch) = ...
    realizedInterferenceBandwidth_Hz;
truth.center_frequency_Hz(iBatch) = freq0;
truth.modulation_index(iBatch) = modulationIndex;
truth.arrival_delay_s(iBatch) = jamDelaySamples / fs;
truth.requested_bandwidth_factor(iBatch) = requestedFactor;
```

- [ ] **Step 6: 运行 Task 2 测试和全部历史测试**

Run: Task 2 Step 2，然后运行 Task 1 Step 2 的整文件命令。

Expected: 全部通过；旧单输出与同种子双输出的第一输出完全一致。

- [ ] **Step 7: 提交 FMB/AMA 修复**

```powershell
git add -- "Jam9/legacy_realized_noise_support.m" "Jam9/Data_gen_NoiseFM_Block.m" "Jam9/Data_gen_NoiseAM_Spot.m" "可配置组合干扰数据生成/tests/TestLegacyCompatibility.m"
git commit -m "fix: expose realized bandwidth truth in legacy noise jammers"
```

## Task 3: 修正 MFT/RangeFakeTarget 的目标数、时延和配对真值

**Files:**

- Create: `Jam9/legacy_sample_unique_delays.m`
- Modify: `Jam9/Data_gen_Multi_fake_target.m:1`
- Modify: `Jam9/Data_gen_Range_fake_target.m:1`
- Modify: `可配置组合干扰数据生成/tests/TestLegacyCompatibility.m`

- [ ] **Step 1: 写唯一时延、范围内目标数和等价分解测试**

```matlab
function uniqueDelaySamplerSortsObservableTotals(testCase)
    rng(31);
    selection = legacy_sample_unique_delays([1, 6], [2, 2], 3, 1, 16);
    testCase.verifyEqual(selection.total_samples, ...
        sort(selection.total_samples));
    testCase.verifyEqual(numel(unique(selection.total_samples)), 3);
    testCase.verifyEqual(selection.arrival_samples, 2);
end

function uniqueDelaySamplerRejectsInsufficientSlots(testCase)
    testCase.verifyError(@() legacy_sample_unique_delays( ...
        [1, 2], [0, 0], 3, 1, 16), ...
        'LegacyJamming:InsufficientDelaySlots');
end

function mftCountDelayAndAmplitudeTruthAreObservable(testCase)
    fs = 1e6;
    reference = complex([ones(16, 1); zeros(240, 1)]);
    rng(32);
    oldSignal = Data_gen_Multi_fake_target(reference, 1, fs, ...
        3, 5, 10e-6, 40e-6, 0.5, 1.5, 20, 1, 0, 5e-6, 5e-6);
    rng(32);
    [signal, truth] = Data_gen_Multi_fake_target(reference, 1, fs, ...
        3, 5, 10e-6, 40e-6, 0.5, 1.5, 20, 1, 0, 5e-6, 5e-6);
    testCase.verifyEqual(oldSignal, signal);
    testCase.verifySize(signal, [20, numel(reference)]);
    testCase.verifyGreaterThanOrEqual(truth.num_targets, 3);
    testCase.verifyLessThanOrEqual(truth.num_targets, 5);
    for row = 1:20
        count = truth.num_targets(row);
        delays = truth.delay_s(row, 1:count);
        amplitudes = truth.amplitude(row, 1:count);
        testCase.verifyEqual(delays, sort(delays));
        testCase.verifyEqual(numel(unique(delays)), count);
        testCase.verifyTrue(all(isfinite(amplitudes)));
        testCase.verifyTrue(isnan(truth.JNR_dB(row)));
        testCase.verifyTrue(isnan(truth.arrival_delay_s(row)));
    end
end

function rangeTruthUsesFinalTotalDelay(testCase)
    fs = 1e6;
    reference = complex([ones(16, 1); zeros(240, 1)]);
    rng(33);
    oldSignal = Data_gen_Range_fake_target(reference, 1, fs, ...
        20e-6, 20e-6, 2, 2, 2, 1, 0, 5e-6, 5e-6);
    rng(33);
    [newSignal, truth] = Data_gen_Range_fake_target(reference, 1, fs, ...
        20e-6, 20e-6, 2, 2, 2, 1, 0, 5e-6, 5e-6);
    testCase.verifyEqual(oldSignal, newSignal);
    testCase.verifyEqual(truth.delay_s(:, 1), repmat(25e-6, 2, 1));
    testCase.verifyEqual(truth.range_m(:, 1), ...
        3e8 * truth.delay_s(:, 1) / 2);
    testCase.verifyTrue(all(isnan(truth.arrival_delay_s)));
end
```

- [ ] **Step 2: 运行新增测试并确认旧实现失败**

Run: `TestLegacyCompatibility` 整文件命令。

Expected: FAIL；旧 MFT 目标数恒为上限加一、公共时延被除以 `1e6`、两个函数没有真值输出。

- [ ] **Step 3: 实现唯一离散时延抽样器**

`delayRange_s` 和 `arrivalRange_s` 均为秒；返回结构中的采样点已按最终总时延排序：

```matlab
function selection = legacy_sample_unique_delays(delayRange_s, ...
        arrivalRange_s, targetCount, fs_Hz, sampleLength)
%LEGACY_SAMPLE_UNIQUE_DELAYS 抽取无碰撞、可记录的最终目标总时延。

    validate_range(delayRange_s, 'delayRange_s');
    validate_range(arrivalRange_s, 'arrivalRange_s');
    if ~isnumeric(targetCount) || ~isreal(targetCount) || ...
            ~isscalar(targetCount) || ~isfinite(targetCount) || ...
            targetCount < 1 || targetCount ~= fix(targetCount) || ...
            ~isnumeric(fs_Hz) || ~isscalar(fs_Hz) || ...
            ~isfinite(fs_Hz) || fs_Hz <= 0 || ...
            ~isnumeric(sampleLength) || ~isscalar(sampleLength) || ...
            ~isfinite(sampleLength) || sampleLength < 1 || ...
            sampleLength ~= fix(sampleLength)
        error('LegacyJamming:InvalidPhysicalParameter', ...
            '目标数、采样率和记录长度必须为有效标量。');
    end
    arrivalMinimum = round(arrivalRange_s(1) * fs_Hz);
    arrivalMaximum = round(arrivalRange_s(2) * fs_Hz);
    arrivalSamples = arrivalMinimum + floor( ...
        (arrivalMaximum - arrivalMinimum + 1) * rand());
    relativeMinimum = round(delayRange_s(1) * fs_Hz);
    relativeMaximum = round(delayRange_s(2) * fs_Hz);
    candidates = relativeMinimum:relativeMaximum;
    candidates = candidates(candidates >= 0 & ...
        candidates + arrivalSamples < sampleLength);
    if numel(candidates) < targetCount
        error('LegacyJamming:InsufficientDelaySlots', ...
            '可用离散时延槽少于目标数量。');
    end
    chosen = candidates(randperm(numel(candidates), targetCount));
    totalUnsorted = chosen + arrivalSamples;
    [totalSamples, sortOrder] = sort(totalUnsorted);
    selection = struct('relative_samples', chosen(sortOrder), ...
        'arrival_samples', arrivalSamples, ...
        'total_samples', totalSamples, 'sort_order', sortOrder);
end

function validate_range(value, name)
    if ~isnumeric(value) || ~isreal(value) || ...
            ~isequal(size(value), [1, 2]) || any(~isfinite(value)) || ...
            value(1) < 0 || value(1) > value(2)
        error('LegacyJamming:InvalidPhysicalParameter', ...
            '%s 必须是有限非负有序 1×2 范围。', name);
    end
end
```

- [ ] **Step 4: 迁移 MFT**

函数签名改为 `[out_sig, truth]`。请求真值时要求 `nRepetition == 1`。每次生成使用：

```matlab
n_fake_tar = randi([n_fake_tar_min, n_fake_tar_max]);
selection = legacy_sample_unique_delays([delay_min, delay_max], ...
    [jam_delay_min, jam_delay_max], n_fake_tar, fs, nSample);
amplitudes = amp_min + (amp_max - amp_min) * rand(1, n_fake_tar);
amplitudes = amplitudes(selection.sort_order);
jam_sig = complex(zeros(1, nSample));
for targetIndex = 1:n_fake_tar
    jam_sig = jam_sig + amplitudes(targetIndex) * ...
        legacy_zero_boundary_delay(sig_LOS, ...
        selection.total_samples(targetIndex));
end
```

仅当 `nargout > 1` 时要求 `nRepetition == 1` 并初始化真值；单输出调用不构建真值。真值采用矩阵槽，容量等于 `n_fake_tar_max`：

```matlab
truth = legacy_initialize_truth(sampleCount, {'JNR_dB', ...
    'num_targets', 'mean_delay_s', 'mean_spacing_s', ...
    'arrival_delay_s'});
truth.delay_s = nan(sampleCount, n_fake_tar_max);
truth.amplitude = nan(sampleCount, n_fake_tar_max);
delays_s = selection.total_samples / fs;
truth.num_targets(iBatch) = n_fake_tar;
truth.mean_delay_s(iBatch) = mean(delays_s);
if n_fake_tar > 1
    truth.mean_spacing_s(iBatch) = mean(diff(delays_s));
end
truth.delay_s(iBatch, 1:n_fake_tar) = delays_s;
truth.amplitude(iBatch, 1:n_fake_tar) = amplitudes;
```

不要给 `truth.JNR_dB` 和 `truth.arrival_delay_s` 赋有限值；它们必须保持 NaN。

- [ ] **Step 5: 迁移 RangeFakeTarget**

保持旧目标数 `n_fake_tar = 1`，使用同一抽样器合并公共时延与相对时延。仅当 `nargout > 1` 时要求 `nRepetition == 1` 并初始化固定容量 4 的 v1 真值：

```matlab
truth = legacy_initialize_truth(sampleCount, {'JNR_dB', ...
    'num_targets', 'mean_delay_s', 'arrival_delay_s'});
truth.delay_s = nan(sampleCount, 4);
truth.range_m = nan(sampleCount, 4);
truth.amplitude = nan(sampleCount, 4);
selection = legacy_sample_unique_delays([delay_min, delay_max], ...
    [jam_delay_min, jam_delay_max], 1, fs, nSample);
delay_s = selection.total_samples / fs;
amplitude = amp_min + (amp_max - amp_min) * rand();
jam_sig = amplitude * legacy_zero_boundary_delay( ...
    sig_LOS, selection.total_samples);
truth.num_targets(iBatch) = 1;
truth.mean_delay_s(iBatch) = delay_s;
truth.delay_s(iBatch, 1) = delay_s;
truth.range_m(iBatch, 1) = 3e8 * delay_s / 2;
truth.amplitude(iBatch, 1) = amplitude;
```

- [ ] **Step 6: 运行历史目标测试和整文件测试**

Run: Task 1 Step 2 的整文件命令。

Expected: 全部通过；MFT 目标数覆盖 3--5 而不是出现 6，活动延迟无重复。

- [ ] **Step 7: 提交目标干扰修复**

```powershell
git add -- "Jam9/legacy_sample_unique_delays.m" "Jam9/Data_gen_Multi_fake_target.m" "Jam9/Data_gen_Range_fake_target.m" "可配置组合干扰数据生成/tests/TestLegacyCompatibility.m"
git commit -m "fix: make legacy false-target truth observable"
```

## Task 4: 修正 Traction 初始距离、速度轨迹和幅度真值

**Files:**

- Modify: `Jam9/Data_gen_Tuoyin.m:1`
- Modify: `可配置组合干扰数据生成/tests/TestLegacyCompatibility.m`

- [ ] **Step 1: 写初始距离、速度和可选第三输出测试**

```matlab
function tractionUsesAbsoluteRangeAndReturnsStageTruth(testCase)
    fs = 1e6;
    rng(41);
    [oldSignal, oldCompressed] = Data_gen_Tuoyin(20e3, 16e-6, ...
        0, fs, 128e-6, 900, 900, 2, 2, 30, 30, 1, 4);
    rng(41);
    [signal, compressed, truth] = Data_gen_Tuoyin(20e3, 16e-6, ...
        0, fs, 128e-6, 900, 900, 2, 2, 30, 30, 1, 4);
    testCase.verifyEqual(oldSignal, signal);
    testCase.verifyEqual(oldCompressed, compressed);
    testCase.verifySize(signal, [4, 128]);
    testCase.verifySize(compressed, [4, 128]);
    testCase.verifyEqual(truth.stage_time_s, [0; 1; 4; 9]);
    testCase.verifyEqual(truth.initial_range_m, repmat(900, 4, 1));
    testCase.verifyEqual(truth.velocity_mps, repmat(30, 4, 1));
    testCase.verifyEqual(truth.nominal_amplitude, repmat(2, 4, 1));
    testCase.verifyEqual(truth.effective_amplitude, repmat(2, 4, 1));
    testCase.verifyEqual(truth.range_change_per_stage_m, ...
        30 * truth.stage_time_s);
    testCase.verifyEqual(truth.delay_change_per_stage_s, ...
        2 * truth.range_change_per_stage_m / 3e8);
    firstNonzero = arrayfun(@(row) find(abs(signal(row, :)) > 0, 1), ...
        (1:4).');
    testCase.verifyGreaterThan(firstNonzero(1), 1);
    testCase.verifyGreaterThanOrEqual(diff(firstNonzero), 0);
end

function tractionRejectsIncompleteFourStageGroups(testCase)
    testCase.verifyError(@() Data_gen_Tuoyin(20e3, 16e-6, 0, 1e6, ...
        128e-6, 900, 900, 2, 2, 30, 30, 1, 3), ...
        'LegacyJamming:InvalidPhysicalParameter');
end
```

- [ ] **Step 2: 运行测试并确认旧 Traction 失败**

Run: Task 1 Step 2 的整文件命令。

Expected: FAIL；旧函数没有第三输出，第一阶段没有初始距离延迟。

- [ ] **Step 3: 重写 Traction 的四阶段生成循环**

签名改为：

```matlab
function [out_sig, out_sig_pc, truth] = Data_gen_Tuoyin(B, Tp, freq0, ...
    fs, Tr, R_min, R_max, A_min, A_max, v_min, v_max, nBatch, nPulse)
```

使用四个明确阶段，要求总行数能被 4 整除：

```matlab
totalRows = nBatch * nPulse;
if totalRows < 4 || mod(totalRows, 4) ~= 0
    error('LegacyJamming:InvalidPhysicalParameter', ...
        'Traction 输出行数 nBatch*nPulse 必须是 4 的正整数倍。');
end
stageElapsed_s = [0, 1, 4, 9];
groupCount = totalRows / 4;
out_sig = complex(zeros(totalRows, nSample));
out_sig_pc = complex(zeros(totalRows, nSample));
truth = struct();
if nargout > 2
    truth = legacy_initialize_truth(totalRows, {'initial_range_m', ...
        'velocity_mps', 'nominal_amplitude', 'effective_amplitude', ...
        'range_change_per_stage_m', 'delay_change_per_stage_s', ...
        'stage_time_s', 'current_range_m', 'current_delay_s'});
end
```

每组采样一次 `R`、`A`、`v`，每阶段按绝对距离延迟同一参考脉冲：

```matlab
for groupIndex = 1:groupCount
    initialRange_m = R_min + (R_max - R_min) * rand();
    amplitude = A_min + (A_max - A_min) * rand();
    velocity_mps = v_min + (v_max - v_min) * rand();
    for stageIndex = 1:4
        outputRow = (groupIndex - 1) * 4 + stageIndex;
        elapsed_s = stageElapsed_s(stageIndex);
        rangeChange_m = velocity_mps * elapsed_s;
        currentRange_m = initialRange_m + rangeChange_m;
        currentDelay_s = 2 * currentRange_m / c;
        delaySamples = round(currentDelay_s * fs);
        delayedPulse = amplitude * legacy_zero_boundary_delay( ...
            sigLFM, delaySamples);
        out_sig(outputRow, :) = delayedPulse;
        out_sig_pc(outputRow, :) = ifftshift( ...
            range_process_mf(sigLFM, delayedPulse, npp));
        if nargout > 2
            truth.initial_range_m(outputRow) = initialRange_m;
            truth.velocity_mps(outputRow) = velocity_mps;
            truth.nominal_amplitude(outputRow) = amplitude;
            truth.effective_amplitude(outputRow) = amplitude;
            truth.range_change_per_stage_m(outputRow) = rangeChange_m;
            truth.delay_change_per_stage_s(outputRow) = ...
                2 * rangeChange_m / c;
            truth.stage_time_s(outputRow) = elapsed_s;
            truth.current_range_m(outputRow) = currentRange_m;
            truth.current_delay_s(outputRow) = delaySamples / fs;
        end
    end
end
```

如果 `delaySamples + Np > nSample`，抛出 `LegacyJamming:InvalidPhysicalParameter`，不得静默输出全零或回绕。

- [ ] **Step 4: 运行 Traction 测试和全部历史测试**

Run: Task 1 Step 2 的整文件命令。

Expected: 全部通过。

- [ ] **Step 5: 运行检查点 B 的定向测试**

```powershell
matlab -batch "repoRoot=pwd; appRoot=fullfile(repoRoot,'可配置组合干扰数据生成'); addpath(repoRoot,fullfile(repoRoot,'Jam9'),appRoot,fullfile(appRoot,'core'),fullfile(appRoot,'generators'),fullfile(appRoot,'tests')); results=runtests(fullfile(appRoot,'tests','TestLegacyCompatibility.m')); disp(table(results)); assertSuccess(results);"
```

Expected: TestLegacyCompatibility 全部通过。

- [ ] **Step 6: 提交 Traction 修复**

```powershell
git add -- "Jam9/Data_gen_Tuoyin.m" "可配置组合干扰数据生成/tests/TestLegacyCompatibility.m"
git commit -m "fix: apply absolute trajectory delay in legacy traction"
```

## Task 5: 强化当前可配置程序的科学语义版本和不变量

**Files:**

- Create: `可配置组合干扰数据生成/core/current_scientific_semantics_version.m`
- Modify: `可配置组合干扰数据生成/core/compose_jamming_sample.m:45`
- Modify: `可配置组合干扰数据生成/core/generate_dataset_from_manifest.m:91`
- Modify: `可配置组合干扰数据生成/core/validate_generated_dataset.m:1`
- Modify: `可配置组合干扰数据生成/core/save_dataset_bundle.m:10`
- Modify: `可配置组合干扰数据生成/tests/TestDatasetGeneration.m:333`
- Modify: `可配置组合干扰数据生成/tests/TestSingleJammingGenerators.m:1780`

- [ ] **Step 1: 写科学语义版本保存和旧版本拒绝测试**

在 `TestDatasetGeneration` 中加入：

```matlab
function bundleCarriesStableScientificSemanticsVersion(testCase)
    [cfg, registry, schema, ~, bundle] = create_small_bundle_fixture();
    expected = current_scientific_semantics_version();
    testCase.verifyEqual(bundle.metadata.scientific_semantics_version, ...
        expected);
    testCase.verifyWarningFree(@() validate_generated_dataset( ...
        bundle, cfg, registry, schema, 'balanced'));
end

function validationRejectsMissingScientificSemanticsVersion(testCase)
    [cfg, registry, schema, ~, bundle] = create_small_bundle_fixture();
    bundle.metadata = rmfield(bundle.metadata, ...
        'scientific_semantics_version');
    testCase.verifyError(@() validate_generated_dataset( ...
        bundle, cfg, registry, schema, 'balanced'), ...
        'JammingDataset:ScientificSemanticsMismatch');
end

function savedBundleRetainsScientificSemanticsVersion(testCase)
    [cfg, registry, schema, ~, bundle] = create_small_bundle_fixture();
    outputPath = [tempname, '.mat'];
    cleanup = onCleanup(@() delete_if_present(outputPath));
    save_dataset_bundle(outputPath, bundle, cfg, registry, schema);
    loaded = load(outputPath, 'metadata');
    testCase.verifyEqual(loaded.metadata.scientific_semantics_version, ...
        current_scientific_semantics_version());
end
```

在文件末尾加入：

```matlab
function delete_if_present(pathValue)
    if isfile(pathValue)
        delete(pathValue);
    end
end
```

- [ ] **Step 2: 写三组篡改检测测试**

```matlab
function validationRejectsTamperedFmbBandwidthTruth(testCase)
    [cfg, registry, schema, ~, bundle] = create_small_bundle_fixture();
    row = find(bundle.label_all(:, strcmp({registry.code}, 'FMB')), 1);
    column = strcmp(schema.field_name, 'FMB.bandwidth_factor');
    bundle.parameter_values(row, column) = ...
        bundle.parameter_values(row, column) + 0.25;
    testCase.verifyError(@() validate_generated_dataset( ...
        bundle, cfg, registry, schema, 'balanced'), ...
        'JammingDataset:ObservableTruthMismatch');
end

function validationRejectsTamperedTargetArrivalMask(testCase)
    [cfg, registry, schema, bundle] = create_single_type_bundle_fixture('MFT');
    column = strcmp(schema.field_name, 'MFT.arrival_delay_s');
    bundle.parameter_values(1, column) = 1e-6;
    bundle.parameter_mask(1, column) = true;
    testCase.verifyError(@() validate_generated_dataset( ...
        bundle, cfg, registry, schema, 'balanced'), ...
        'JammingDataset:ObservableTruthMismatch');
end

function validationRejectsTamperedTractionAmplitudeRelation(testCase)
    [cfg, registry, schema, bundle] = ...
        create_single_type_bundle_fixture('Traction');
    bundle.component_effective_amplitude{1}(1) = ...
        bundle.component_effective_amplitude{1}(1) * 2;
    testCase.verifyError(@() validate_generated_dataset( ...
        bundle, cfg, registry, schema, 'balanced'), ...
        'JammingDataset:ObservableTruthMismatch');
end
```

新增的 `create_single_type_bundle_fixture` 必须构造单模式单样本；Traction 时设置 `cfg.radar.num_pulses = 2`，其他类型保持 1。

在 `TestDatasetGeneration.m` 文件末尾加入完整夹具：

```matlab
function [cfg, registry, schema, bundle] = ...
        create_single_type_bundle_fixture(code)
    cfg = create_default_config();
    cfg.types.enabled = {code};
    cfg.types.known = {code};
    cfg.types.unknown = {};
    cfg.dataset.min_components = 1;
    cfg.dataset.max_components = 1;
    cfg.dataset.combination_mode = 'all';
    cfg.dataset.samples_per_pattern = 1;
    cfg.output.save_v73 = false;
    cfg.output.save_legacy = false;
    if strcmp(code, 'Traction')
        cfg.radar.num_pulses = 2;
    else
        cfg.radar.num_pulses = 1;
    end
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    manifest = build_balanced_manifest(cfg, registry, 'train');
    bundle = generate_dataset_from_manifest( ...
        cfg, registry, schema, manifest);
end
```

- [ ] **Step 3: 运行 Task 5 新测试并确认失败**

Run: `TestDatasetGeneration` 整文件命令。

Expected: FAIL；metadata 尚无版本字段，验证器尚不识别三类语义篡改。

- [ ] **Step 4: 实现唯一版本函数并写入 bundle metadata**

```matlab
function value = current_scientific_semantics_version()
%CURRENT_SCIENTIFIC_SEMANTICS_VERSION 返回当前数据科学语义版本。
%   该版本独立于 registry v1/v2；注册表描述字段宽度，本值描述物理真值
%   是否采用可观测离散带宽、最终总时延和 Traction 单一幅度尺度。

    value = 'observable-truth-2026-07-27';
end
```

在 `generate_dataset_from_manifest.m` 的 metadata 中加入：

```matlab
'scientific_semantics_version', current_scientific_semantics_version(), ...
```

- [ ] **Step 5: 在合成层增加分量即时语义校验**

在 `componentParams{componentIndex} = params;` 之前调用：

```matlab
validate_observable_component_params(code, params, cfg, ...
    nominalAmplitude(componentIndex), effectiveAmplitude(componentIndex), ...
    offset_dB);
```

本地校验函数必须执行：

```matlab
function validate_observable_component_params(code, params, cfg, ...
        nominalAmplitude, effectiveAmplitude, offset_dB)
    if ismember(code, {'FMB', 'AMA'})
        expected = params.interference_bandwidth_Hz / ...
            cfg.radar.bandwidth_Hz;
        if abs(params.bandwidth_factor - expected) > ...
                128 * eps(max(1, abs(expected)))
            throw_observable_truth_mismatch( ...
                [code ' 带宽因子与实际带宽不一致。']);
        end
    elseif ismember(code, {'MFT', 'RangeFakeTarget'})
        delayNames = fieldnames(params);
        delayNames = delayNames(~cellfun('isempty', ...
            regexp(delayNames, '^delay_[0-9][0-9]_s$', 'once')));
        delays = cellfun(@(name) params.(name), delayNames);
        active = delays(isfinite(delays));
        if ~isnan(params.arrival_delay_s) || ...
                any(diff(active) <= 0)
            throw_observable_truth_mismatch( ...
                [code ' 最终目标时延不可辨识或存在碰撞。']);
        end
    elseif strcmp(code, 'Traction')
        expected = nominalAmplitude * 10 ^ (offset_dB / 20);
        if ~isfinite(expected) || abs(effectiveAmplitude - expected) > ...
                128 * eps(max(1, abs(expected))) || ...
                abs(params.amplitude - effectiveAmplitude) > ...
                128 * eps(max(1, abs(expected)))
            throw_observable_truth_mismatch( ...
                'Traction 最终幅度与功率偏置不一致。');
        end
    end
end
```

将上述语义错误从功率错误中区分出来，新增本地函数：

```matlab
function throw_observable_truth_mismatch(message)
    error('JammingDataset:ObservableTruthMismatch', '%s', message);
end
```

- [ ] **Step 6: 在数据集验证层加入版本和三组不变量**

`validate_generated_dataset.m` 在结构检查后先验证：

```matlab
if ~isfield(bundle.metadata, 'scientific_semantics_version') || ...
        ~strcmp(bundle.metadata.scientific_semantics_version, ...
        current_scientific_semantics_version())
    error('JammingDataset:ScientificSemanticsMismatch', ...
        '数据缺少当前科学语义版本，不能按已修正真值解释。');
end
```

在逐样本循环中按活动代码调用本地函数，使用 schema 找到字段：

```matlab
validate_observable_parameter_row(bundle, cfg, schema, ...
    sampleIndex, codes);
```

该函数必须检查：

1. FMB/AMA 的 `bandwidth_factor = interference_bandwidth_Hz / radar.bandwidth_Hz`，且 `interference_bandwidth_Hz / (fs/sample_length)` 为正偶整数容差内的离散跨度。
2. MFT/RangeFakeTarget 的活动 `delay_XX_s` 严格递增、数量等于 `num_targets`、未使用槽 NaN/false、`arrival_delay_s` NaN/false；Range 的 `range_XX_m = c*delay_XX_s/2`。
3. Traction 的分量 `effective = nominal*10^(offset/20)`，参数列 `Traction.amplitude` 等于 effective，且 nominal JNR 为 NaN。

任何不一致统一抛出 `JammingDataset:ObservableTruthMismatch`。

- [ ] **Step 7: 强化保存函数**

`save_dataset_bundle.m` 在创建 `saveData` 前拒绝缺失或未知版本：

```matlab
if ~isfield(bundle, 'metadata') || ...
        ~isstruct(bundle.metadata) || ~isscalar(bundle.metadata) || ...
        ~isfield(bundle.metadata, 'scientific_semantics_version') || ...
        ~strcmp(bundle.metadata.scientific_semantics_version, ...
        current_scientific_semantics_version())
    error('JammingDataset:ScientificSemanticsMismatch', ...
        '只允许保存带当前科学语义版本的数据包。');
end
```

- [ ] **Step 8: 补强当前生成器回归而不改变注册表字段**

在 `TestSingleJammingGenerators` 中增加断言：

- FMB/AMA 实际带宽除以 FFT bin spacing 后是正偶整数；
- MFT/Range 的活动最终时延严格递增且 `arrival_delay_s` 为 NaN；
- Traction 的 `amplitude` 仍是注册表唯一幅度字段，不新增 JNR 或 arrival 字段。

不得向生成器 `params` 添加诊断字段，因为 `generate_single_jamming.m` 要求字段集合与注册表精确一致。

- [ ] **Step 9: 运行检查点 C 定向测试**

```powershell
matlab -batch "repoRoot=pwd; appRoot=fullfile(repoRoot,'可配置组合干扰数据生成'); addpath(repoRoot,fullfile(repoRoot,'Jam9'),appRoot,fullfile(appRoot,'core'),fullfile(appRoot,'generators'),fullfile(appRoot,'tests')); results=runtests({fullfile(appRoot,'tests','TestSingleJammingGenerators.m'),fullfile(appRoot,'tests','TestDatasetGeneration.m')}); disp(table(results)); assertSuccess(results);"
```

Expected: 两个测试类全部通过。

- [ ] **Step 10: 提交当前程序强化**

```powershell
git add -- "可配置组合干扰数据生成/core/current_scientific_semantics_version.m" "可配置组合干扰数据生成/core/compose_jamming_sample.m" "可配置组合干扰数据生成/core/generate_dataset_from_manifest.m" "可配置组合干扰数据生成/core/validate_generated_dataset.m" "可配置组合干扰数据生成/core/save_dataset_bundle.m" "可配置组合干扰数据生成/tests/TestSingleJammingGenerators.m" "可配置组合干扰数据生成/tests/TestDatasetGeneration.m"
git commit -m "feat: version and validate observable truth semantics"
```

## Task 6: 构建 v1 79 维历史真值 bundle 和旁车保存器

**Files:**

- Create: `Jam9/build_legacy_truth_bundle.m`
- Create: `Jam9/save_legacy_truth_sidecar.m`
- Modify: `可配置组合干扰数据生成/tests/TestLegacyCompatibility.m`

- [ ] **Step 1: 写组合行映射、未迁移掩码和旁车回载测试**

```matlab
function legacyBundleUsesExplicitSourceSampleIndex(testCase)
    label = logical([1, 0; 1, 1]);
    codes = {'FMB', 'MFT'};
    sourceSampleIndex = [2; 1];
    atomicTruth = struct();
    atomicTruth.FMB = make_fmb_truth([10; 20]);
    atomicTruth.MFT = make_mft_truth([1; 2]);
    metadata = struct('legacy_entrypoint', 'unit-test', ...
        'legacy_main_file', 'fixture.mat', 'nRepetition', 1);
    bundle = build_legacy_truth_bundle(label, codes, atomicTruth, ...
        sourceSampleIndex, metadata);
    fmbJnr = strcmp(bundle.parameter_schema.field_name, 'FMB.JNR_dB');
    mftCount = strcmp(bundle.parameter_schema.field_name, 'MFT.num_targets');
    testCase.verifyEqual(bundle.parameter_values(:, fmbJnr), [20; 10]);
    testCase.verifyTrue(bundle.parameter_mask(1, fmbJnr));
    testCase.verifyFalse(bundle.parameter_mask(1, mftCount));
    testCase.verifyEqual(bundle.parameter_values(2, mftCount), 1);
    testCase.verifyEqual(bundle.source_sample_index, sourceSampleIndex);
    testCase.verifySize(bundle.parameter_values, [2, 79]);
end

function unmigratedActiveTypeRemainsMasked(testCase)
    label = logical([1, 1]);
    codes = {'FMB', 'SawFM'};
    atomicTruth = struct('FMB', make_fmb_truth(30));
    metadata = struct('legacy_entrypoint', 'unit-test', ...
        'legacy_main_file', 'fixture.mat', 'nRepetition', 1);
    bundle = build_legacy_truth_bundle(label, codes, atomicTruth, 1, metadata);
    sawColumns = strcmp(bundle.parameter_schema.owner_code, 'SawFM');
    testCase.verifyTrue(all(isnan(bundle.parameter_values(1, sawColumns))));
    testCase.verifyFalse(any(bundle.parameter_mask(1, sawColumns)));
end

function invalidSourceIndexIsRejected(testCase)
    metadata = struct('legacy_entrypoint', 'unit-test', ...
        'legacy_main_file', 'fixture.mat', 'nRepetition', 1);
    atomicTruth = struct('FMB', make_fmb_truth(30));
    testCase.verifyError(@() build_legacy_truth_bundle(true, {'FMB'}, ...
        atomicTruth, 2, metadata), 'LegacyJamming:InvalidTruthShape');
end

function legacySidecarRoundTripsTopLevelVariables(testCase)
    metadata = struct('legacy_entrypoint', 'unit-test', ...
        'legacy_main_file', 'fixture.mat', 'nRepetition', 1);
    bundle = build_legacy_truth_bundle(true, {'FMB'}, ...
        struct('FMB', make_fmb_truth(30)), 1, metadata);
    tempDir = tempname;
    mkdir(tempDir);
    cleanup = onCleanup(@() rmdir(tempDir, 's'));
    mainPath = fullfile(tempDir, 'Jam4_fixture.mat');
    data = 1;
    save(mainPath, 'data');
    truthPath = save_legacy_truth_sidecar(mainPath, bundle, false);
    loaded = load(truthPath);
    testCase.verifyEqual(truthPath, ...
        fullfile(tempDir, 'Jam4_fixture_truth.mat'));
    testCase.verifyTrue(isfield(loaded, 'parameter_values'));
    testCase.verifyTrue(isfield(loaded, 'atomic_truth'));
    testCase.verifyTrue(isfield(loaded, 'metadata'));
end
```

测试本地构造器必须返回与历史函数一致的列向量字段；MFT 的 `delay_s` 和 `amplitude` 为固定容量矩阵。

在测试文件末尾加入以下完整构造器：

```matlab
function truth = make_fmb_truth(jnrValues)
    jnrValues = jnrValues(:);
    count = numel(jnrValues);
    truth = struct('JNR_dB', jnrValues, ...
        'bandwidth_factor', repmat(2, count, 1), ...
        'interference_bandwidth_Hz', repmat(10e6, count, 1), ...
        'center_frequency_Hz', zeros(count, 1), ...
        'arrival_delay_s', zeros(count, 1), ...
        'requested_bandwidth_factor', repmat(2.1, count, 1));
end

function truth = make_mft_truth(targetCounts)
    targetCounts = targetCounts(:);
    count = numel(targetCounts);
    truth = legacy_initialize_truth(count, {'JNR_dB', 'num_targets', ...
        'mean_delay_s', 'mean_spacing_s', 'arrival_delay_s'});
    truth.delay_s = nan(count, 8);
    truth.amplitude = nan(count, 8);
    for row = 1:count
        activeCount = targetCounts(row);
        delays = (1:activeCount) * 1e-6;
        truth.num_targets(row) = activeCount;
        truth.mean_delay_s(row) = mean(delays);
        if activeCount > 1
            truth.mean_spacing_s(row) = mean(diff(delays));
        end
        truth.delay_s(row, 1:activeCount) = delays;
        truth.amplitude(row, 1:activeCount) = 1:activeCount;
    end
end
```

- [ ] **Step 2: 运行测试并确认 bundle 函数缺失而失败**

Run: Task 1 Step 2 的整文件命令。

Expected: FAIL，`build_legacy_truth_bundle` 未定义。

- [ ] **Step 3: 实现 v1 注册表加载和输入校验**

`build_legacy_truth_bundle` 签名固定为：

```matlab
function bundle = build_legacy_truth_bundle(label, legacyClassCodes, ...
        atomicTruth, sourceSampleIndex, metadata)
```

函数开头保存并恢复 path，显式请求 v1：

```matlab
jam9Dir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(jam9Dir);
appRoot = fullfile(repoRoot, '可配置组合干扰数据生成');
originalPath = path;
cleanup = onCleanup(@() path(originalPath));
addpath(appRoot, fullfile(appRoot, 'core'), ...
    fullfile(appRoot, 'generators'));
registry = build_jamming_registry(struct());
schema = build_parameter_schema(registry);
if numel(registry) ~= 10 || height(schema) ~= 79
    error('LegacyJamming:InvalidLegacyLabelMap', ...
        '历史旁车必须使用 v1 十类/79 维注册表。');
end
```

输入校验必须保证：

- `label` 是 N×C logical 或 0/1 矩阵；
- `legacyClassCodes` 是长度 C、唯一、属于 v1 的 cellstr；
- `sourceSampleIndex` 是 N×1 有限正整数；
- `metadata.nRepetition == 1`；
- `atomicTruth` 是标量结构体，存在的字段名必须属于 `legacyClassCodes`。

- [ ] **Step 4: 实现每行参数编码**

预分配：

```matlab
sampleCount = size(label, 1);
parameterValues = nan(sampleCount, height(schema));
parameterMask = false(sampleCount, height(schema));
```

每个输出行只编码已迁移活动类型：

```matlab
for rowIndex = 1:sampleCount
    activeColumns = find(label(rowIndex, :) ~= 0);
    activeCodes = {};
    componentParams = {};
    for activeIndex = 1:numel(activeColumns)
        code = legacyClassCodes{activeColumns(activeIndex)};
        if ~isfield(atomicTruth, code)
            continue;
        end
        sourceRow = sourceSampleIndex(rowIndex);
        params = legacy_truth_row_to_v1(code, atomicTruth.(code), ...
            sourceRow, registry);
        activeCodes{end + 1} = code;
        componentParams{end + 1} = params;
    end
    if ~isempty(activeCodes)
        [parameterValues(rowIndex, :), parameterMask(rowIndex, :)] = ...
            encode_parameter_targets(schema, registry, ...
            activeCodes, componentParams);
    end
end
```

`legacy_truth_row_to_v1` 先将该类型注册字段全部设为 NaN，再填入可观测字段。明确映射：

- FMB/AMA：同名标量字段；忽略 `requested_bandwidth_factor`。
- MFT：`num_targets`、`mean_delay_s`、`mean_spacing_s`、`delay_01_s...delay_08_s`、`amplitude_01...amplitude_08`；`JNR_dB` 和 `arrival_delay_s` 保持 NaN。
- RangeFakeTarget：相同规则并增加 `range_01_m...range_04_m`。
- 本次三个入口未使用 Traction，因此旁车映射器只实现 FMB、AMA、MFT 和 RangeFakeTarget。若输入 `atomicTruth.Traction`，函数以 `LegacyJamming:InvalidLegacyLabelMap` 拒绝；Traction 的直接第三输出仍由 Task 4 完整提供，不伪装成 v1 脉间参数。

实现该映射的主体必须是确定性字段复制，不接受额外默认值：

```matlab
function params = legacy_truth_row_to_v1(code, truth, rowIndex, registry)
    registryIndex = find(strcmp({registry.code}, code));
    if numel(registryIndex) ~= 1
        error('LegacyJamming:InvalidLegacyLabelMap', ...
            '原子真值代码在 v1 注册表中不是唯一项。');
    end
    names = registry(registryIndex).parameter_names;
    params = struct();
    for nameIndex = 1:numel(names)
        params.(names{nameIndex}) = NaN;
    end
    switch code
        case {'FMB', 'AMA'}
            for nameIndex = 1:numel(names)
                name = names{nameIndex};
                if isfield(truth, name)
                    params.(name) = require_truth_scalar( ...
                        truth.(name), rowIndex, code, name);
                end
            end
        case 'MFT'
            scalarNames = {'num_targets', 'mean_delay_s', 'mean_spacing_s'};
            for nameIndex = 1:numel(scalarNames)
                name = scalarNames{nameIndex};
                params.(name) = require_truth_scalar( ...
                    truth.(name), rowIndex, code, name);
            end
            for slot = 1:8
                params.(sprintf('delay_%02d_s', slot)) = ...
                    truth.delay_s(rowIndex, slot);
                params.(sprintf('amplitude_%02d', slot)) = ...
                    truth.amplitude(rowIndex, slot);
            end
        case 'RangeFakeTarget'
            scalarNames = {'num_targets', 'mean_delay_s'};
            for nameIndex = 1:numel(scalarNames)
                name = scalarNames{nameIndex};
                params.(name) = require_truth_scalar( ...
                    truth.(name), rowIndex, code, name);
            end
            for slot = 1:4
                params.(sprintf('delay_%02d_s', slot)) = ...
                    truth.delay_s(rowIndex, slot);
                params.(sprintf('range_%02d_m', slot)) = ...
                    truth.range_m(rowIndex, slot);
                params.(sprintf('amplitude_%02d', slot)) = ...
                    truth.amplitude(rowIndex, slot);
            end
        otherwise
            error('LegacyJamming:InvalidLegacyLabelMap', ...
                '该类型尚未定义历史旁车映射。');
    end
end

function value = require_truth_scalar(column, rowIndex, code, name)
    if ~isnumeric(column) || ~isreal(column) || ...
            size(column, 2) ~= 1 || rowIndex > size(column, 1)
        error('LegacyJamming:InvalidTruthShape', ...
            '%s.%s 原子真值不是足够长的实数列。', code, name);
    end
    value = double(column(rowIndex));
    if ~isfinite(value) && ~isnan(value)
        error('LegacyJamming:InvalidTruthShape', ...
            '%s.%s 原子真值不能包含 Inf。', code, name);
    end
end
```

- [ ] **Step 5: 构建覆盖表和 metadata**

输出 bundle：

```matlab
bundle = struct();
bundle.parameter_values = parameterValues;
bundle.parameter_mask = parameterMask;
bundle.parameter_schema = schema;
bundle.label = logical(label);
bundle.legacy_class_codes = legacyClassCodes;
bundle.truth_coverage = coverageTable;
bundle.atomic_truth = atomicTruth;
bundle.source_sample_index = sourceSampleIndex;
metadata.scientific_semantics_version = ...
    current_scientific_semantics_version();
metadata.parameter_registry_version = 'v1';
metadata.sample_count = sampleCount;
metadata.truth_width = height(schema);
metadata.corrected_legacy_physics = true;
bundle.metadata = metadata;
```

`coverageTable` 至少有 `code`、`migrated`、`reason` 三列。存在有效 `atomicTruth.(code)` 时 `migrated=true`；否则理由为 `legacy generator has no migrated truth output`。

- [ ] **Step 6: 实现旁车保存器**

```matlab
function truthPath = save_legacy_truth_sidecar(mainFilePath, bundle, saveV73)
%SAVE_LEGACY_TRUTH_SIDECAR 保存与历史主文件同 stem 的真值旁车。

    if nargin < 3
        saveV73 = true;
    end
    if ~(ischar(mainFilePath) && isrow(mainFilePath)) && ...
            ~(isstring(mainFilePath) && isscalar(mainFilePath))
        error('LegacyJamming:InvalidTruthShape', ...
            'mainFilePath 必须是文本标量。');
    end
    mainFilePath = char(mainFilePath);
    required = {'parameter_values', 'parameter_mask', 'parameter_schema', ...
        'label', 'legacy_class_codes', 'truth_coverage', 'atomic_truth', ...
        'source_sample_index', 'metadata'};
    if ~isstruct(bundle) || ~isscalar(bundle) || ...
            ~all(isfield(bundle, required)) || ...
            size(bundle.parameter_values, 2) ~= 79 || ...
            ~isequal(size(bundle.parameter_values), size(bundle.parameter_mask)) || ...
            size(bundle.parameter_values, 1) ~= size(bundle.label, 1)
        error('LegacyJamming:InvalidTruthShape', ...
            '旁车 bundle 字段、行数或 79 维宽度不合法。');
    end
    [parent, stem, extension] = fileparts(mainFilePath);
    if ~strcmpi(extension, '.mat')
        error('LegacyJamming:InvalidTruthShape', ...
            '历史主文件扩展名必须为 .mat。');
    end
    truthPath = fullfile(parent, [stem '_truth.mat']);
    saveData = bundle;
    if saveV73
        save(truthPath, '-struct', 'saveData', '-v7.3');
    else
        save(truthPath, '-struct', 'saveData');
    end
    if ~isfile(truthPath)
        error('LegacyJamming:SidecarSaveFailed', ...
            '真值旁车文件未成功创建。');
    end
end
```

- [ ] **Step 7: 运行 Task 6 测试并确认通过**

Run: Task 1 Step 2 的整文件命令。

Expected: 所有历史测试通过，旁车宽度严格为 79。

- [ ] **Step 8: 提交旁车核心**

```powershell
git add -- "Jam9/build_legacy_truth_bundle.m" "Jam9/save_legacy_truth_sidecar.m" "可配置组合干扰数据生成/tests/TestLegacyCompatibility.m"
git commit -m "feat: build v1 truth sidecars for legacy datasets"
```

## Task 7: 接入 Gen_com、SP_gen_comjam 和 Gen_openjamset

**Files:**

- Modify: `组合干扰生成/Gen_com.m:1,76,83,118,247,279`
- Modify: `组合干扰生成/SP_gen_comjam.m:1,60,67,102,123,156`
- Modify: `Gen_openjamset.m:1,55,62,97,117,161,199,237,274`
- Modify: `可配置组合干扰数据生成/tests/TestLegacyCompatibility.m`

- [ ] **Step 1: 写入口接线静态检查和五组开放集映射测试**

```matlab
function legacyEntrypointsDeclareTruthBundleAndSidecar(testCase)
    repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    paths = {fullfile(repoRoot, '组合干扰生成', 'Gen_com.m'), ...
        fullfile(repoRoot, '组合干扰生成', 'SP_gen_comjam.m'), ...
        fullfile(repoRoot, 'Gen_openjamset.m')};
    for index = 1:numel(paths)
        source = fileread(paths{index});
        testCase.verifyNotEmpty(strfind(source, ...
            'build_legacy_truth_bundle'));
        testCase.verifyNotEmpty(strfind(source, ...
            'save_legacy_truth_sidecar'));
        testCase.verifyNotEmpty(strfind(source, ...
            'source_sample_index'));
    end
end

function openSetEntrypointDeclaresFiveExplicitClassMaps(testCase)
    repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    source = fileread(fullfile(repoRoot, 'Gen_openjamset.m'));
    expectedMaps = { ...
        "{'AMA','SawFM','MFT','ISRJ'}", ...
        "{'FMB','SawFM','MFT','ISRJ'}", ...
        "{'FMB','AMA','MFT','ISRJ'}", ...
        "{'FMB','AMA','SawFM','ISRJ'}", ...
        "{'FMB','AMA','SawFM','MFT'}"};
    compactSource = regexprep(source, '\s+', '');
    for index = 1:numel(expectedMaps)
        testCase.verifyNotEmpty(strfind(compactSource, expectedMaps{index}));
    end
end
```

- [ ] **Step 2: 运行入口测试并确认失败**

Run: Task 1 Step 2 的整文件命令。

Expected: FAIL；三个脚本尚未调用旁车构建/保存函数。

- [ ] **Step 3: 给三个入口增加相对脚本位置的 path 引导**

`Gen_com.m` 和 `SP_gen_comjam.m` 在开头 `clear` 后加入：

```matlab
scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
originalPath = path;
pathCleanup = onCleanup(@() path(originalPath));
addpath(fullfile(repoRoot, 'Jam9'), ...
    fullfile(repoRoot, '可配置组合干扰数据生成'), ...
    fullfile(repoRoot, '可配置组合干扰数据生成', 'core'), ...
    fullfile(repoRoot, '可配置组合干扰数据生成', 'generators'));
```

根目录 `Gen_openjamset.m` 使用：

```matlab
repoRoot = fileparts(mfilename('fullpath'));
originalPath = path;
pathCleanup = onCleanup(@() path(originalPath));
addpath(fullfile(repoRoot, 'Jam9'), ...
    fullfile(repoRoot, '可配置组合干扰数据生成'), ...
    fullfile(repoRoot, '可配置组合干扰数据生成', 'core'), ...
    fullfile(repoRoot, '可配置组合干扰数据生成', 'generators'));
```

- [ ] **Step 4: 将三类已使用核心函数改为请求真值**

三个脚本中统一改为：

将三个现有完整调用的左值分别从 `Jam_FMB =`、`Jam_AMA =`、`Jam_MFT =` 改为 `[Jam_FMB, truth_FMB] =`、`[Jam_AMA, truth_AMA] =`、`[Jam_MFT, truth_MFT] =`，参数列表逐字保留。随后加入：

```matlab
atomic_truth = struct('FMB', truth_FMB, 'AMA', truth_AMA, ...
    'MFT', truth_MFT);
```

保留 SawFM、SinFM、SN、VFT、ISRJ 原有单输出调用。它们在旁车中属于活动但未迁移类型，参数列必须保持 NaN/false。

- [ ] **Step 5: 接入 `Gen_com.m` 的 4 类组合旁车**

使用精确类序：

```matlab
legacy_class_codes = {'FMB','SawFM','MFT','ISRJ'};
source_sample_index = repmat((1:nBatch).', size(zuhe_one_hot, 1), 1);
mainFilePath = 'D:\干扰数据\复合干扰数据\Jam4_zh0_tttt.mat';
labelFilePath = 'D:\干扰数据\复合干扰数据\label4_zh0_tttt.mat';
metadata = struct('legacy_entrypoint', '组合干扰生成/Gen_com.m', ...
    'legacy_main_file', mainFilePath, 'nRepetition', nRepetition);
truth_bundle = build_legacy_truth_bundle(label, legacy_class_codes, ...
    atomic_truth, source_sample_index, metadata);
save(mainFilePath, 'data', '-v7.3');
save(labelFilePath, 'label', '-v7.3');
save_legacy_truth_sidecar(mainFilePath, truth_bundle, true);
```

路径字符串必须沿用文件中的现有实际路径文本；如果文件当前编码导致显示乱码，只把原字符串赋给变量，不改动磁盘目标目录。

编辑三个历史脚本时保持原文件编码和未触及行的字节内容，避免把已有中文注释整文件重编码。

- [ ] **Step 6: 接入 `SP_gen_comjam.m` 的 7 类组合旁车**

```matlab
legacy_class_codes = ...
    {'FMB','SawFM','SinFM','SN','MFT','VFT','ISRJ'};
source_sample_index = repmat((1:nBatch).', size(zuhe_one_hot, 1), 1);
metadata = struct('legacy_entrypoint', ...
    '组合干扰生成/SP_gen_comjam.m', ...
    'legacy_main_file', mainFilePath, 'nRepetition', nRepetition);
truth_bundle = build_legacy_truth_bundle(label, legacy_class_codes, ...
    atomic_truth, source_sample_index, metadata);
save_legacy_truth_sidecar(mainFilePath, truth_bundle, true);
```

FMB 与 MFT 有效；其余活动类型保持 NaN/false。虽然 AMA 已生成真值，但当前 7 类数组未包含 AMA，因此不得写入任何 AMA 参数列。

- [ ] **Step 7: 接入 `Gen_openjamset.m` 的五个旁车**

五段组合按原顺序分别使用：

```matlab
legacy_class_codes_1 = {'AMA','SawFM','MFT','ISRJ'};
legacy_class_codes_2 = {'FMB','SawFM','MFT','ISRJ'};
legacy_class_codes_3 = {'FMB','AMA','MFT','ISRJ'};
legacy_class_codes_4 = {'FMB','AMA','SawFM','ISRJ'};
legacy_class_codes_5 = {'FMB','AMA','SawFM','MFT'};
```

每段在其 `zuhe_one_hot` 生成后建立独立：

```matlab
source_sample_index = repmat((1:nBatch).', size(zuhe_one_hot, 1), 1);
metadata = struct('legacy_entrypoint', 'Gen_openjamset.m', ...
    'legacy_main_file', mainFilePath, 'nRepetition', nRepetition, ...
    'open_set_partition_index', partitionIndex);
truth_bundle = build_legacy_truth_bundle(label, legacy_class_codes, ...
    atomic_truth, source_sample_index, metadata);
save_legacy_truth_sidecar(mainFilePath, truth_bundle, true);
```

分别生成 `Jam5_1_truth.mat` 至 `Jam5_5_truth.mat`；原 `Jam5_i.mat` 和 `label5_i.mat` 继续按原文件名保存。

- [ ] **Step 8: 运行入口静态测试和旁车功能测试**

Run: Task 1 Step 2 的整文件命令。

Expected: 全部通过；不执行三个 400 样本正式脚本。

- [ ] **Step 9: 人工检查入口差异不含数据参数漂移**

```powershell
git diff -- "组合干扰生成/Gen_com.m" "组合干扰生成/SP_gen_comjam.m" "Gen_openjamset.m"
```

Expected: 原雷达参数、JNR 范围、组合循环、`data` 和 `label` 结构未被改写；差异只包含 path 引导、可选真值输出、显式类映射、原子行索引和旁车保存。

- [ ] **Step 10: 提交三个入口接线**

```powershell
git add -- "组合干扰生成/Gen_com.m" "组合干扰生成/SP_gen_comjam.m" "Gen_openjamset.m" "可配置组合干扰数据生成/tests/TestLegacyCompatibility.m"
git commit -m "feat: save truth sidecars from legacy entrypoints"
```

## Task 8: 完整验证、说明文件和最终复核

**Files:**

- Create: `历史组合干扰程序真值旁车说明.md`
- Modify only if verification exposes defects: files already listed in Tasks 1--7

- [ ] **Step 1: 写程序说明文件**

说明文件必须包含以下完整章节：

```markdown
# 历史组合干扰程序真值旁车说明

## 1. 本次修正的三个核心问题
- FMB/AMA 离散频谱掩膜和实际带宽真值
- MFT/RangeFakeTarget 最终总时延、目标数和延迟碰撞
- Traction 初始距离、速度轨迹和单一幅度尺度

## 2. 保持兼容的旧接口
列出五个函数的旧调用和可选真值调用。

## 3. `_truth.mat` 顶层变量
逐项解释 parameter_values、parameter_mask、parameter_schema、label、
legacy_class_codes、truth_coverage、atomic_truth、source_sample_index、metadata。

## 4. 三个入口的类别顺序与真值覆盖
分别列出 Gen_com、SP_gen_comjam 和 Gen_openjamset 五段类序。

## 5. 科学语义版本
说明 observable-truth-2026-07-27 与 registry v1/v2 的区别。

## 6. 使用限制
说明未迁移类型为 NaN/false、旧无版本 MAT 不得当作修正数据、
本次测试未正式生成或覆盖数据集。

## 7. 验证命令
给出 MATLAB 全测试命令和旁车 load 示例。
```

- [ ] **Step 2: 运行全量 MATLAB 测试**

Run: 本计划“1. 统一测试命令”的完整回归命令。

Expected: 原基线 231 项和全部新增测试均通过，0 failed、0 incomplete。

- [ ] **Step 3: 运行 Git 与占位符检查**

```powershell
git diff --check
$redFlags = @('TO'+'DO', 'TB'+'D', 'place'+'holder', '待'+'补充', '待'+'定')
rg -n ($redFlags -join '|') "Jam9" "可配置组合干扰数据生成" "组合干扰生成" "Gen_openjamset.m" "历史组合干扰程序真值旁车说明.md"
git status --short
```

Expected: `git diff --check` 无输出；占位符扫描无新增命中；状态只包含本计划授权的文件。

- [ ] **Step 4: 验证未生成正式数据文件**

```powershell
git status --short | Select-String -Pattern '\.mat$|复合干扰数据0716'
```

Expected: 无输出。测试产生的 MAT 文件只存在于 MATLAB 临时目录并已清理。

- [ ] **Step 5: 复核五个核心兼容接口**

```powershell
rg -n "^function" "Jam9/Data_gen_NoiseFM_Block.m" "Jam9/Data_gen_NoiseAM_Spot.m" "Jam9/Data_gen_Multi_fake_target.m" "Jam9/Data_gen_Range_fake_target.m" "Jam9/Data_gen_Tuoyin.m"
```

Expected: 前四个函数为 `[out_sig, truth]`，Traction 为 `[out_sig, out_sig_pc, truth]`；输入参数顺序与改造前一致。

- [ ] **Step 6: 提交说明和最终测试修正**

```powershell
git add -- "历史组合干扰程序真值旁车说明.md"
git commit -m "docs: explain legacy observable truth sidecars"
```

- [ ] **Step 7: 最终提交历史与工作树检查**

```powershell
git status --short
git log --oneline --decorate -10
```

Expected: 工作树干净；历史中包含各检查点的独立提交，便于逐批审查或回退。

## 结束条件

只有同时满足以下条件才可宣布实现完成：

1. 全量 MATLAB 测试通过，失败数为 0。
2. 三个历史入口仍保存原主文件与标签文件，并新增对应旁车。
3. `parameter_values`/`parameter_mask` 严格为 N×79，组合行通过 `source_sample_index` 对齐原子样本。
4. 未迁移类型保持 NaN/false，`truth_coverage` 明确报告缺口。
5. 当前可配置程序保存并验证 `observable-truth-2026-07-27`。
6. 没有正式数据集被测试生成、覆盖或纳入 Git。
7. `历史组合干扰程序真值旁车说明.md` 已与最终实现一致。
