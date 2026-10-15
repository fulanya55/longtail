# Three New Jamming Types Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the configurable compound-jamming generator with versioned 10-class/v1 and 13-class/v2 registries plus physically validated CombSpectrum, ClutterPulse, and standard integer-subpulse SMSP generators.

**Architecture:** Keep all existing four/five/six-class configurations on the exact v1 registry and make the no-argument default a v2 configuration. Add one shared zero-centered-reference helper, three isolated atomic generators, version-aware normalization and fixed truth schemas, then verify balanced, imbalanced, and open-set composition through the existing manifest and power-calibration pipeline.

**Tech Stack:** MATLAB, `matlab.unittest`, complex-baseband LFM simulation, MAT v7.3/HDF5, PowerShell, Git.

---

## File Map

### Create

- `可配置组合干扰数据生成/core/build_centered_reference_signal.m` — derive a zero-centered complex-baseband reference and realized discrete support for the three new generators.
- `可配置组合干扰数据生成/generators/generate_comb_spectrum.m` — generate 3–8 exact LFM comb teeth with equal/random amplitude modes.
- `可配置组合干扰数据生成/generators/generate_clutter_pulse.m` — generate independent or repeated random gated-LFM pulse layouts with 32 fixed truth slots.
- `可配置组合干扰数据生成/generators/generate_spectral_dispersion.m` — generate standard SMSP with integer subpulse count `M=2..8`, full reference bandwidth per subpulse, and coherent repetition.
- `可配置组合干扰数据生成/README_新增三类干扰.md` — Chinese usage, parameter, truth, compatibility, and physical-model documentation.

### Modify

- `可配置组合干扰数据生成/create_default_config.m` — add optional registry version, v2 defaults, and the three new parameter blocks while preserving exact v1 shape.
- `可配置组合干扰数据生成/build_jamming_registry.m` — resolve v1/v2 and append stable IDs 11–13 with 30/169/17 truth fields.
- `可配置组合干扰数据生成/create_closedset_6class_config.m` — explicitly request the v1 template; four/five-class and power-sweep wrappers inherit this unchanged.
- `可配置组合干扰数据生成/core/normalize_and_validate_config.m` — normalize registry version and validate only enabled, version-owned parameters including capacities and physical support.
- `可配置组合干扰数据生成/core/generate_single_jamming.m` — allow only the new fixed-slot names to use `NaN` padding.
- `可配置组合干扰数据生成/README_使用说明.md` — explain v1/v2 selection and editable known/unknown lists.
- `可配置组合干扰数据生成/程序说明_组合干扰数据生成系统.md` — update architecture, label counts, and parameter-schema sizes.
- `可配置组合干扰数据生成/tests/TestConfigAndRegistry.m` — versioning, defaults, schema size, and physical-validation tests.
- `可配置组合干扰数据生成/tests/TestSingleJammingGenerators.m` — centered-reference and atomic waveform/truth tests.
- `可配置组合干扰数据生成/tests/TestManifestBuilders.m` — v2 type selection and imbalance/open-set manifest tests.
- `可配置组合干扰数据生成/tests/TestDatasetGeneration.m` — v1 readback compatibility and v2 end-to-end dataset tests.

### Preserve

- Existing IDs 1–10, their parameter ordering, and v1 schema size 79.
- Existing `G:\论文相关\干扰数据\复合干扰数据0716` MAT files and all current generation outputs.
- Existing four/five/six-class selected types, seeds, JNR profiles, sample counts, paths, and resume semantics.

---

### Task 1: Establish and Record the Clean Baseline

**Files:**
- Read: `可配置组合干扰数据生成/tests/*.m`
- Read: `docs/superpowers/specs/2026-07-25-three-new-jamming-types-design.md`

- [ ] **Step 1: Verify branch and worktree state**

Run:

```powershell
git status --short --branch
git branch --show-current
```

Expected: branch `codex/four-five-class-datasets`; only intentional plan progress may be present.

- [ ] **Step 2: Run the complete existing MATLAB baseline**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); r=runtests(fullfile(root,'tests')); fprintf('BASELINE_TOTAL=%d BASELINE_FAILED=%d\n',numel(r),sum([r.Failed])); assertSuccess(r);"
```

Expected: `BASELINE_TOTAL=200 BASELINE_FAILED=0`.

- [ ] **Step 3: Confirm no data-generation command is running**

```powershell
Get-Process matlab -ErrorAction SilentlyContinue
```

Expected: no unexpected formal dataset generation process. Do not delete, rename, or overwrite any existing MAT file.

---

### Task 2: Add the Shared Zero-Centered Reference Helper

**Files:**
- Create: `可配置组合干扰数据生成/core/build_centered_reference_signal.m`
- Test: `可配置组合干扰数据生成/tests/TestSingleJammingGenerators.m`

- [ ] **Step 1: Write failing centered-reference tests**

Add these test methods to `TestSingleJammingGenerators`:

```matlab
function centeredReferencePreservesEnvelopeAndRecentersSupport(testCase)
    cfg = normalize_and_validate_config(create_default_config(), 'balanced');
    reference = build_reference_signal(cfg);
    centered = build_centered_reference_signal(cfg, reference);
    active = 1:cfg.derived.samples_per_pulse;

    testCase.verifySize(centered.pulse, size(reference.pulse));
    testCase.verifySize(centered.train, size(reference.train));
    testCase.verifyEqual(abs(centered.pulse(active)), ...
        abs(reference.pulse(active)), 'AbsTol', 64 * eps);
    testCase.verifyEqual(mean(centered.support_Hz), 0, ...
        'AbsTol', cfg.radar.fs_Hz * 64 * eps);
    testCase.verifyLessThan(centered.support_Hz(2), cfg.radar.fs_Hz / 2);
    testCase.verifyGreaterThan(centered.support_Hz(1), -cfg.radar.fs_Hz / 2);
end

function centeredReferenceRepeatsExactlyAcrossPris(testCase)
    cfg = normalize_and_validate_config(create_default_config(), 'balanced');
    reference = build_reference_signal(cfg);
    centered = build_centered_reference_signal(cfg, reference);
    blocks = reshape(centered.train, cfg.derived.samples_per_pri, []).';
    for pulseIndex = 1:cfg.radar.num_pulses
        testCase.verifyEqual(blocks(pulseIndex, :), centered.pulse);
    end
end
```

- [ ] **Step 2: Run the focused tests and verify failure**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestSingleJammingGenerators.m'),'Name','*centeredReference*'); disp(r); assert(any([r.Failed]));"
```

Expected: failure because `build_centered_reference_signal` does not exist.

- [ ] **Step 3: Create the centered-reference implementation**

Create `core/build_centered_reference_signal.m` with this interface and algorithm:

```matlab
function centered = build_centered_reference_signal(cfg, reference)
%BUILD_CENTERED_REFERENCE_SIGNAL 将现有离散LFM搬移到零中心复基带。

    fs_Hz = cfg.radar.fs_Hz;
    samplesPerPulse = cfg.derived.samples_per_pulse;
    samplesPerPri = cfg.derived.samples_per_pri;
    active = complex(reference.pulse(1:samplesPerPulse));
    if samplesPerPulse < 2
        error('JammingGenerator:InvalidPhysicalParameter', ...
            '零中心参考LFM至少需要两个有效样点。');
    end

    adjacentPhase = unwrap(angle(active(2:end) .* conj(active(1:end - 1))));
    frequency_Hz = adjacentPhase * fs_Hz / (2 * pi);
    originalSupport_Hz = [min(frequency_Hz), max(frequency_Hz)];
    midpoint_Hz = mean(originalSupport_Hz);
    timePulse_s = (0:samplesPerPulse - 1) / fs_Hz;
    centeredActive = active .* exp(-1i * 2 * pi * midpoint_Hz * timePulse_s);

    centeredPulse = complex(zeros(1, samplesPerPri));
    centeredPulse(1:samplesPerPulse) = centeredActive;
    centeredTrain = complex(repmat(centeredPulse, 1, cfg.radar.num_pulses));
    centered = struct( ...
        'pulse', centeredPulse, ...
        'train', centeredTrain, ...
        'support_Hz', originalSupport_Hz - midpoint_Hz, ...
        'frequency_shift_Hz', -midpoint_Hz);
end
```

Include complete Chinese comments explaining adjacent-sample instantaneous frequency, why the midpoint rather than nominal `B/2` is used, and why the helper consumes no random numbers.

- [ ] **Step 4: Run the focused tests**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestSingleJammingGenerators.m'),'Name','*centeredReference*'); assertSuccess(r);"
```

Expected: both centered-reference tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- '可配置组合干扰数据生成/core/build_centered_reference_signal.m' '可配置组合干扰数据生成/tests/TestSingleJammingGenerators.m'
git commit -m "feat: add centered LFM reference helper"
```

---

### Task 3: Implement CombSpectrum with Exact Tooth Truth

**Files:**
- Create: `可配置组合干扰数据生成/generators/generate_comb_spectrum.m`
- Test: `可配置组合干扰数据生成/tests/TestSingleJammingGenerators.m`

- [ ] **Step 1: Write failing direct-generator tests**

Add a local fixture that appends only the CombSpectrum block to a normalized configuration, then add:

```matlab
function combSpectrumEqualModeHasExactTeethAndPadding(testCase)
    cfg = create_new_type_fixture();
    cfg.params.CombSpectrum = struct( ...
        'num_teeth', [3, 3], 'spacing_Hz', [0.4e6, 0.4e6], ...
        'center_offset_Hz', [0, 0], 'amplitude', [0.5, 1.5], ...
        'amplitude_mode', 'equal', 'max_teeth', 8);
    reference = build_reference_signal(cfg);
    [signal, truth] = generate_comb_spectrum( ...
        cfg, RandStream('mt19937ar','Seed',31), reference);

    testCase.verifySize(signal, [1, cfg.derived.sample_length]);
    testCase.verifyEqual(truth.num_teeth, 3);
    testCase.verifyEqual([truth.frequency_01_Hz, ...
        truth.frequency_02_Hz, truth.frequency_03_Hz], ...
        [-0.4e6, 0, 0.4e6], 'AbsTol', 1e-9);
    testCase.verifyEqual([truth.amplitude_01, truth.amplitude_02, ...
        truth.amplitude_03], [1, 1, 1]);
    testCase.verifyTrue(isnan(truth.frequency_04_Hz));
    testCase.verifyTrue(isnan(truth.phase_08_rad));
end

function combSpectrumRandomModeIsSeedReproducible(testCase)
    cfg = create_new_type_fixture();
    cfg.params.CombSpectrum = struct( ...
        'num_teeth', [8, 8], 'spacing_Hz', [0.2e6, 0.6e6], ...
        'center_offset_Hz', [-0.3e6, 0.3e6], ...
        'amplitude', [0.5, 1.5], 'amplitude_mode', 'random', ...
        'max_teeth', 8);
    reference = build_reference_signal(cfg);
    [a, ta] = generate_comb_spectrum(cfg, ...
        RandStream('mt19937ar','Seed',32), reference);
    [b, tb] = generate_comb_spectrum(cfg, ...
        RandStream('mt19937ar','Seed',32), reference);
    testCase.verifyEqual(a, b);
    testCase.verifyEqual(ta, tb);
    testCase.verifyGreaterThanOrEqual(ta.amplitude_01, 0.5);
    testCase.verifyLessThanOrEqual(ta.amplitude_08, 1.5);
end
```

Add this local fixture below the test class; it fixes common JNR and arrival
delay and never calls global `rng`:

```matlab
function cfg = create_new_type_fixture()
    cfg = create_default_config();
    cfg.params.common.JNR_dB = [25, 25];
    cfg.params.common.arrival_delay_s = [0, 0];
    cfg = normalize_and_validate_config(cfg, 'balanced');
end
```

- [ ] **Step 2: Run the focused tests and verify failure**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestSingleJammingGenerators.m'),'Name','*combSpectrum*'); disp(r); assert(any([r.Failed]));"
```

Expected: failure because `generate_comb_spectrum` does not exist.

- [ ] **Step 3: Implement the physical comb generator**

Create `generators/generate_comb_spectrum.m`. The main body must implement this exact sequence:

```matlab
capacity = 8;
centered = build_centered_reference_signal(cfg, reference);
JNR_dB = sample_range(cfg.params.common.JNR_dB, stream);
numTeeth = sample_integer(cfg.params.CombSpectrum.num_teeth, stream);
outerIndex = (numTeeth - 1) / 2;
guard_Hz = cfg.radar.fs_Hz / 2 - max( ...
    eps(cfg.radar.fs_Hz / 2), cfg.radar.fs_Hz * 1e-12);

supportMinimum_Hz = centered.support_Hz(1);
supportMaximum_Hz = centered.support_Hz(2);
spacingRange = cfg.params.CombSpectrum.spacing_Hz;
centerRange = cfg.params.CombSpectrum.center_offset_Hz;
if outerIndex > 0
    spacingUpper_Hz = min([spacingRange(2), ...
        (centerRange(2) + guard_Hz + supportMinimum_Hz) / outerIndex, ...
        (guard_Hz - supportMaximum_Hz - centerRange(1)) / outerIndex]);
else
    spacingUpper_Hz = spacingRange(2);
end
if spacingUpper_Hz < spacingRange(1)
    throw_invalid_physical_parameter('CombSpectrum没有可行的齿间隔和中心偏移组合。');
end
spacing_Hz = sample_range([spacingRange(1), spacingUpper_Hz], stream);
lowerCenter_Hz = max(centerRange(1), ...
    -guard_Hz - supportMinimum_Hz + outerIndex * spacing_Hz);
upperCenter_Hz = min(centerRange(2), ...
    guard_Hz - supportMaximum_Hz - outerIndex * spacing_Hz);
centerOffset_Hz = sample_range([lowerCenter_Hz, upperCenter_Hz], stream);
frequencies_Hz = centerOffset_Hz + ...
    ((1:numTeeth) - (numTeeth + 1) / 2) * spacing_Hz;
```

Then sample one phase per tooth; set amplitudes to all ones for `equal` or sample each amplitude for `random`; sum

```matlab
amplitudes(k) * centered.train .* ...
    exp(1i * (2*pi*frequencies_Hz(k)*reference.time_train_s + phases(k)))
```

for every active tooth. Apply the common arrival delay with a local zero-fill helper, never `circshift`. Return the six summary fields followed by `frequency_01_Hz..08`, `amplitude_01..08`, and `phase_01_rad..08`; inactive fields are `NaN`.

Add local validators for fixed capacity 8, integer `num_teeth`, positive spacing/amplitude ranges, supported mode strings, delay support, and final shifted support strictly inside the guard. All failures use `JammingGenerator:InvalidPhysicalParameter`, except an unsupported mode uses `JammingGenerator:UnsupportedMode`.

- [ ] **Step 4: Add dechirped-FFT and zero-boundary tests**

Add tests that dechirp the zero-delay comb by `conj(centered.train)`, verify peak separations equal saved `spacing_Hz` within one FFT bin, and verify a positive arrival delay creates leading zeros without copying tail samples to the front.

- [ ] **Step 5: Run all CombSpectrum tests**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestSingleJammingGenerators.m'),'Name','*combSpectrum*'); assertSuccess(r);"
```

Expected: equal/random modes, 3/8 boundaries, FFT spacing, support guard, padding, and zero-delay semantics all pass.

- [ ] **Step 6: Commit**

```powershell
git add -- '可配置组合干扰数据生成/generators/generate_comb_spectrum.m' '可配置组合干扰数据生成/tests/TestSingleJammingGenerators.m'
git commit -m "feat: add comb spectrum generator"
```

---

### Task 4: Implement ClutterPulse Independent and Repeated Layouts

**Files:**
- Create: `可配置组合干扰数据生成/generators/generate_clutter_pulse.m`
- Test: `可配置组合干扰数据生成/tests/TestSingleJammingGenerators.m`

- [ ] **Step 1: Write failing fixed-layout tests**

Add these test methods with `num_pulses=4`, zero common delay, and a fixed
three-pulse-per-PRI range:

```matlab
function clutterPulseRepeatedModeCopiesLayoutAcrossPris(testCase)
    cfg = create_new_type_fixture();
    cfg.params.ClutterPulse = struct( ...
        'pulses_per_pri', [3, 3], 'pulse_width_s', [4e-6, 8e-6], ...
        'relative_amplitude', [0.5, 1.5], ...
        'frequency_offset_Hz', [-1e6, 1e6], ...
        'mode', 'repeated', 'max_pulses_per_pri', 8, ...
        'max_total_pulses', 32);
    reference = build_reference_signal(cfg);
    [~, truth] = generate_clutter_pulse(cfg, ...
        RandStream('mt19937ar','Seed',41), reference);

    testCase.verifyEqual(truth.mode_code, 1);
    testCase.verifyEqual(truth.total_pulse_count, 12);
    for priIndex = 2:cfg.radar.num_pulses
        offset = (priIndex - 1) * 3;
        for localIndex = 1:3
            base = localIndex;
            current = offset + localIndex;
            testCase.verifyEqual(read_truth(truth,'width',current,'_s'), ...
                read_truth(truth,'width',base,'_s'));
            testCase.verifyEqual(read_truth(truth,'amplitude',current,''), ...
                read_truth(truth,'amplitude',base,''));
            testCase.verifyEqual(read_truth(truth,'frequency',current,'_Hz'), ...
                read_truth(truth,'frequency',base,'_Hz'));
            testCase.verifyEqual(read_truth(truth,'phase',current,'_rad'), ...
                read_truth(truth,'phase',base,'_rad'));
            testCase.verifyEqual( ...
                read_truth(truth,'start',current,'_s') - ...
                read_truth(truth,'start',base,'_s'), ...
                (priIndex - 1) * cfg.radar.pri_s, 'AbsTol', 1e-12);
        end
    end
end

function clutterPulseIndependentModeChangesFixedSeedLayouts(testCase)
    cfg = create_new_type_fixture();
    cfg.params.ClutterPulse = struct( ...
        'pulses_per_pri', [3, 3], 'pulse_width_s', [4e-6, 8e-6], ...
        'relative_amplitude', [0.5, 1.5], ...
        'frequency_offset_Hz', [-1e6, 1e6], ...
        'mode', 'independent', 'max_pulses_per_pri', 8, ...
        'max_total_pulses', 32);
    reference = build_reference_signal(cfg);
    [~, truth] = generate_clutter_pulse(cfg, ...
        RandStream('mt19937ar','Seed',42), reference);
    firstStarts = [truth.start_01_s, truth.start_02_s, truth.start_03_s];
    secondStarts = [truth.start_04_s, truth.start_05_s, truth.start_06_s] ...
        - cfg.radar.pri_s;
    testCase.verifyNotEqual(firstStarts, secondStarts);
end
```

Add the local test helper:

```matlab
function value = read_truth(truth, prefix, index, suffix)
    value = truth.(sprintf('%s_%02d%s', prefix, index, suffix));
end
```

- [ ] **Step 2: Run the focused tests and verify failure**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestSingleJammingGenerators.m'),'Name','*clutterPulse*'); disp(r); assert(any([r.Failed]));"
```

Expected: failure because `generate_clutter_pulse` does not exist.

- [ ] **Step 3: Implement layout sampling and coherent gated-LFM summation**

Create `generators/generate_clutter_pulse.m` with constants:

```matlab
capacityPerPri = 8;
totalCapacity = 32;
```

Validate `max_pulses_per_pri==8`, `max_total_pulses==32`, and
`cfg.radar.num_pulses * pulses_per_pri(2) <= 32`. Quantize one common arrival
delay before sampling layouts. For each pulse, use:

```matlab
widthSamples = max(1, round(sample_range( ...
    cfg.params.ClutterPulse.pulse_width_s, stream) * fs_Hz));
latestStart = samplesPerPri - widthSamples - arrivalDelaySamples;
if latestStart < 0
    throw_invalid_physical_parameter( ...
        'ClutterPulse脉宽与公共到达时延无法完整位于PRI内。');
end
intraStartSamples = randi(stream, [0, latestStart], 1, 1);
amplitude = sample_range( ...
    cfg.params.ClutterPulse.relative_amplitude, stream);
frequency_Hz = sample_range( ...
    cfg.params.ClutterPulse.frequency_offset_Hz, stream);
phase_rad = 2 * pi * rand(stream);
localTime_s = (0:widthSamples - 1) / fs_Hz;
gated = centered.pulse(1:widthSamples) .* ...
    exp(1i * (2*pi*frequency_Hz*localTime_s + phase_rad));
```

For `repeated`, sample and sort the first PRI layout once, then copy its
intra-PRI starts, widths, amplitudes, frequencies, and phases into every PRI. For
`independent`, sample and sort each PRI separately. Add overlapping pulses by
complex summation. Apply the common delay with zero-fill only after the clean
record is assembled.

Flatten truth in PRI-major and start-time order. Compute final absolute
`start_XX_s` after adding the common arrival delay. Compute duty cycle from a
logical occupied-sample union so overlap contributes once. Return exactly:

```text
JNR_dB, mode_code, total_pulse_count, mean_width_s, duty_cycle,
mean_amplitude, minimum_frequency_Hz, maximum_frequency_Hz, arrival_delay_s,
start_01_s..32, width_01_s..32, amplitude_01..32,
frequency_01_Hz..32, phase_01_rad..32
```

Unused fields are `NaN`. Mode codes are `0=independent`, `1=repeated`.

- [ ] **Step 4: Add overlap, capacity, support, and delay tests**

Add tests that:

- force overlapping gates and verify `duty_cycle` equals union occupancy;
- verify 8 pulses per PRI and 32 total slots are accepted;
- verify 33 required slots are rejected with `JammingGenerator:InvalidPhysicalParameter`;
- verify every saved frequency is within `[-1e6,1e6]` and its shifted centered-reference support remains within Nyquist;
- verify positive arrival delay produces leading zeros and no record-end wraparound.

- [ ] **Step 5: Run all ClutterPulse tests**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestSingleJammingGenerators.m'),'Name','*clutterPulse*'); assertSuccess(r);"
```

Expected: both modes, ordering, capacity, overlap truth, support, and zero-boundary delay pass.

- [ ] **Step 6: Commit**

```powershell
git add -- '可配置组合干扰数据生成/generators/generate_clutter_pulse.m' '可配置组合干扰数据生成/tests/TestSingleJammingGenerators.m'
git commit -m "feat: add clutter pulse generator"
```

---

### Task 5: Implement Standard Integer-Subpulse SMSP

**Files:**
- Create: `可配置组合干扰数据生成/generators/generate_spectral_dispersion.m`
- Test: `可配置组合干扰数据生成/tests/TestSingleJammingGenerators.m`

- [ ] **Step 1: Write failing tests for `M=2` and `M=8`**

Add:

```matlab
function smspUsesIntegerSubpulseCountAndOriginalBandwidth(testCase)
    cfg = create_new_type_fixture();
    cfg.params.SpectralDispersion = struct( ...
        'num_subpulses', [8, 8], 'max_subpulses', 8);
    cfg.params.common.arrival_delay_s = [0, 0];
    reference = build_reference_signal(cfg);
    [signal, truth] = generate_spectral_dispersion(cfg, ...
        RandStream('mt19937ar','Seed',51), reference);

    expectedSamples = round(cfg.derived.samples_per_pulse / 8);
    testCase.verifyEqual(truth.num_subpulses, 8);
    testCase.verifyEqual(truth.subpulse_width_s, ...
        expectedSamples / cfg.radar.fs_Hz, 'AbsTol', 1e-15);
    testCase.verifyEqual(truth.subpulse_bandwidth_Hz, ...
        cfg.radar.bandwidth_Hz);
    testCase.verifyEqual(truth.output_duration_s, ...
        8 * expectedSamples / cfg.radar.fs_Hz, 'AbsTol', 1e-15);
    testCase.verifySize(signal, [1, cfg.derived.sample_length]);
end

function smspRepeatsIdenticalLocalSubpulses(testCase)
    cfg = create_new_type_fixture();
    cfg.params.SpectralDispersion = struct( ...
        'num_subpulses', [2, 2], 'max_subpulses', 8);
    cfg.params.common.arrival_delay_s = [0, 0];
    reference = build_reference_signal(cfg);
    [signal, truth] = generate_spectral_dispersion(cfg, ...
        RandStream('mt19937ar','Seed',52), reference);
    n = round(truth.subpulse_width_s * cfg.radar.fs_Hz);
    testCase.verifyEqual(signal(1:n), signal(n+1:2*n), ...
        'AbsTol', 64 * eps);
    testCase.verifyEqual(truth.subpulse_start_01_s, 0);
    testCase.verifyEqual(truth.subpulse_start_02_s, n/cfg.radar.fs_Hz);
    testCase.verifyTrue(isnan(truth.subpulse_start_03_s));
end
```

- [ ] **Step 2: Run the focused tests and verify failure**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestSingleJammingGenerators.m'),'Name','*smsp*'); disp(r); assert(any([r.Failed]));"
```

Expected: failure because `generate_spectral_dispersion` does not exist.

- [ ] **Step 3: Implement the standard model without a continuous ratio**

Create `generators/generate_spectral_dispersion.m`. The core construction is:

```matlab
capacity = 8;
fs_Hz = cfg.radar.fs_Hz;
centered = build_centered_reference_signal(cfg, reference);
guard_Hz = fs_Hz / 2 - max(eps(fs_Hz / 2), fs_Hz * 1e-12);
if centered.support_Hz(1) <= -guard_Hz || ...
        centered.support_Hz(2) >= guard_Hz
    throw_invalid_physical_parameter( ...
        'SMSP零中心参考带宽必须严格位于Nyquist范围内。');
end
M = sample_integer(cfg.params.SpectralDispersion.num_subpulses, stream);
subpulseSamples = round(cfg.derived.samples_per_pulse / M);
if subpulseSamples < 2
    throw_invalid_physical_parameter('SMSP每个子脉冲至少需要两个样点。');
end
subpulseWidth_s = subpulseSamples / fs_Hz;
chirpRate_Hz_per_s = cfg.radar.bandwidth_Hz / subpulseWidth_s;
chirpRateMultiplier = cfg.radar.pulse_width_s / subpulseWidth_s;
outputSamples = M * subpulseSamples;
outputDuration_s = outputSamples / fs_Hz;
arrivalDelaySamples = round(sample_range( ...
    cfg.params.common.arrival_delay_s, stream) * fs_Hz);
if arrivalDelaySamples + outputSamples > cfg.derived.samples_per_pri
    throw_invalid_physical_parameter( ...
        'SMSP公共到达时延加输出时长必须完整位于PRI内。');
end

initialPhase_rad = 2 * pi * rand(stream);
localTime_s = (0:subpulseSamples - 1) / fs_Hz;
centerTime_s = (subpulseSamples - 1) / (2 * fs_Hz);
subpulse = exp(1i * (pi * chirpRate_Hz_per_s * ...
    (localTime_s - centerTime_s) .^ 2 + initialPhase_rad));
smspPulse = complex(zeros(1, cfg.derived.samples_per_pri));
smspPulse(1:outputSamples) = repmat(subpulse, 1, M);
smspPulse = delay_with_zeros(smspPulse, arrivalDelaySamples);
signal = complex(repmat(smspPulse, 1, cfg.radar.num_pulses));
```

Validate `num_subpulses` as a positive integer range within 2–8,
`max_subpulses==8`, reference bandwidth strictly inside Nyquist, common JNR and
delay ranges, and output support. Return 17 fields exactly:

```text
JNR_dB, num_subpulses, subpulse_width_s, chirp_rate_multiplier,
subpulse_chirp_rate_Hz_per_s, subpulse_bandwidth_Hz, output_duration_s,
arrival_delay_s, initial_phase_rad, subpulse_start_01_s..08
```

The eight starts describe the first PRI before the common arrival delay and are
`(0:M-1)*subpulseWidth_s`; unused slots are `NaN`.

- [ ] **Step 4: Add chirp-rate, time-frequency, and defensive tests**

Measure adjacent-sample instantaneous frequency separately inside each active
subpulse. Verify its fitted slope equals `subpulse_chirp_rate_Hz_per_s` within
one discrete-frequency bin, its minimum/maximum remain in the original centered
5 MHz band, and each boundary resets from high frequency to low frequency. Add
rejection tests for `M=1`, `M=9`, `max_subpulses=9`, fewer than two samples per
subpulse, and delay plus output exceeding one PRI.

- [ ] **Step 5: Run all SMSP tests**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestSingleJammingGenerators.m'),'Name','*smsp*'); assertSuccess(r);"
```

Expected: `M=2..8`, exact repetition, original bandwidth, realized truth,
zero-boundary delay, and all defensive errors pass at 10 MHz.

- [ ] **Step 6: Commit**

```powershell
git add -- '可配置组合干扰数据生成/generators/generate_spectral_dispersion.m' '可配置组合干扰数据生成/tests/TestSingleJammingGenerators.m'
git commit -m "feat: add standard SMSP generator"
```

---

### Task 6: Integrate Versioned Defaults, Registry, Validation, and Dispatcher

**Files:**
- Modify: `可配置组合干扰数据生成/create_default_config.m`
- Modify: `可配置组合干扰数据生成/build_jamming_registry.m`
- Modify: `可配置组合干扰数据生成/create_closedset_6class_config.m`
- Modify: `可配置组合干扰数据生成/core/normalize_and_validate_config.m`
- Modify: `可配置组合干扰数据生成/core/generate_single_jamming.m`
- Test: `可配置组合干扰数据生成/tests/TestConfigAndRegistry.m`
- Test: `可配置组合干扰数据生成/tests/TestSingleJammingGenerators.m`

- [ ] **Step 1: Write failing v1/v2 contract tests**

Add tests asserting:

```matlab
function defaultConfigAndRegistryUseV2(testCase)
    cfg = create_default_config();
    registry = build_jamming_registry();
    schema = build_parameter_schema(registry);
    expected = {'FMB','AMA','SawFM','SinFM','SN','MFT','VFT','ISRJ', ...
        'RangeFakeTarget','Traction','CombSpectrum','ClutterPulse', ...
        'SpectralDispersion'};
    testCase.verifyEqual(cfg.registry.version, 2);
    testCase.verifyEqual({registry.code}, expected);
    testCase.verifyEqual(height(schema), 295);
    testCase.verifyEqual(cfg.types.known, ...
        {'FMB','AMA','SawFM','MFT','ISRJ','CombSpectrum', ...
         'ClutterPulse','SpectralDispersion'});
end

function explicitV1PreservesLegacyShape(testCase)
    cfg = create_default_config(1);
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    testCase.verifyFalse(isfield(cfg, 'registry'));
    testCase.verifyNumElements(registry, 10);
    testCase.verifyEqual(height(schema), 79);
    testCase.verifyEqual(numel(cfg.imbalance.component.weights), 10);
end

function configWithoutRegistrySelectsV1(testCase)
    cfg = rmfield(create_default_config(2), 'registry');
    cfg.types.enabled = cfg.types.enabled(1:10);
    cfg.types.known = {'FMB','AMA','SawFM','MFT','ISRJ'};
    cfg.types.unknown = setdiff(cfg.types.enabled,cfg.types.known,'stable');
    registry = build_jamming_registry(cfg);
    testCase.verifyNumElements(registry, 10);
end

function closedsetWrappersStayV1(testCase)
    cfg = create_closedset_6class_config();
    registry = build_jamming_registry(cfg);
    testCase.verifyFalse(isfield(cfg, 'registry'));
    testCase.verifyNumElements(registry, 10);
    testCase.verifyEqual(height(build_parameter_schema(registry)), 79);
end
```

- [ ] **Step 2: Run focused version tests and verify failure**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestConfigAndRegistry.m')); disp(r); assert(any([r.Failed]));"
```

Expected: failures because registry versioning and IDs 11–13 are not integrated.

- [ ] **Step 3: Version `create_default_config`**

Change the signature to:

```matlab
function cfg = create_default_config(varargin)
```

Resolve no argument to version 2; accept only numeric scalar integer 1 or 2;
reject other inputs with `JammingConfig:InvalidRegistryVersion`. Build the
existing v1 structure exactly as today. Return immediately for version 1 before
adding any v2 field. For version 2 add:

```matlab
cfg.registry.version = 2;
cfg.types.enabled = [allTypes, ...
    {'CombSpectrum','ClutterPulse','SpectralDispersion'}];
cfg.types.known = {'FMB','AMA','SawFM','MFT','ISRJ', ...
    'CombSpectrum','ClutterPulse','SpectralDispersion'};
cfg.types.unknown = setdiff(cfg.types.enabled, cfg.types.known, 'stable');
cfg.params.CombSpectrum = struct( ...
    'num_teeth',[3,8], 'spacing_Hz',[0.2e6,0.6e6], ...
    'center_offset_Hz',[-0.3e6,0.3e6], 'amplitude',[0.5,1.5], ...
    'amplitude_mode','equal', 'max_teeth',8);
cfg.params.ClutterPulse = struct( ...
    'pulses_per_pri',[3,8], 'pulse_width_s',[2e-6,20e-6], ...
    'relative_amplitude',[0.5,1.5], ...
    'frequency_offset_Hz',[-1e6,1e6], 'mode','independent', ...
    'max_pulses_per_pri',8, 'max_total_pulses',32);
cfg.params.SpectralDispersion = struct( ...
    'num_subpulses',[2,8], 'max_subpulses',8);
cfg.imbalance.component.weights = ones(1,13);
```

- [ ] **Step 4: Append the v2 registry entries**

Refactor `build_jamming_registry` into a v1 builder plus conditional v2 append.
Resolution is exact: no argument means v2; a config without `cfg.registry`
means v1; `cfg.registry.version` selects 1 or 2. Append IDs 11–13 and generate
the numbered names programmatically:

```matlab
registry(11) = create_entry(11,'CombSpectrum','梳状谱干扰', ...
    @generate_comb_spectrum, combNames, combUnits);
registry(12) = create_entry(12,'ClutterPulse','杂乱脉冲干扰', ...
    @generate_clutter_pulse, clutterNames, clutterUnits);
registry(13) = create_entry(13,'SpectralDispersion','频谱弥散干扰', ...
    @generate_spectral_dispersion, smspNames, smspUnits);
```

Construct the arrays exactly as follows:

```matlab
names8 = @(prefix,suffix) arrayfun(@(k) ...
    sprintf('%s_%02d%s',prefix,k,suffix),1:8,'UniformOutput',false);
names32 = @(prefix,suffix) arrayfun(@(k) ...
    sprintf('%s_%02d%s',prefix,k,suffix),1:32,'UniformOutput',false);

combNames = [{'JNR_dB','num_teeth','spacing_Hz','center_offset_Hz', ...
    'amplitude_mode_code','arrival_delay_s'}, ...
    names8('frequency','_Hz'), names8('amplitude',''), ...
    names8('phase','_rad')];
combUnits = [{'dB','1','Hz','Hz','1','s'}, ...
    repmat({'Hz'},1,8), repmat({'1'},1,8), repmat({'rad'},1,8)];

clutterNames = [{'JNR_dB','mode_code','total_pulse_count', ...
    'mean_width_s','duty_cycle','mean_amplitude', ...
    'minimum_frequency_Hz','maximum_frequency_Hz','arrival_delay_s'}, ...
    names32('start','_s'), names32('width','_s'), ...
    names32('amplitude',''), names32('frequency','_Hz'), ...
    names32('phase','_rad')];
clutterUnits = [{'dB','1','1','s','1','1','Hz','Hz','s'}, ...
    repmat({'s'},1,32), repmat({'s'},1,32), ...
    repmat({'1'},1,32), repmat({'Hz'},1,32), ...
    repmat({'rad'},1,32)];

smspNames = [{'JNR_dB','num_subpulses','subpulse_width_s', ...
    'chirp_rate_multiplier','subpulse_chirp_rate_Hz_per_s', ...
    'subpulse_bandwidth_Hz','output_duration_s','arrival_delay_s', ...
    'initial_phase_rad'}, names8('subpulse_start','_s')];
smspUnits = [{'dB','1','s','1','Hz/s','Hz','s','s','rad'}, ...
    repmat({'s'},1,8)];

assert(numel(combNames)==30 && numel(clutterNames)==169 && ...
    numel(smspNames)==17);
```

Assert every name/unit pair has equal length before returning the registry.

- [ ] **Step 5: Pin all legacy closed-set wrappers to v1**

Change only the template line in `create_closedset_6class_config.m`:

```matlab
cfg = create_default_config(1);
```

Do not add `cfg.registry` afterward. Four/five-class and power-sweep configs
already inherit this function and therefore remain v1 without further changes.

- [ ] **Step 6: Extend normalization and physical validation**

Update integer recognition to include:

```matlab
{'num_targets','num_replicas','num_teeth','pulses_per_pri','num_subpulses'}
```

and fixed capacities:

```matlab
{'max_targets','max_teeth','max_pulses_per_pri', ...
 'max_total_pulses','max_subpulses'}
```

Treat `amplitude_mode` and `mode` as text fields. Add enabled-type-specific
validation for Comb feasibility, Clutter 8/32 capacity and PRI support, and SMSP
2–8 count, at least two samples per subpulse, unchanged 5 MHz band, and
arrival-delay-plus-output support. Disabled new types must not block a valid
configuration.

- [ ] **Step 7: Extend dispatcher `NaN` permissions narrowly**

In `allows_nan_padding`, permit only:

```matlab
CombSpectrum: frequency_XX_Hz, amplitude_XX, phase_XX_rad
ClutterPulse: start_XX_s, width_XX_s, amplitude_XX,
              frequency_XX_Hz, phase_XX_rad
SpectralDispersion: subpulse_start_XX_s
```

Do not allow `NaN` in any summary field.

- [ ] **Step 8: Make existing unit tests explicitly version-aware**

Audit every `create_default_config()` and `build_jamming_registry()` call in
`TestConfigAndRegistry.m` and `TestSingleJammingGenerators.m`. Tests asserting
legacy counts, old exact code lists, or old parameter ordering must call
`create_default_config(1)` and `build_jamming_registry(cfg)`. Tests exercising
the current default must assert 13 types and include the three new generators.
Do not perform a blind replacement: each test must state whether it covers v1
compatibility or v2 default behavior in its Chinese comment.

- [ ] **Step 9: Run configuration, registry, and dispatcher tests**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); files={fullfile(root,'tests','TestConfigAndRegistry.m'),fullfile(root,'tests','TestSingleJammingGenerators.m')}; r=runtests(files); assertSuccess(r);"
```

Expected: v1=10/79, v2=13/295, all thirteen dispatcher contracts pass, and
legacy closed-set wrappers still expose ten labels.

- [ ] **Step 10: Commit**

```powershell
git add -- '可配置组合干扰数据生成/create_default_config.m' '可配置组合干扰数据生成/build_jamming_registry.m' '可配置组合干扰数据生成/create_closedset_6class_config.m' '可配置组合干扰数据生成/core/normalize_and_validate_config.m' '可配置组合干扰数据生成/core/generate_single_jamming.m' '可配置组合干扰数据生成/tests/TestConfigAndRegistry.m' '可配置组合干扰数据生成/tests/TestSingleJammingGenerators.m'
git commit -m "feat: add versioned thirteen-type registry"
```

---

### Task 7: Verify Balanced, Imbalanced, Open-Set, and Legacy Dataset Paths

**Files:**
- Modify: `可配置组合干扰数据生成/tests/TestManifestBuilders.m`
- Modify: `可配置组合干扰数据生成/tests/TestDatasetGeneration.m`

- [ ] **Step 1: Add one reusable small v2 fixture**

Add a local helper in each test file:

```matlab
function cfg = create_small_v2_fixture()
    cfg = create_default_config(2);
    cfg.types.enabled = ...
        {'CombSpectrum','ClutterPulse','SpectralDispersion'};
    cfg.types.known = {'CombSpectrum','ClutterPulse'};
    cfg.types.unknown = {'SpectralDispersion'};
    cfg.dataset.num_samples = 12;
    cfg.dataset.min_components = 1;
    cfg.dataset.max_components = 3;
    cfg.dataset.combination_mode = 'random';
    cfg.dataset.samples_per_pattern = 1;
    cfg.params.common.JNR_dB = [20, 20];
    cfg.params.common.arrival_delay_s = [0, 0];
    cfg.output.save_legacy = false;
end
```

Before adding new tests, audit existing fixed-width assertions in both files.
Any test for a legacy closed-set wrapper must keep 10 labels/79 parameters;
generic no-argument default tests must move to 13 labels/295 parameters. Update
Chinese comments so the intended registry version is explicit.

- [ ] **Step 2: Write failing manifest tests for all three scenarios**

Add:

```matlab
function v2BalancedManifestUsesNewTypeIds(testCase)
    cfg = normalize_and_validate_config(create_small_v2_fixture(), 'balanced');
    registry = build_jamming_registry(cfg);
    manifest = build_balanced_manifest(cfg, registry, 'train');
    active = unique([manifest.active_type_ids{:}]);
    testCase.verifyTrue(all(ismember(active, 11:13)));
end

function v2ImbalanceWeightsAcceptThirteenRegistryEntries(testCase)
    cfg = create_small_v2_fixture();
    cfg.imbalance.component.enable = true;
    cfg.imbalance.component.weights = [zeros(1,10), 1, 2, 3];
    cfg.imbalance.combination.enable = true;
    cfg.imbalance.combination.cardinality_weights = [3, 2, 1];
    cfg.imbalance.power.enable = true;
    cfg.imbalance.power.relative_power_dB = [-12, -6];
    cfg = normalize_and_validate_config(cfg, 'imbalanced');
    registry = build_jamming_registry(cfg);
    manifest = build_imbalanced_manifest(cfg, registry, 'train');
    testCase.verifyEqual(height(manifest), cfg.dataset.num_samples);
    testCase.verifyTrue(all(cellfun(@(ids) all(ismember(ids,11:13)), ...
        manifest.active_type_ids)));
end

function v2OpenSetMarksSmspUnknown(testCase)
    cfg = create_small_v2_fixture();
    cfg.openset.train_known = 4;
    cfg.openset.calibration_known = 2;
    cfg.openset.validation_mixed = 4;
    cfg.openset.test_mixed = 4;
    cfg = normalize_and_validate_config(cfg, 'openset');
    registry = build_jamming_registry(cfg);
    splits = build_openset_manifest(cfg, registry);
    smspId = find(strcmp({registry.code},'SpectralDispersion'));
    testCase.verifyFalse(any(cellfun(@(ids) ismember(smspId,ids), ...
        splits.train_known.active_type_ids)));
    testCase.verifyTrue(any(cellfun(@(ids) ismember(smspId,ids), ...
        splits.test_mixed.active_type_ids)));
end
```

- [ ] **Step 3: Run focused manifest tests and verify failure before integration is complete**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestManifestBuilders.m'),'Name','*v2*'); disp(r); assert(any([r.Failed]));"
```

Expected: at least one failure until all v2 normalization and registry behavior
from Task 6 is present.

- [ ] **Step 4: Add an end-to-end v2 bundle test**

```matlab
function v2BundleStoresThirteenLabelsAnd295Parameters(testCase)
    cfg = create_small_v2_fixture();
    cfg.dataset.num_samples = 3;
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    manifest = build_balanced_manifest(cfg, registry, 'train');
    bundle = generate_dataset_from_manifest(cfg, registry, schema, manifest);

    testCase.verifySize(bundle.label_all, [3, 13]);
    testCase.verifySize(bundle.parameter_values, [3, 295]);
    testCase.verifySize(bundle.parameter_mask, [3, 295]);
    testCase.verifyTrue(all(isfinite(real(bundle.iq_complex)), 'all'));
    testCase.verifyTrue(all(isfinite(imag(bundle.iq_complex)), 'all'));
end
```

- [ ] **Step 5: Add a v1 backward-compatibility dry-run test**

```matlab
function legacyClosedsetDryRunRemainsTenBy79(testCase)
    cfg = create_closedset_6class_config();
    cfg.dataset.samples_per_pattern = 1;
    cfg.dataset.num_samples = 63;
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    manifest = build_balanced_manifest(cfg, registry, 'test');
    testCase.verifyNumElements(registry, 10);
    testCase.verifyEqual(height(schema), 79);
    testCase.verifyEqual(width(manifest), 14);

    one = generate_dataset_from_manifest( ...
        cfg, registry, schema, manifest(1,:));
    testCase.verifySize(one.label_all, [1,10]);
    testCase.verifySize(one.parameter_values, [1,79]);
end
```

Do not load or rewrite the formal 0716 MAT files; this test proves the same
registry and schema contract through a one-row in-memory dry run.

- [ ] **Step 6: Run both test files**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); files={fullfile(root,'tests','TestManifestBuilders.m'),fullfile(root,'tests','TestDatasetGeneration.m')}; r=runtests(files); assertSuccess(r);"
```

Expected: balanced, component/composition/power imbalance, open-set partition,
v2 bundle dimensions, and v1 one-row dry run all pass.

- [ ] **Step 7: Commit**

```powershell
git add -- '可配置组合干扰数据生成/tests/TestManifestBuilders.m' '可配置组合干扰数据生成/tests/TestDatasetGeneration.m'
git commit -m "test: cover new types across dataset scenarios"
```

---

### Task 8: Complete Chinese Documentation and Code Comments

**Files:**
- Create: `可配置组合干扰数据生成/README_新增三类干扰.md`
- Modify: `可配置组合干扰数据生成/README_使用说明.md`
- Modify: `可配置组合干扰数据生成/程序说明_组合干扰数据生成系统.md`
- Modify: every file created or changed in Tasks 2–7

- [ ] **Step 1: Write the dedicated new-type guide**

The new README must include these exact sections:

```markdown
# 梳状谱、杂乱脉冲与频谱弥散干扰使用说明

## 1. v1与v2注册表
## 2. 已知类、未知类和启用类型的修改方法
## 3. CombSpectrum数学模型与参数表
## 4. ClutterPulse数学模型、两种模式与32槽语义
## 5. SMSP标准整数子脉冲模型
## 6. 10 MHz采样与零中心参考LFM
## 7. 295维参数真值和NaN掩码
## 8. 平衡、不平衡和开放集示例
## 9. 常见配置错误及稳定错误标识
```

Include executable examples for `create_default_config(2)`,
`create_default_config(1)`, selecting each new type alone, changing
known/unknown lists, equal/random comb amplitudes, independent/repeated clutter,
and fixed SMSP `num_subpulses=[4,4]`.

- [ ] **Step 2: Update the general README and program description**

State unambiguously:

- no argument and explicit version 2 use 13 labels and 295 parameters;
- missing `cfg.registry` and `create_default_config(1)` use 10 labels and 79 parameters;
- old four/five/six-class configs remain v1;
- SMSP has no independent continuous compression ratio;
- `num_subpulses` is an integer 2–8 and determines width, repeat count, and chirp-rate multiplier;
- the registry extension does not automatically increase configured dataset sample counts.

- [ ] **Step 3: Audit comments in all implementation files**

Every public function must document inputs, outputs, units, random-stream use,
zero-boundary delay, physical limits, truth fields, and error identifiers. Every
non-obvious equation or quantization decision must have a Chinese comment next
to the corresponding code. Remove stale comments saying there are only ten
types unless the comment is explicitly describing v1.

- [ ] **Step 4: Scan documentation for stale design terms**

```powershell
rg -n "1\.1|1\.8|compression_ratio|num_segments|max_segments|301列|301维|固定十类|只有十类" '可配置组合干扰数据生成'
```

Expected: no stale SMSP continuous-ratio terms; occurrences of ten classes are
explicitly marked as v1/legacy only.

- [ ] **Step 5: Commit**

```powershell
git add -- '可配置组合干扰数据生成/README_新增三类干扰.md' '可配置组合干扰数据生成/README_使用说明.md' '可配置组合干扰数据生成/程序说明_组合干扰数据生成系统.md' '可配置组合干扰数据生成/create_default_config.m' '可配置组合干扰数据生成/build_jamming_registry.m' '可配置组合干扰数据生成/core' '可配置组合干扰数据生成/generators' '可配置组合干扰数据生成/tests'
git commit -m "docs: explain thirteen-type generator"
```

---

### Task 9: Run Full Verification and Produce the Program Explanation

**Files:**
- Verify: all files under `可配置组合干扰数据生成`
- Read: `docs/superpowers/specs/2026-07-25-three-new-jamming-types-design.md`

- [ ] **Step 1: Run all MATLAB tests**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); r=runtests(fullfile(root,'tests')); fprintf('FINAL_TOTAL=%d FINAL_FAILED=%d\n',numel(r),sum([r.Failed])); assertSuccess(r);"
```

Expected: every legacy and new test passes; `FINAL_FAILED=0` and total exceeds
the 200-test baseline.

- [ ] **Step 2: Run MATLAB Code Analyzer over every `.m` file**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; files=dir(fullfile(root,'**','*.m')); issueCount=0; for k=1:numel(files); issues=checkcode(fullfile(files(k).folder,files(k).name),'-id'); issueCount=issueCount+numel(issues); end; fprintf('MATLAB_FILES=%d CHECKCODE_ISSUES=%d\n',numel(files),issueCount); assert(issueCount==0);"
```

Expected: `CHECKCODE_ISSUES=0`.

- [ ] **Step 3: Run a deterministic thirteen-type smoke test**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); cfg=normalize_and_validate_config(create_default_config(2),'balanced'); registry=build_jamming_registry(cfg); reference=build_reference_signal(cfg); for k=1:numel(registry); [a,ta]=generate_single_jamming(registry(k).code,cfg,registry,RandStream('mt19937ar','Seed',1000+k),reference); [b,tb]=generate_single_jamming(registry(k).code,cfg,registry,RandStream('mt19937ar','Seed',1000+k),reference); assert(isequal(a,b)); assert(isequaln(ta,tb)); end; fprintf('SMOKE_TYPES=%d SCHEMA=%d\n',numel(registry),height(build_parameter_schema(registry)));"
```

Expected: `SMOKE_TYPES=13 SCHEMA=295`.

- [ ] **Step 4: Verify exact legacy dimensions independently**

```powershell
$root=(Resolve-Path '可配置组合干扰数据生成').Path
matlab -batch "root='$($root.Replace("'","''"))'; addpath(genpath(root)); cfg=create_closedset_6class_config(); registry=build_jamming_registry(cfg); schema=build_parameter_schema(registry); assert(numel(registry)==10); assert(height(schema)==79); assert(~isfield(cfg,'registry')); fprintf('LEGACY_TYPES=%d LEGACY_SCHEMA=%d\n',numel(registry),height(schema));"
```

Expected: `LEGACY_TYPES=10 LEGACY_SCHEMA=79`.

- [ ] **Step 5: Review the implementation against every acceptance criterion**

Confirm all of the following from fresh command output and source inspection:

- v1 remains 10 labels/79 parameters and old closed-set configs contain no registry field;
- v2 has stable IDs 1–13 and 295 parameters;
- all three new generators are reference-coupled, deterministic by scalar `RandStream`, and return finite complex rows;
- Comb supports both amplitude modes and strict shifted-support validation;
- Clutter supports both layout modes, overlap-aware duty cycle, and 8/32 capacities;
- SMSP uses only integer `M=2..8`, repeats identical full-band subpulses, and has no continuous compression ratio;
- all new arrival delays use zero boundaries;
- balanced, imbalanced, and open-set tests cover the new types;
- comments and user documentation are complete.

- [ ] **Step 6: Commit any verification-only corrections, then confirm clean status**

```powershell
git status --short
git diff --check
git log --oneline -8
```

If verification required a correction, commit only the corrected files with a
specific message before rerunning Steps 1–5. Final expected status is clean.

- [ ] **Step 7: Write the user-facing program explanation after implementation**

The final handoff must summarize:

- where the three generator files and documentation are stored;
- how to choose v1/v2, enabled types, and known/unknown types;
- all adjustable parameters and fixed capacities;
- standard SMSP physical and mathematical relationships;
- label and parameter storage formats;
- exact verification totals and Code Analyzer result;
- confirmation that no formal dataset was generated unless separately requested.

---

## Completion Definition

The implementation is complete only when all nine tasks are checked, the full
MATLAB suite and Code Analyzer pass from the current worktree, v1 reports
10/79, v2 reports 13/295, Git is clean, and the user receives the program
explanation. Formal large-scale data generation remains a separate explicit
action.
