# Four-Class and Five-Class Dataset Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and formally generate four-class and five-class closed-set compound-jamming datasets, each with an independent 10–30 dB full-power dataset and a paired nine-level power sweep.

**Architecture:** Generalize only the category-dependent seams in the proven six-class pipeline: output file stems, subset configuration, power-profile configuration, full-power truth verification, and family orchestration. Reuse the existing physical generators, manifest builders, atomic partial/final workflow, resume semantics, and power-sweep runner. Keep four independent seed domains and run one MATLAB generation process serially.

**Tech Stack:** MATLAB, `matlab.unittest`, MAT v7.3/HDF5, PowerShell, Git, SHA-256.

---

## File Map

### Create

- `可配置组合干扰数据生成/core/resolve_closedset_output_paths.m` — derive final and partial paths from `cfg.output.dataset_stem`.
- `可配置组合干扰数据生成/core/audit_closedset_seed_domains.m` — prove four experiment seed domains are pairwise disjoint.
- `可配置组合干扰数据生成/create_closedset_subset_config.m` — generic full-power subset configuration.
- `可配置组合干扰数据生成/create_power_sweep_subset_config.m` — generic paired sweep configuration.
- `可配置组合干扰数据生成/create_closedset_4class_config.m` — approved four-class full-power wrapper.
- `可配置组合干扰数据生成/create_closedset_5class_config.m` — approved five-class full-power wrapper.
- `可配置组合干扰数据生成/create_power_sweep_4class_config.m` — approved four-class sweep wrapper.
- `可配置组合干扰数据生成/create_power_sweep_5class_config.m` — approved five-class sweep wrapper.
- `可配置组合干扰数据生成/core/run_subset_dataset_family.m` — run one class-count family in full-power then sweep order.
- `可配置组合干扰数据生成/generate_closedset_4class_datasets.m` — public four-class entry.
- `可配置组合干扰数据生成/generate_closedset_5class_datasets.m` — public five-class entry.
- `可配置组合干扰数据生成/generate_closedset_4class_5class_all.m` — public serial master entry.
- `可配置组合干扰数据生成/README_四类五类闭集数据.md` — program usage and storage contract.

### Modify

- `可配置组合干扰数据生成/create_default_config.m` — preserve the old six-class config shape unchanged.
- `可配置组合干扰数据生成/core/normalize_and_validate_config.m` — validate and normalize `dataset_stem`.
- `可配置组合干扰数据生成/core/run_closedset_splits.m` — use resolved category-specific paths.
- `可配置组合干扰数据生成/core/run_power_sweep_levels.m` — use resolved category-specific paths in level results.
- `可配置组合干扰数据生成/core/verify_saved_closedset_split.m` — verify full-power component JNR/power truth as well as sweep truth.
- `可配置组合干扰数据生成/tests/TestDatasetGeneration.m` — TDD coverage for every new behavior and backward compatibility.

### Generate after implementation

- `G:\论文相关\干扰数据\复合干扰数据0716\4类\...`
- `G:\论文相关\干扰数据\复合干扰数据0716\5类\...`
- Chinese dataset descriptions and 20-row/40-row SHA-256 inventories.

All new public functions, configuration functions, runner helpers, and non-obvious
test fixtures must contain complete Chinese comments explaining inputs, outputs,
power semantics, seed semantics, failure behavior, and why the implementation is
safe for the existing six-class workflow.

---

### Task 1: Establish a Clean Baseline

**Files:**
- Test: `可配置组合干扰数据生成/tests/TestDatasetGeneration.m`

- [ ] **Step 1: Verify branch and preserve unrelated files**

Run:

```powershell
git status --short --branch
git branch --show-current
```

Expected branch: `codex/four-five-class-datasets`. The pre-existing untracked Chinese Markdown file remains untracked and untouched.

- [ ] **Step 2: Run the full MATLAB baseline**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=runtests(fullfile(root,'tests')); assertSuccess(r); fprintf('BASELINE_TOTAL=%d\n',numel(r));"
```

Expected: all existing tests pass; the current known baseline is 192 tests.

- [ ] **Step 3: Confirm formal target directories are not already populated**

```powershell
$root='G:\论文相关\干扰数据\复合干扰数据0716'
Get-ChildItem -LiteralPath $root -Force | Where-Object Name -in @('4类','5类')
```

Expected: either no directories, or empty directories containing no conflicting MAT files. If conflicting files exist, stop and report them; do not delete them.

---

### Task 2: Generalize Closed-Set Output File Names

**Files:**
- Create: `可配置组合干扰数据生成/core/resolve_closedset_output_paths.m`
- Preserve: `可配置组合干扰数据生成/create_default_config.m`
- Modify: `可配置组合干扰数据生成/core/normalize_and_validate_config.m`
- Modify: `可配置组合干扰数据生成/core/run_closedset_splits.m`
- Modify: `可配置组合干扰数据生成/core/run_power_sweep_levels.m`
- Test: `可配置组合干扰数据生成/tests/TestDatasetGeneration.m`

- [ ] **Step 1: Write failing tests for custom and default stems**

Add these test methods:

```matlab
function outputStemControlsClosedsetPaths(testCase)
    tempDir = tempname;
    mkdir(tempDir);
    cleanup = onCleanup(@() rmdir(tempDir, 's')); %#ok<NASGU>
    [cfg, registry, schema] = ...
        create_small_closedset_runner_fixture(tempDir);
    cfg.output.dataset_stem = 'closedset_4class';
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);

    result = run_closedset_splits( ...
        cfg, registry, schema, false, 'both', 'resume');

    testCase.verifyEqual(result.train_path, fullfile( ...
        tempDir, 'closedset_4class_train.mat'));
    testCase.verifyEqual(result.test_path, fullfile( ...
        tempDir, 'closedset_4class_test.mat'));
    testCase.verifyEqual(result.splits.train.partial_path, fullfile( ...
        tempDir, 'closedset_4class_train.partial.mat'));
end

function defaultOutputStemKeepsSixClassNames(testCase)
    cfg = create_default_config();
    testCase.verifyFalse(isfield(cfg.output, 'dataset_stem'));
    cfg = normalize_and_validate_config(cfg, 'balanced');
    testCase.verifyFalse(isfield(cfg.output, 'dataset_stem'));
    paths = resolve_closedset_output_paths(cfg);
    testCase.verifyEqual(paths.train_name, 'closedset_6class_train.mat');
    testCase.verifyEqual(paths.test_name, 'closedset_6class_test.mat');
end

function invalidOutputStemIsRejected(testCase)
    cfg = create_default_config();
    cfg.output.dataset_stem = '..\bad-name';
    testCase.verifyError(@() normalize_and_validate_config( ...
        cfg, 'balanced'), 'JammingConfig:InvalidOutputStem');
end
```

- [ ] **Step 2: Run the focused tests and verify failure**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestDatasetGeneration.m'),'Name','*OutputStem*'); disp(r); assert(any([r.Failed]));"
```

Expected: failures because `dataset_stem` and `resolve_closedset_output_paths` do not exist.

- [ ] **Step 3: Add optional normalization without changing old configs**

In `normalize_and_validate_config.m`, after boolean normalization, add:

```matlab
if isfield(cfg.output, 'dataset_stem')
    cfg.output.dataset_stem = normalize_dataset_stem( ...
        cfg.output.dataset_stem);
end
```

Do not add the field to `create_default_config.m`; the resolver supplies the
six-class fallback when it is absent. This preserves exact config equality with
already generated six-class MAT files.

Add the local helper:

```matlab
function value = normalize_dataset_stem(value)
    if isstring(value) && isscalar(value) && ~ismissing(value)
        value = char(value);
    elseif ~(ischar(value) && isrow(value))
        error('JammingConfig:InvalidOutputStem', ...
            'cfg.output.dataset_stem 必须是文本标量。');
    end
    value = strtrim(value);
    if isempty(regexp(value, '^closedset_[A-Za-z0-9_]+$', 'once'))
        error('JammingConfig:InvalidOutputStem', ...
            ['cfg.output.dataset_stem 必须以 closedset_ 开头，' ...
             '且只能包含英文字母、数字和下划线。']);
    end
end
```

- [ ] **Step 4: Create the focused path resolver**

Create `core/resolve_closedset_output_paths.m`:

```matlab
function paths = resolve_closedset_output_paths(cfg)
%RESOLVE_CLOSEDSET_OUTPUT_PATHS 由配置生成final和partial精确路径。

    if ~(isstruct(cfg) && isscalar(cfg) && ...
            isfield(cfg, 'output_dir') && isfield(cfg, 'output'))
        error('JammingDataset:InvalidRunnerInput', ...
            'cfg 缺少 output_dir 或 output。');
    end
    if isfield(cfg.output, 'dataset_stem')
        stem = cfg.output.dataset_stem;
    else
        stem = 'closedset_6class';
    end
    paths = struct( ...
        'train_name', [stem '_train.mat'], ...
        'test_name', [stem '_test.mat'], ...
        'train_partial_name', [stem '_train.partial.mat'], ...
        'test_partial_name', [stem '_test.partial.mat']);
    paths.train = fullfile(cfg.output_dir, paths.train_name);
    paths.test = fullfile(cfg.output_dir, paths.test_name);
    paths.train_partial = fullfile( ...
        cfg.output_dir, paths.train_partial_name);
    paths.test_partial = fullfile( ...
        cfg.output_dir, paths.test_partial_name);
end
```

- [ ] **Step 5: Replace hard-coded paths in both runners**

In `run_closedset_splits.m`, replace the four hard-coded paths with:

```matlab
paths = resolve_closedset_output_paths(cfg);
trainPath = paths.train;
testPath = paths.test;
trainPartialPath = paths.train_partial;
testPartialPath = paths.test_partial;
```

In `run_power_sweep_levels.m` inside `initialize_level_result(cfg)`, use:

```matlab
paths = resolve_closedset_output_paths(cfg);
```

and assign:

```matlab
'train_path', paths.train, ...
'test_path', paths.test, ...
```

- [ ] **Step 6: Run focused and backward-compatibility tests**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestDatasetGeneration.m')); assertSuccess(r);"
```

Expected: all tests pass, including existing six-class path assertions.

- [ ] **Step 7: Commit**

```powershell
git add -- '可配置组合干扰数据生成/core/normalize_and_validate_config.m' '可配置组合干扰数据生成/core/resolve_closedset_output_paths.m' '可配置组合干扰数据生成/core/run_closedset_splits.m' '可配置组合干扰数据生成/core/run_power_sweep_levels.m' '可配置组合干扰数据生成/tests/TestDatasetGeneration.m'
git commit -m "refactor: generalize closedset output names"
```

---

### Task 3: Add Generic Full-Power Subset Configurations

**Files:**
- Create: `可配置组合干扰数据生成/create_closedset_subset_config.m`
- Create: `可配置组合干扰数据生成/create_closedset_4class_config.m`
- Create: `可配置组合干扰数据生成/create_closedset_5class_config.m`
- Test: `可配置组合干扰数据生成/tests/TestDatasetGeneration.m`

- [ ] **Step 1: Write failing configuration tests**

```matlab
function fourAndFiveClassConfigsMatchApprovedDesign(testCase)
    four = create_closedset_4class_config();
    five = create_closedset_5class_config();
    testCase.verifyEqual(four.types.enabled, ...
        {'FMB','SawFM','MFT','ISRJ'});
    testCase.verifyEqual(four.types.known, four.types.enabled);
    testCase.verifyEmpty(four.types.unknown);
    testCase.verifyEqual(four.seed, 100260722);
    testCase.verifyEqual(four.dataset.max_components, 4);
    testCase.verifyEqual(four.dataset.num_samples, 6000);
    testCase.verifyEqual(four.output.dataset_stem, 'closedset_4class');
    testCase.verifyEqual(four.params.common.JNR_dB, [10,30]);
    testCase.verifyEqual(four.output_dir, fullfile( ...
        'G:\论文相关\干扰数据\复合干扰数据0716', ...
        '4类', 'full_power_10_30dB_v1'));
    testCase.verifyEqual(five.types.enabled, ...
        {'FMB','SinFM','MFT','VFT','ISRJ'});
    testCase.verifyEqual(five.seed, 300260722);
    testCase.verifyEqual(five.dataset.max_components, 5);
    testCase.verifyEqual(five.dataset.num_samples, 12400);
    testCase.verifyEqual(five.output.dataset_stem, 'closedset_5class');
end
```

- [ ] **Step 2: Run and verify failure**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestDatasetGeneration.m'),'Name','*fourAndFiveClassConfigsMatchApprovedDesign*'); disp(r); assert(any([r.Failed]));"
```

Expected: the configuration wrapper functions are undefined.

- [ ] **Step 3: Implement the generic full-power configuration**

Create `create_closedset_subset_config.m`:

```matlab
function cfg = create_closedset_subset_config( ...
        selectedTypes, seed, outputDir)
%CREATE_CLOSEDSET_SUBSET_CONFIG 创建全部非空组合的闭集类型子集配置。

    if ~(iscell(selectedTypes) && isrow(selectedTypes) && ...
            ~isempty(selectedTypes) && ...
            all(cellfun(@(x) ischar(x) && isrow(x), selectedTypes)))
        error('JammingSubset:InvalidTypes', ...
            'selectedTypes 必须是非空字符行单元。');
    end
    if numel(unique(selectedTypes, 'stable')) ~= numel(selectedTypes)
        error('JammingSubset:InvalidTypes', 'selectedTypes 不允许重复。');
    end
    if ~(isnumeric(seed) && isreal(seed) && isscalar(seed) && ...
            isfinite(seed) && seed >= 0 && seed == floor(seed))
        error('JammingSubset:InvalidSeed', 'seed 必须是非负整数标量。');
    end
    if isstring(outputDir) && isscalar(outputDir)
        outputDir = char(outputDir);
    end
    if ~(ischar(outputDir) && isrow(outputDir) && ~isempty(strtrim(outputDir)))
        error('JammingSubset:InvalidOutputDir', ...
            'outputDir 必须是非空文本标量。');
    end
    cfg = create_closedset_6class_config();
    classCount = numel(selectedTypes);
    patternCount = 2 ^ classCount - 1;
    cfg.seed = double(seed);
    cfg.output_dir = outputDir;
    cfg.output.dataset_stem = sprintf('closedset_%dclass', classCount);
    cfg.types.enabled = selectedTypes;
    cfg.types.known = selectedTypes;
    cfg.types.unknown = {};
    cfg.dataset.num_samples = patternCount * 400;
    cfg.dataset.min_components = 1;
    cfg.dataset.max_components = classCount;
    cfg.dataset.combination_mode = 'all';
    cfg.dataset.samples_per_pattern = 400;
    cfg.params.common.JNR_dB = [10, 30];
    cfg.experiment = struct();
    cfg.experiment.name = sprintf( ...
        'closedset_%dclass_full_power_10_30dB_v1', classCount);
    cfg.experiment.full_power = struct( ...
        'JNR_range_dB', [10, 30], ...
        'independent_components', true);
end
```

- [ ] **Step 4: Add approved wrappers**

```matlab
function cfg = create_closedset_4class_config()
    root = fullfile( ...
        'G:\论文相关\干扰数据\复合干扰数据0716', '4类');
    cfg = create_closedset_subset_config( ...
        {'FMB','SawFM','MFT','ISRJ'}, 100260722, ...
        fullfile(root, 'full_power_10_30dB_v1'));
end
```

```matlab
function cfg = create_closedset_5class_config()
    root = fullfile( ...
        'G:\论文相关\干扰数据\复合干扰数据0716', '5类');
    cfg = create_closedset_subset_config( ...
        {'FMB','SinFM','MFT','VFT','ISRJ'}, 300260722, ...
        fullfile(root, 'full_power_10_30dB_v1'));
end
```

- [ ] **Step 5: Verify normalized manifests**

Normalize both configs, build train manifests, and assert 6,000/15 and 12,400/31. Group counts must be exactly 400 for every combination.

- [ ] **Step 6: Run full tests and commit**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=runtests(fullfile(root,'tests')); assertSuccess(r);"
git add -- '可配置组合干扰数据生成/create_closedset_subset_config.m' '可配置组合干扰数据生成/create_closedset_4class_config.m' '可配置组合干扰数据生成/create_closedset_5class_config.m' '可配置组合干扰数据生成/tests/TestDatasetGeneration.m'
git commit -m "feat: add four and five class full-power configs"
```

---

### Task 4: Add Generic Four-Class and Five-Class Power Sweeps

**Files:**
- Create: `可配置组合干扰数据生成/create_power_sweep_subset_config.m`
- Create: `可配置组合干扰数据生成/create_power_sweep_4class_config.m`
- Create: `可配置组合干扰数据生成/create_power_sweep_5class_config.m`
- Test: `可配置组合干扰数据生成/tests/TestDatasetGeneration.m`

- [ ] **Step 1: Write failing sweep configuration tests**

Assert the four-class fixed/swept partition, seed `200260722`, and stem `closedset_4class`; assert the five-class partition, seed `400260722`, and stem `closedset_5class`.

- [ ] **Step 2: Run and verify failure**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestDatasetGeneration.m'),'Name','*subsetPowerSweepConfigsPartitionTypesAndLevels*'); disp(r); assert(any([r.Failed]));"
```

Expected: the subset sweep configuration functions are undefined.

- [ ] **Step 3: Implement the generic sweep configuration**

Create `create_power_sweep_subset_config.m`:

```matlab
function cfg = create_power_sweep_subset_config( ...
        selectedTypes, fixedCodes, sweptCodes, seed, sweepRoot, sweepJnr_dB)
%CREATE_POWER_SWEEP_SUBSET_CONFIG 创建类型子集的一个配对功率挡位。

    levels = 10:5:50;
    if ~(isnumeric(sweepJnr_dB) && isreal(sweepJnr_dB) && ...
            isscalar(sweepJnr_dB) && isfinite(sweepJnr_dB) && ...
            ismember(double(sweepJnr_dB), levels))
        error('JammingPowerSweep:InvalidLevel', ...
            'sweepJnr_dB 必须属于 10:5:50。');
    end
    if ~isempty(intersect(fixedCodes, sweptCodes)) || ...
            ~isempty(setxor([fixedCodes, sweptCodes], selectedTypes))
        error('JammingPowerSweep:InvalidProfile', ...
            'fixedCodes和sweptCodes必须无交集且并集等于selectedTypes。');
    end
    levelIndex = find(levels == double(sweepJnr_dB), 1);
    cfg = create_closedset_subset_config( ...
        selectedTypes, seed, sweepRoot);
    cfg.params.common.JNR_dB = [30, 30];
    cfg.output_dir = fullfile(sweepRoot, sprintf( ...
        'level_%02d__jnr_%ddB', levelIndex, sweepJnr_dB));
    cfg.experiment = struct();
    cfg.experiment.name = sprintf( ...
        'closedset_%dclass_power_sweep_9level_v1', ...
        numel(selectedTypes));
    cfg.experiment.power_sweep = struct( ...
        'fixed_codes', {fixedCodes}, ...
        'swept_codes', {sweptCodes}, ...
        'fixed_JNR_dB', 30, ...
        'sweep_JNR_dB', double(sweepJnr_dB), ...
        'standard_levels_dB', levels, ...
        'level_index', levelIndex, ...
        'paired_seeds', true);
end
```

- [ ] **Step 4: Add approved wrappers**

```matlab
function cfg = create_power_sweep_4class_config(level_dB)
    root = fullfile( ...
        'G:\论文相关\干扰数据\复合干扰数据0716', ...
        '4类', 'power_sweep_9level_v1');
    cfg = create_power_sweep_subset_config( ...
        {'FMB','SawFM','MFT','ISRJ'}, {'FMB','SawFM'}, ...
        {'MFT','ISRJ'}, 200260722, root, level_dB);
end
```

```matlab
function cfg = create_power_sweep_5class_config(level_dB)
    root = fullfile( ...
        'G:\论文相关\干扰数据\复合干扰数据0716', ...
        '5类', 'power_sweep_9level_v1');
    cfg = create_power_sweep_subset_config( ...
        {'FMB','SinFM','MFT','VFT','ISRJ'}, {'FMB','SinFM'}, ...
        {'MFT','VFT','ISRJ'}, 400260722, root, level_dB);
end
```

- [ ] **Step 5: Test paired manifests at 10, 30, and 50 dB**

For the all-types rows, assert exact offsets:

```matlab
fourExpected = {[0,0,-20,-20], zeros(1,4), [0,0,20,20]};
fiveExpected = {[0,0,-20,-20,-20], zeros(1,5), [0,0,20,20,20]};
```

Also assert the corresponding weak masks plus equal `sample_seed`, `combination_id`, and `active_type_codes` across levels.

- [ ] **Step 6: Run full tests and commit**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=runtests(fullfile(root,'tests')); assertSuccess(r);"
git add -- '可配置组合干扰数据生成/create_power_sweep_subset_config.m' '可配置组合干扰数据生成/create_power_sweep_4class_config.m' '可配置组合干扰数据生成/create_power_sweep_5class_config.m' '可配置组合干扰数据生成/tests/TestDatasetGeneration.m'
git commit -m "feat: add subset power sweep configs"
```

---

### Task 5: Verify Full-Power JNR and Power Truth on Readback

**Files:**
- Modify: `可配置组合干扰数据生成/core/verify_saved_closedset_split.m`
- Test: `可配置组合干扰数据生成/tests/TestDatasetGeneration.m`

- [ ] **Step 1: Write a failing full-power readback test**

Create a small four-class full-power fixture, save it, and assert:

```matlab
verification = verify_saved_closedset_split( ...
    outputPath, manifest, cfg, schema);
testCase.verifyTrue(verification.power_truth_checked);
testCase.verifyTrue(verification.power_truth_equal);
testCase.verifyEqual(verification.power_truth_mode, ...
    'independent_range');
```

Corrupt one active type's parameter JNR and expect `JammingDataset:SavedFileVerificationFailed`.

- [ ] **Step 2: Run and verify failure**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestDatasetGeneration.m'),'Name','*fullPowerReadbackVerifiesComponentTruth*'); disp(r); assert(any([r.Failed]));"
```

Expected: ordinary full-power configs currently return `power_truth_checked=false`.

- [ ] **Step 3: Detect the power truth mode**

Replace the sweep-only boolean with:

```matlab
powerTruthMode = detect_power_truth_mode(cfg);
powerTruthChecked = ~strcmp(powerTruthMode, 'none');
```

Add:

```matlab
function mode = detect_power_truth_mode(cfg)
    mode = 'none';
    if isfield(cfg, 'experiment') && isstruct(cfg.experiment)
        if isfield(cfg.experiment, 'power_sweep')
            mode = 'power_sweep';
        elseif isfield(cfg.experiment, 'full_power')
            mode = 'independent_range';
        end
    end
end
```

- [ ] **Step 4: Add full-power truth verification**

Dispatch by `powerTruthMode`. For every sample, require effective JNRs within the configured range, target power equal to noise power plus effective JNR, measured power within tolerance, and each active type's parameter JNR equal to the component truth.

Core checks:

```matlab
actualJnr = effectiveJnr{sampleIndex};
if ~numeric_row_in_range(actualJnr, profile.JNR_range_dB)
    fail_verification('第%d条有效JNR超出全功率范围。', sampleIndex);
end
expectedTarget = noisePower_dB + actualJnr;
if ~numeric_row_matches(targetPower{sampleIndex}, ...
        expectedTarget, tolerance_dB)
    fail_verification('第%d条目标功率与有效JNR不一致。', sampleIndex);
end
if ~numeric_row_matches(measuredPower{sampleIndex}, ...
        expectedTarget, tolerance_dB)
    fail_verification('第%d条实测功率超出允许误差。', sampleIndex);
end
```

Return `power_truth_mode`, `power_truth_checked`, and `power_truth_equal` in the verification struct.

- [ ] **Step 5: Run tests and commit**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=runtests(fullfile(root,'tests')); assertSuccess(r);"
git add -- '可配置组合干扰数据生成/core/verify_saved_closedset_split.m' '可配置组合干扰数据生成/tests/TestDatasetGeneration.m'
git commit -m "test: verify full-power truth on readback"
```

---

### Task 6: Add Family Runners and Public Entries

**Files:**
- Create: `可配置组合干扰数据生成/core/run_subset_dataset_family.m`
- Create: `可配置组合干扰数据生成/core/audit_closedset_seed_domains.m`
- Create: `可配置组合干扰数据生成/generate_closedset_4class_datasets.m`
- Create: `可配置组合干扰数据生成/generate_closedset_5class_datasets.m`
- Create: `可配置组合干扰数据生成/generate_closedset_4class_5class_all.m`
- Test: `可配置组合干扰数据生成/tests/TestDatasetGeneration.m`

- [ ] **Step 1: Write failing public dry-run tests**

```matlab
function publicFourFiveClassDryRunsAreWriteFree(testCase)
four = generate_closedset_4class_datasets( ...
    false, 'all', [10,50], 'both', 'resume');
testCase.verifyEqual(four.full_power.train_summary.sample_count, 6000);
testCase.verifyEqual( ...
    four.power_sweep.levels(1).train_summary.sample_count, 6000);
testCase.verifyTrue(four.power_sweep.paired_seed_verified);
five = generate_closedset_5class_datasets( ...
    false, 'all', [10,50], 'both', 'resume');
testCase.verifyEqual(five.full_power.test_summary.sample_count, 12400);
end
```

Dry runs must not create the formal `4类` or `5类` directories.

Run:

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestDatasetGeneration.m'),'Name','*publicFourFiveClassDryRunsAreWriteFree*'); disp(r); assert(any([r.Failed]));"
```

Expected: the public entries are undefined.

- [ ] **Step 2: Implement one-family orchestration**

Create:

```matlab
function result = run_subset_dataset_family( ...
        fullCfg, sweepConfigs, executeGeneration, datasetSelection, ...
        splitSelection, existingPolicy)
```

Normalize `datasetSelection` to `all`, `full_power`, or `power_sweep`. Call `run_closedset_splits` for full power and `run_power_sweep_levels` for the sweep. Return fixed fields for unselected sections with status `not_selected`.

- [ ] **Step 3: Implement four-class and five-class public entries**

Both entries use:

```matlab
function result = generate_closedset_4class_datasets( ...
        executeGeneration, datasetSelection, levels_dB, ...
        splitSelection, existingPolicy)
```

Defaults are `true, 'all', 10:5:50, 'both', 'resume'`. Build sweep configs from the approved wrapper and call the family runner.

Each public entry must call a local `add_local_paths` helper before using core
functions, matching the existing six-class entries so it works from any current
directory.

- [ ] **Step 4: Implement pairwise seed-domain auditing**

Create `audit_closedset_seed_domains.m`:

```matlab
function audit = audit_closedset_seed_domains(configs, names)
    vectors = cell(2 * numel(configs), 1);
    labels = cell(size(vectors));
    cursor = 0;
    for index = 1:numel(configs)
        cfg = normalize_and_validate_config(configs{index}, 'balanced');
        registry = build_jamming_registry(cfg);
        for split = {'train','test'}
            manifest = build_closedset_manifest(cfg, registry, split{1});
            cursor = cursor + 1;
            vectors{cursor} = manifest.sample_seed;
            labels{cursor} = [names{index} ':' split{1}];
        end
    end
    for left = 1:numel(vectors)
        for right = left + 1:numel(vectors)
            if ~isempty(intersect(vectors{left}, vectors{right}))
                error('JammingSubset:SeedDomainOverlap', ...
                    '%s与%s的sample_seed发生交集。', ...
                    labels{left}, labels{right});
            end
        end
    end
    audit = struct('verified', true, ...
        'domain_count', numel(vectors), 'labels', {labels});
end
```

- [ ] **Step 5: Implement the serial master entry**

Use the exact signature:

```matlab
function result = generate_closedset_4class_5class_all( ...
        executeGeneration, datasetSelection, levels_dB, ...
        splitSelection, existingPolicy)
```

Apply the same defaults as the family entries, call four-class first and then
five-class, and audit representative full and sweep configs:

```matlab
configs = {create_closedset_4class_config(), ...
    create_power_sweep_4class_config(10), ...
    create_closedset_5class_config(), ...
    create_power_sweep_5class_config(10)};
names = {'4class_full','4class_sweep', ...
    '5class_full','5class_sweep'};
result.seed_audit = audit_closedset_seed_domains(configs, names);
```

Return the fixed top-level structure:

```matlab
result = struct('four_class', fourResult, ...
    'five_class', fiveResult, 'seed_audit', seedAudit);
```

- [ ] **Step 6: Run all tests and commit**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=runtests(fullfile(root,'tests')); assertSuccess(r);"
git add -- '可配置组合干扰数据生成/core/run_subset_dataset_family.m' '可配置组合干扰数据生成/core/audit_closedset_seed_domains.m' '可配置组合干扰数据生成/generate_closedset_4class_datasets.m' '可配置组合干扰数据生成/generate_closedset_5class_datasets.m' '可配置组合干扰数据生成/generate_closedset_4class_5class_all.m' '可配置组合干扰数据生成/tests/TestDatasetGeneration.m'
git commit -m "feat: add four and five class dataset entries"
```

---

### Task 7: Add Small End-to-End Atomic Generation Fixtures

**Files:**
- Modify: `可配置组合干扰数据生成/tests/TestDatasetGeneration.m`

- [ ] **Step 1: Add a four-path end-to-end test**

Use temporary output roots and set `samples_per_pattern=1`, `save_legacy=false`. Generate four-class full power, four-class 10/50 dB sweep, five-class full power, and five-class 10/50 dB sweep.

Assert category-specific filenames, final existence, no partials, full-power truth mode, sweep truth mode, 15/31 pattern counts, paired sweep seeds, and pairwise experiment seed isolation.

Use this concrete test structure:

```matlab
function fourFiveClassSmallEndToEnd(testCase)
    tempRoot = tempname;
    mkdir(tempRoot);
    cleanup = onCleanup(@() rmdir(tempRoot, 's')); %#ok<NASGU>

    fourFull = create_closedset_4class_config();
    fourFull.output_dir = fullfile(tempRoot, '4full');
    fourFull.output.save_legacy = false;
    fourFull.dataset.samples_per_pattern = 1;
    fourFull.dataset.num_samples = 15;
    fourFull = normalize_and_validate_config(fourFull, 'balanced');
    fourRegistry = build_jamming_registry(fourFull);
    fourSchema = build_parameter_schema(fourRegistry);
    fourResult = run_closedset_splits(fourFull, fourRegistry, ...
        fourSchema, true, 'both', 'overwrite');

    fourSweep = cell(2,1);
    levels = [10,50];
    for k = 1:2
        cfg = create_power_sweep_4class_config(levels(k));
        cfg.output_dir = fullfile(tempRoot, '4sweep', sprintf( ...
            'level_%02d__jnr_%ddB', k, levels(k)));
        cfg.output.save_legacy = false;
        cfg.dataset.samples_per_pattern = 1;
        cfg.dataset.num_samples = 15;
        fourSweep{k} = cfg;
    end
    fourSweepResult = run_power_sweep_levels( ...
        fourSweep, true, 'both', 'overwrite', true);

    fiveFull = create_closedset_5class_config();
    fiveFull.output_dir = fullfile(tempRoot, '5full');
    fiveFull.output.save_legacy = false;
    fiveFull.dataset.samples_per_pattern = 1;
    fiveFull.dataset.num_samples = 31;
    fiveFull = normalize_and_validate_config(fiveFull, 'balanced');
    fiveRegistry = build_jamming_registry(fiveFull);
    fiveSchema = build_parameter_schema(fiveRegistry);
    fiveResult = run_closedset_splits(fiveFull, fiveRegistry, ...
        fiveSchema, true, 'both', 'overwrite');

    fiveSweep = cell(2,1);
    for k = 1:2
        cfg = create_power_sweep_5class_config(levels(k));
        cfg.output_dir = fullfile(tempRoot, '5sweep', sprintf( ...
            'level_%02d__jnr_%ddB', k, levels(k)));
        cfg.output.save_legacy = false;
        cfg.dataset.samples_per_pattern = 1;
        cfg.dataset.num_samples = 31;
        fiveSweep{k} = cfg;
    end
    fiveSweepResult = run_power_sweep_levels( ...
        fiveSweep, true, 'both', 'overwrite', true);

    testCase.verifyEqual( ...
        fourResult.splits.train.verification.power_truth_mode, ...
        'independent_range');
    testCase.verifyEqual( ...
        fiveResult.splits.test.verification.power_truth_mode, ...
        'independent_range');
    testCase.verifyTrue(fourSweepResult.paired_seed_verified);
    testCase.verifyTrue(fiveSweepResult.paired_seed_verified);
    testCase.verifyEqual(fourResult.train_summary.pattern_count, 15);
    testCase.verifyEqual(fiveResult.train_summary.pattern_count, 31);
    testCase.verifyEmpty(dir(fullfile(tempRoot, '**', '*.partial.mat')));
end
```

- [ ] **Step 2: Run the focused test**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=runtests(fullfile(root,'tests','TestDatasetGeneration.m'),'Name','*fourFiveClassSmallEndToEnd*'); assertSuccess(r);"
```

- [ ] **Step 3: Run the full suite and commit**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=runtests(fullfile(root,'tests')); assertSuccess(r);"
git add -- '可配置组合干扰数据生成/tests/TestDatasetGeneration.m'
git commit -m "test: cover four and five class atomic generation"
```

---

### Task 8: Add Program Documentation

**Files:**
- Create: `可配置组合干扰数据生成/README_四类五类闭集数据.md`
- Modify: `可配置组合干扰数据生成/README_功率不均衡九挡扫描.md`

- [ ] **Step 1: Write the new README**

Document the exact type lists, power semantics, output tree, all public entries, dry-run and formal examples, 15/31 combinations, 400 samples per pattern, seed rules, MAT fields, partial/resume behavior, and FMB Nyquist truth.

- [ ] **Step 2: Add a six-class compatibility note**

State that the original six-class entry and filenames remain unchanged.

- [ ] **Step 3: Verify and commit**

```powershell
rg -n "generate_closedset_4class|generate_closedset_5class|368,000|FMB" '可配置组合干扰数据生成/README_四类五类闭集数据.md'
git diff --check
git add -- '可配置组合干扰数据生成/README_四类五类闭集数据.md' '可配置组合干扰数据生成/README_功率不均衡九挡扫描.md'
git commit -m "docs: explain four and five class datasets"
```

---

### Task 9: Complete Verification Before Formal Generation

**Files:**
- Verify all modified and created MATLAB files.

- [ ] **Step 1: Run the complete MATLAB suite**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=runtests(fullfile(root,'tests')); assertSuccess(r); fprintf('FINAL_TESTS=%d\n',numel(r));"
```

- [ ] **Step 2: Run Code Analyzer**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; files=dir(fullfile(root,'**','*.m')); issueCount=0; for k=1:numel(files); issues=checkcode(fullfile(files(k).folder,files(k).name),'-id'); issueCount=issueCount+numel(issues); end; fprintf('MATLAB_FILES=%d CHECKCODE_ISSUES=%d\n',numel(files),issueCount); assert(issueCount==0);"
```

- [ ] **Step 3: Run full-scale write-free dry runs**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=generate_closedset_4class_5class_all(false,'all',10:5:50,'both','resume'); assert(r.seed_audit.verified); fprintf('DRYRUN_4_5_COMPLETE\n');"
```

Expected: four-class splits 6,000/15; five-class splits 12,400/31; no MAT/CSV writes; eight disjoint split domains.

---

### Task 10: Formal Serial Generation

**Files generated:**
- `G:\论文相关\干扰数据\复合干扰数据0716\4类\...`
- `G:\论文相关\干扰数据\复合干扰数据0716\5类\...`

- [ ] **Step 1: Preflight disk, process, and path state**

Require at least 50 GiB free, zero matching MATLAB generators, and zero conflicting MAT/status files. Do not remove unexpected files.

- [ ] **Step 2: Start exactly one MATLAB process**

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(root); result=generate_closedset_4class_5class_all(true,'all',10:5:50,'both','resume'); disp(result.seed_audit); fprintf('FOUR_FIVE_CLASS_GENERATION_COMPLETE\n');"
```

Execution order: four full, four sweep 10–50, five full, five sweep 10–50; train before test everywhere.

- [ ] **Step 3: Monitor without starting another generator**

Inspect the current partial size, final count, sweep CSV statuses, MATLAB liveness, and free space. On failure, preserve finals and resume with the same command after correcting the exact cause.

---

### Task 11: Independent Final Readback and Seed Audit

**Files:**
- Read all 40 generated MAT files.

- [ ] **Step 1: Rebuild expected configs and manifests in a fresh MATLAB process**

Run the public master entry in `resume` mode from a fresh MATLAB process. Every
existing final is compared with a freshly rebuilt expected config and manifest
through `verify_saved_closedset_split` before it can be marked `reused`:

```powershell
matlab -batch "root='G:\论文相关\干扰数据\干扰信号生成程序\可配置组合干扰数据生成'; addpath(genpath(root)); r=generate_closedset_4class_5class_all(true,'all',10:5:50,'both','resume'); assert(strcmp(r.four_class.full_power.splits.train.status,'reused')); assert(strcmp(r.five_class.full_power.splits.test.status,'reused')); assert(all(strcmp({r.four_class.power_sweep.levels.train_status},'reused'))); assert(all(strcmp({r.five_class.power_sweep.levels.test_status},'reused'))); assert(r.seed_audit.verified); fprintf('INDEPENDENT_40FILE_READBACK_COMPLETE\n');"
```

- [ ] **Step 2: Assert exact scale and truth**

Four-class: 6,000 samples, 15 patterns, 400 per pattern. Five-class: 12,400 samples, 31 patterns, 400 per pattern. All verification results require `power_truth_equal=true`.

- [ ] **Step 3: Compare paired fields across nine levels**

Compare exactly:

```matlab
{'sample_id','split','sample_seed','active_type_ids', ...
 'active_type_codes','combination_id','combination_size', ...
 'component_sampling_weight','pattern_sampling_weight', ...
 'imbalance_flags','unknown_present','unknown_component_count'}
```

Build expected manifests for all nine levels in the same fresh process and
compare these table slices with `isequaln`. Step 1 already proves every saved
manifest equals its freshly built expected manifest, so this also proves the
saved files have identical paired fields within each class count and split.

- [ ] **Step 4: Re-run seed and filesystem audits**

Expected: `domain_count=8`, `verified=true`, 20 MAT files under each class folder, 40 total, zero partials, and two nine-row sweep CSVs fully generated.

```powershell
$root='G:\论文相关\干扰数据\复合干扰数据0716'
$four=@(Get-ChildItem -LiteralPath (Join-Path $root '4类') -Recurse -File -Filter '*.mat')
$five=@(Get-ChildItem -LiteralPath (Join-Path $root '5类') -Recurse -File -Filter '*.mat')
$partial=@(Get-ChildItem -LiteralPath $root -Recurse -File -Filter '*.partial.mat')
if($four.Count-ne20 -or $five.Count-ne20 -or $partial.Count-ne0){
    throw 'Final file count mismatch'
}
```

---

### Task 12: Generate SHA-256 Inventories and Data Descriptions

**Files generated:**
- `G:\论文相关\干扰数据\复合干扰数据0716\4类\四类文件SHA256清单.csv`
- `G:\论文相关\干扰数据\复合干扰数据0716\5类\五类文件SHA256清单.csv`
- `G:\论文相关\干扰数据\复合干扰数据0716\四类五类文件SHA256总清单.csv`
- `G:\论文相关\干扰数据\复合干扰数据0716\4类\四类组合干扰数据集说明.md`
- `G:\论文相关\干扰数据\复合干扰数据0716\5类\五类组合干扰数据集说明.md`
- `G:\论文相关\干扰数据\复合干扰数据0716\四类五类数据集总说明.md`

- [ ] **Step 1: Compute SHA-256**

Record `class_count,dataset_kind,level_index,sweep_JNR_dB,split,file_path,bytes,GiB,SHA256,last_write_time`. Write 20-row class inventories and one 40-row combined inventory.

```powershell
$root='G:\论文相关\干扰数据\复合干扰数据0716'
$files=Get-ChildItem -LiteralPath (Join-Path $root '4类'),(Join-Path $root '5类') -Recurse -File -Filter '*.mat'
$rows=foreach($file in $files){
    $classCount=if($file.FullName -like '*\4类\*'){4}else{5}
    $kind=if($file.FullName -like '*\full_power_10_30dB_v1\*'){'full_power'}else{'power_sweep'}
    $split=if($file.BaseName -like '*_train'){'train'}else{'test'}
    $hash=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    [pscustomobject]@{class_count=$classCount;dataset_kind=$kind;split=$split;file_path=$file.FullName;bytes=$file.Length;GiB=[math]::Round($file.Length/1GB,6);SHA256=$hash;last_write_time=$file.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')}
}
$rows|Export-Csv -LiteralPath (Join-Path $root '四类五类文件SHA256总清单.csv') -NoTypeInformation -Encoding utf8
$rows|Where-Object class_count -eq 4|Export-Csv -LiteralPath (Join-Path $root '4类\四类文件SHA256清单.csv') -NoTypeInformation -Encoding utf8
$rows|Where-Object class_count -eq 5|Export-Csv -LiteralPath (Join-Path $root '5类\五类文件SHA256清单.csv') -NoTypeInformation -Encoding utf8
```

- [ ] **Step 2: Write class-specific descriptions**

Include selected types, power semantics, counts, parameters, seed rules, directory tree, MAT sizes, byte totals, independent validation, SHA path, and FMB Nyquist semantics.

- [ ] **Step 3: Write and verify the root summary**

Summarize 40 files and 368,000 records while keeping the four experiments distinct. Read back all CSV/Markdown artifacts and assert 20+20+40 rows, 40 unique hashes, matching byte totals, existing paths, and UTF-8 readability.

---

### Task 13: Final Repository Verification and Integration

**Files:**
- All intended implementation and documentation files.

- [ ] **Step 1: Run tests and Code Analyzer one final time**

Expected: all tests pass and zero analyzer issues.

- [ ] **Step 2: Inspect the diff and preserve unrelated work**

```powershell
git status --short
git diff --check
git log --oneline --decorate -12
```

Do not stage the pre-existing unrelated Markdown file. Generated MAT files remain outside the repository.

- [ ] **Step 3: Commit final program documentation if needed**

Stage only the exact program and repository documentation files listed in the
File Map; never use `git add .`. Commit with
`docs: finalize four and five class dataset workflow`.

- [ ] **Step 4: Integrate and push**

Use the finishing-development-branch workflow. Integrate only after verification evidence is recorded, then push to the existing private GitHub repository when network connectivity is available.

---

## Completion Criteria

- generalized code remains backward-compatible with the six-class dataset;
- all MATLAB tests pass and Code Analyzer reports zero issues;
- full dry-run counts and seed audits pass without writing files;
- 40 formal MAT files exist and zero partial files remain;
- all 40 independent readbacks pass;
- both nine-level sweeps are paired internally;
- all four experiment seed domains and their train/test splits are pairwise disjoint;
- 40 unique SHA-256 hashes are recorded;
- three Chinese data-description files are present and readable;
- new and modified program files contain complete Chinese explanatory comments;
- unrelated user files remain untouched.
