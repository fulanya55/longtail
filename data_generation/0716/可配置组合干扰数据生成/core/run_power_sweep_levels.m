function result = run_power_sweep_levels(configs, executeGeneration, ...
        splitSelection, existingPolicy, writeStatus)
%RUN_POWER_SWEEP_LEVELS 串行执行一组已配置的闭集功率扫描挡位。
%   RESULT = RUN_POWER_SWEEP_LEVELS(CONFIGS, EXECUTEGENERATION,
%   SPLITSELECTION, EXISTINGPOLICY, WRITESTATUS) 接收配置 cell 列，每个配置
%   必须包含合法 cfg.experiment.power_sweep 且输出目录具有同一父目录。
%   本函数复用 RUN_CLOSEDSET_SPLITS 的原子保存、partial和resume机制。
%
%   无论正式或干运行，函数都会重新构建每挡所选 split 的期望 manifest，
%   对完整 sample_seed 向量执行挡位间 ISEQUAL，并在 both 模式核对train与
%   test无交集。仅把计数、首尾种子和和校验值作为轻量签名返回。
%
%   WRITESTATUS 仅在 EXECUTEGENERATION=true 时生效。状态CSV在每挡开始前
%   和完成后原子更新，因此中断后可以看到最后进入的挡位；最终完成判定
%   仍以各 MAT 文件的增强回载结果为准。

    configs = normalize_configs(configs);
    executeGeneration = normalize_execute_flag(executeGeneration);
    splitSelection = normalize_choice(splitSelection, ...
        {'both', 'train', 'test'}, ...
        'JammingDataset:InvalidSplitSelection', 'splitSelection');
    existingPolicy = normalize_choice(existingPolicy, ...
        {'resume', 'error', 'overwrite'}, ...
        'JammingDataset:InvalidExistingPolicy', 'existingPolicy');
    writeStatus = normalize_execute_flag(writeStatus);
    writeStatus = writeStatus && executeGeneration;

    normalizedConfigs = cell(size(configs));
    sweepRoot = '';
    levels = repmat(initialize_level_result(), numel(configs), 1);
    for index = 1:numel(configs)
        cfg = normalize_and_validate_config(configs{index}, 'balanced');
        validate_power_config(cfg);
        currentRoot = fileparts(cfg.output_dir);
        if index == 1
            sweepRoot = currentRoot;
        elseif ~strcmp(currentRoot, sweepRoot)
            error('JammingPowerSweep:InconsistentOutputRoot', ...
                '全部功率挡位的 output_dir 必须具有同一父目录。');
        end
        normalizedConfigs{index} = cfg;
        levels(index) = initialize_level_result(cfg);
    end

    result = struct();
    result.executed = executeGeneration;
    result.levels_dB = reshape(cellfun(@(cfg) ...
        cfg.experiment.power_sweep.sweep_JNR_dB, ...
        normalizedConfigs), 1, []);
    result.split_selection = splitSelection;
    result.existing_policy = existingPolicy;
    result.sweep_root = sweepRoot;
    result.manifest_path = fullfile(sweepRoot, 'power_sweep_manifest.csv');
    result.paired_seed_verified = false;
    result.levels = levels;

    if writeStatus
        if ~isfolder(sweepRoot)
            mkdir(sweepRoot);
        end
        write_power_sweep_manifest(sweepRoot, result.levels);
    end

    referenceTrainSeeds = [];
    referenceTestSeeds = [];
    for index = 1:numel(normalizedConfigs)
        cfg = normalizedConfigs{index};
        registry = build_jamming_registry(cfg);
        schema = build_parameter_schema(registry);
        result.levels(index).train_status = ...
            initial_selected_status(splitSelection, 'train');
        result.levels(index).test_status = ...
            initial_selected_status(splitSelection, 'test');
        if writeStatus
            write_power_sweep_manifest(sweepRoot, result.levels);
        end

        runResult = run_closedset_splits(cfg, registry, schema, ...
            executeGeneration, splitSelection, existingPolicy);
        [trainSeeds, testSeeds, trainSignature, testSignature] = ...
            paired_seed_evidence(cfg, registry, splitSelection);
        referenceTrainSeeds = verify_paired_vector( ...
            referenceTrainSeeds, trainSeeds, 'train');
        referenceTestSeeds = verify_paired_vector( ...
            referenceTestSeeds, testSeeds, 'test');
        if ~isempty(trainSeeds) && ~isempty(testSeeds) && ...
                ~isempty(intersect(trainSeeds, testSeeds))
            error('JammingPowerSweep:TrainTestSeedOverlap', ...
                '功率扫描 train 与 test 的 sample_seed 发生交集。');
        end

        result.levels(index) = compact_level_result( ...
            result.levels(index), runResult, ...
            trainSignature, testSignature);
        if writeStatus
            write_power_sweep_manifest(sweepRoot, result.levels);
        end
        clear runResult trainSeeds testSeeds registry schema
    end
    result.paired_seed_verified = true;
end

function configs = normalize_configs(configs)
%NORMALIZE_CONFIGS 规范非空配置cell列。

    if ~(iscell(configs) && ~isempty(configs) && ...
            all(cellfun(@(value) isstruct(value) && isscalar(value), configs)))
        error('JammingPowerSweep:InvalidConfigs', ...
            'configs 必须是由标量配置结构体组成的非空cell。');
    end
    configs = reshape(configs, [], 1);
end

function validate_power_config(cfg)
%VALIDATE_POWER_CONFIG 触发功率剖面的严格配置校验。

    if ~(isfield(cfg, 'experiment') && ...
            isfield(cfg.experiment, 'power_sweep'))
        error('JammingPowerSweep:InvalidProfile', ...
            '每个挡位配置都必须包含 experiment.power_sweep。');
    end
    registry = build_jamming_registry(cfg);
    probe = build_balanced_manifest_for_probe(cfg, registry);
    apply_power_sweep_jnr_profile(probe, cfg);
end

function manifest = build_balanced_manifest_for_probe(cfg, registry)
%BUILD_BALANCED_MANIFEST_FOR_PROBE 用一行合法表触发剖面校验而不分配全清单。

    codes = cfg.types.enabled;
    registryCodes = {registry.code};
    ids = cellfun(@(code) find(strcmp(registryCodes, code), 1), codes);
    componentCount = numel(codes);
    manifest = table(1, {'train'}, double(cfg.seed) + 1, {ids}, {codes}, ...
        {strjoin(codes, '+')}, componentCount, {zeros(1, componentCount)}, ...
        {false(1, componentCount)}, 1, 1, ...
        {logical([false, false, false])}, false, 0, 'VariableNames', { ...
        'sample_id', 'split', 'sample_seed', 'active_type_ids', ...
        'active_type_codes', 'combination_id', 'combination_size', ...
        'component_target_power_dB', 'weak_component_mask', ...
        'component_sampling_weight', 'pattern_sampling_weight', ...
        'imbalance_flags', 'unknown_present', 'unknown_component_count'});
end

function value = normalize_execute_flag(value)
%NORMALIZE_EXECUTE_FLAG 规范逻辑或0/1数值标量。

    valid = (islogical(value) || isnumeric(value)) && isscalar(value) && ...
        isreal(value) && isfinite(double(value)) && ...
        ismember(double(value), [0, 1]);
    if ~valid
        error('JammingPowerSweep:InvalidExecuteFlag', ...
            '执行和状态写入开关必须是逻辑标量或数值0/1。');
    end
    value = logical(value);
end

function value = normalize_choice(value, allowed, errorId, path)
%NORMALIZE_CHOICE 规范有限选项文本。

    if isstring(value) && isscalar(value) && ~ismissing(value)
        value = char(value);
    elseif ~(ischar(value) && isrow(value))
        error(errorId, '%s 必须是文本标量。', path);
    end
    value = lower(strtrim(value));
    if ~ismember(value, allowed)
        error(errorId, '%s 仅允许：%s。', path, strjoin(allowed, '、'));
    end
end

function level = initialize_level_result(cfg)
%INITIALIZE_LEVEL_RESULT 创建固定字段的轻量挡位结果。

    if nargin < 1
        levelIndex = 0;
        sweepJnr_dB = NaN;
        fixedJnr_dB = NaN;
        outputDir = '';
        expectedPatternCount = 0;
        samplesPerPattern = 0;
        expectedSampleCount = 0;
    else
        profile = cfg.experiment.power_sweep;
        levelIndex = profile.level_index;
        sweepJnr_dB = profile.sweep_JNR_dB;
        fixedJnr_dB = profile.fixed_JNR_dB;
        outputDir = cfg.output_dir;
        expectedPatternCount = count_all_patterns(cfg);
        samplesPerPattern = double(cfg.dataset.samples_per_pattern);
        expectedSampleCount = expectedPatternCount * samplesPerPattern;
    end
    level = struct( ...
        'level_index', levelIndex, ...
        'sweep_JNR_dB', sweepJnr_dB, ...
        'fixed_JNR_dB', fixedJnr_dB, ...
        'output_dir', outputDir, ...
        'train_path', fullfile(outputDir, 'closedset_6class_train.mat'), ...
        'test_path', fullfile(outputDir, 'closedset_6class_test.mat'), ...
        'expected_samples_per_split', expectedSampleCount, ...
        'expected_pattern_count', expectedPatternCount, ...
        'samples_per_pattern', samplesPerPattern, ...
        'train_status', 'pending', ...
        'test_status', 'pending', ...
        'train_summary', struct(), ...
        'test_summary', struct(), ...
        'train_seed_signature', empty_seed_signature(), ...
        'test_seed_signature', empty_seed_signature(), ...
        'train_verification', struct('performed', false), ...
        'test_verification', struct('performed', false));
end

function count = count_all_patterns(cfg)
%COUNT_ALL_PATTERNS 计算all模式在配置阶数范围内的组合总数。

    if ~strcmp(cfg.dataset.combination_mode, 'all')
        error('JammingPowerSweep:InvalidProfile', ...
            '九挡功率扫描仅支持 dataset.combination_mode="all"。');
    end
    typeCount = numel(cfg.types.enabled);
    count = 0;
    for componentCount = cfg.dataset.min_components: ...
            cfg.dataset.max_components
        count = count + nchoosek(typeCount, componentCount);
    end
end

function status = initial_selected_status(selection, splitName)
%INITIAL_SELECTED_STATUS 返回running或not_selected。

    if strcmp(selection, 'both') || strcmp(selection, splitName)
        status = 'running';
    else
        status = 'not_selected';
    end
end

function [trainSeeds, testSeeds, trainSignature, testSignature] = ...
        paired_seed_evidence(cfg, registry, selection)
%PAIRED_SEED_EVIDENCE 构建所选split的完整种子向量和轻量签名。

    trainSeeds = [];
    testSeeds = [];
    trainSignature = empty_seed_signature();
    testSignature = empty_seed_signature();
    if strcmp(selection, 'both') || strcmp(selection, 'train')
        manifest = build_closedset_manifest(cfg, registry, 'train');
        trainSeeds = manifest.sample_seed;
        trainSignature = make_seed_signature(trainSeeds);
        clear manifest
    end
    if strcmp(selection, 'both') || strcmp(selection, 'test')
        manifest = build_closedset_manifest(cfg, registry, 'test');
        testSeeds = manifest.sample_seed;
        testSignature = make_seed_signature(testSeeds);
    end
end

function reference = verify_paired_vector(reference, current, splitName)
%VERIFY_PAIRED_VECTOR 精确比较挡位间完整sample_seed向量。

    if isempty(current)
        return;
    end
    if isempty(reference)
        reference = current;
    elseif ~isequal(reference, current)
        error('JammingPowerSweep:UnpairedSeeds', ...
            '%s 的 sample_seed 在功率挡位间不一致。', splitName);
    end
end

function signature = make_seed_signature(seeds)
%MAKE_SEED_SIGNATURE 返回不替代完整比较的轻量可读种子摘要。

    seeds = double(seeds(:));
    signature = struct('count', numel(seeds), 'first', seeds(1), ...
        'last', seeds(end), 'sum', sum(seeds));
end

function signature = empty_seed_signature()
%EMPTY_SEED_SIGNATURE 创建未选择split的空签名。

    signature = struct('count', 0, 'first', NaN, 'last', NaN, 'sum', 0);
end

function level = compact_level_result( ...
        level, runResult, trainSignature, testSignature)
%COMPACT_LEVEL_RESULT 从单挡执行结果提取轻量持久字段。

    level.train_status = runResult.splits.train.status;
    level.test_status = runResult.splits.test.status;
    level.train_summary = runResult.splits.train.summary;
    level.test_summary = runResult.splits.test.summary;
    level.train_seed_signature = trainSignature;
    level.test_seed_signature = testSignature;
    level.train_verification = runResult.splits.train.verification;
    level.test_verification = runResult.splits.test.verification;
end
