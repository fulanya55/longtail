function outputPath = write_power_sweep_manifest(sweepRoot, levels)
%WRITE_POWER_SWEEP_MANIFEST 原子写入九挡功率扫描状态CSV。
%   OUTPUTPATH = WRITE_POWER_SWEEP_MANIFEST(SWEEPROOT, LEVELS) 把轻量挡位
%   结构数组写为 SWEEPROOT/power_sweep_manifest.csv。函数先写同目录精确
%   partial名称，成功后再 MOVEFILE 覆盖最终CSV，避免中断留下半张状态表。

    sweepRoot = normalize_root(sweepRoot);
    validate_levels(levels);
    if ~isfolder(sweepRoot)
        mkdir(sweepRoot);
    end
    outputPath = fullfile(sweepRoot, 'power_sweep_manifest.csv');
    partialPath = fullfile(sweepRoot, 'power_sweep_manifest.partial.csv');
    if isfile(partialPath)
        delete(partialPath);
    end

    levelIndex = reshape([levels.level_index], [], 1);
    sweepJnr_dB = reshape([levels.sweep_JNR_dB], [], 1);
    fixedJnr_dB = reshape([levels.fixed_JNR_dB], [], 1);
    outputDir = reshape({levels.output_dir}, [], 1);
    trainPath = reshape({levels.train_path}, [], 1);
    testPath = reshape({levels.test_path}, [], 1);
    trainStatus = reshape({levels.train_status}, [], 1);
    testStatus = reshape({levels.test_status}, [], 1);
    expectedSamples = reshape( ...
        [levels.expected_samples_per_split], [], 1);
    expectedPatterns = reshape( ...
        [levels.expected_pattern_count], [], 1);
    samplesPerPattern = reshape([levels.samples_per_pattern], [], 1);
    statusTable = table(levelIndex, sweepJnr_dB, fixedJnr_dB, ...
        outputDir, trainPath, testPath, expectedSamples, expectedPatterns, ...
        samplesPerPattern, trainStatus, testStatus, 'VariableNames', { ...
        'level_index', 'sweep_JNR_dB', 'fixed_JNR_dB', 'output_dir', ...
        'train_path', 'test_path', 'expected_samples_per_split', ...
        'expected_pattern_count', 'samples_per_pattern', ...
        'train_status', 'test_status'});
    writetable(statusTable, partialPath, 'Encoding', 'UTF-8');
    [success, message] = movefile(partialPath, outputPath, 'f');
    if ~success
        error('JammingPowerSweep:StatusWriteFailed', ...
            '无法原子提交功率扫描状态CSV：%s', message);
    end
end

function root = normalize_root(root)
%NORMALIZE_ROOT 规范非空输出根目录文本。

    if isstring(root) && isscalar(root) && ~ismissing(root)
        root = char(root);
    elseif ~(ischar(root) && isrow(root))
        error('JammingPowerSweep:StatusWriteFailed', ...
            'sweepRoot 必须是非空文本标量。');
    end
    root = strtrim(root);
    if isempty(root)
        error('JammingPowerSweep:StatusWriteFailed', ...
            'sweepRoot 不能为空。');
    end
end

function validate_levels(levels)
%VALIDATE_LEVELS 校验写表需要的固定字段。

    required = {'level_index', 'sweep_JNR_dB', 'fixed_JNR_dB', ...
        'output_dir', 'train_path', 'test_path', ...
        'expected_samples_per_split', 'expected_pattern_count', ...
        'samples_per_pattern', 'train_status', 'test_status'};
    if ~isstruct(levels) || isempty(levels) || ...
            ~all(isfield(levels, required))
        error('JammingPowerSweep:StatusWriteFailed', ...
            'levels 缺少写入状态CSV所需的固定字段。');
    end
end
