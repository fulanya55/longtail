function cfg = create_power_sweep_subset_config( ...
        selectedTypes, fixedCodes, sweptCodes, seed, ...
        sweepRoot, sweepJnr_dB)
%CREATE_POWER_SWEEP_SUBSET_CONFIG 创建类型子集的一个配对功率挡位。
%   CFG = CREATE_POWER_SWEEP_SUBSET_CONFIG(SELECTEDTYPES, FIXEDCODES,
%   SWEPTCODES, SEED, SWEEPROOT, SWEEPJNR_DB) 以通用闭集子集配置为基础，
%   把固定组最终JNR锁定为30 dB，把扫描组最终JNR设置为10:5:50中的
%   指定挡位。固定组和扫描组必须无交集，且并集严格等于启用类型。
%
%   所有挡位使用同一SEED和标称30 dB生成路径，实际差异由manifest中的
%   确定性相对功率偏置实现，因此同一split和sample_id的组合、分量子流、
%   噪声子流及非JNR参数保持配对。函数只构造配置，不创建目录或文件。

    selectedTypes = normalize_code_list( ...
        selectedTypes, 'selectedTypes', false);
    fixedCodes = normalize_code_list( ...
        fixedCodes, 'fixedCodes', false);
    sweptCodes = normalize_code_list( ...
        sweptCodes, 'sweptCodes', false);
    validate_partition(selectedTypes, fixedCodes, sweptCodes);
    sweepJnr_dB = validate_level(sweepJnr_dB);

    standardLevels_dB = 10:5:50;
    levelIndex = find( ...
        standardLevels_dB == sweepJnr_dB, 1, 'first');
    cfg = create_closedset_subset_config( ...
        selectedTypes, seed, sweepRoot);
    cfg.params.common.JNR_dB = [30, 30];
    cfg.output_dir = fullfile(sweepRoot, sprintf( ...
        'level_%02d__jnr_%ddB', levelIndex, sweepJnr_dB));

    % 覆盖全功率实验标记，仅保存当前确定性九挡功率剖面。
    cfg.experiment = struct();
    cfg.experiment.name = sprintf( ...
        'closedset_%dclass_power_sweep_9level_v1', ...
        numel(selectedTypes));
    cfg.experiment.power_sweep = struct( ...
        'fixed_codes', {fixedCodes}, ...
        'swept_codes', {sweptCodes}, ...
        'fixed_JNR_dB', 30, ...
        'sweep_JNR_dB', sweepJnr_dB, ...
        'standard_levels_dB', standardLevels_dB, ...
        'level_index', levelIndex, ...
        'paired_seeds', true);
end

function codes = normalize_code_list(value, path, allowEmpty)
%NORMALIZE_CODE_LIST 规范类型代码字符行单元并拒绝重复。

    if isstring(value) && isrow(value) && all(~ismissing(value))
        value = cellstr(value);
    end
    valid = iscell(value) && isrow(value) && ...
        (allowEmpty || ~isempty(value)) && ...
        all(cellfun(@(item) ischar(item) && isrow(item) && ...
        ~isempty(strtrim(item)), value));
    if ~valid
        error('JammingPowerSweep:InvalidProfile', ...
            '%s必须是非空字符行单元。', path);
    end
    codes = cellfun(@strtrim, value, 'UniformOutput', false);
    if numel(unique(codes, 'stable')) ~= numel(codes)
        error('JammingPowerSweep:InvalidProfile', ...
            '%s不允许包含重复类型。', path);
    end
end

function validate_partition(selectedTypes, fixedCodes, sweptCodes)
%VALIDATE_PARTITION 核对固定组和扫描组对启用类型的严格划分。

    overlap = intersect(fixedCodes, sweptCodes, 'stable');
    covered = [fixedCodes, sweptCodes];
    missing = selectedTypes(~ismember(selectedTypes, covered));
    extra = covered(~ismember(covered, selectedTypes));
    valid = isempty(overlap) && isempty(missing) && isempty(extra) && ...
        numel(covered) == numel(selectedTypes);
    if ~valid
        error('JammingPowerSweep:InvalidProfile', ...
            ['fixedCodes和sweptCodes必须无交集，且并集严格等于' ...
            'selectedTypes。']);
    end
end

function level = validate_level(value)
%VALIDATE_LEVEL 规范九挡集合中的一个有限实数标量。

    valid = isnumeric(value) && isreal(value) && isscalar(value) && ...
        isfinite(value) && ismember(double(value), 10:5:50);
    if ~valid
        error('JammingPowerSweep:InvalidLevel', ...
            'sweepJnr_dB必须属于九挡集合10:5:50。');
    end
    level = double(value);
end
