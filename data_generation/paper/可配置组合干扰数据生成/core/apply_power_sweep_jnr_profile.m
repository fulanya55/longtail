function manifest = apply_power_sweep_jnr_profile(manifest, cfg)
%APPLY_POWER_SWEEP_JNR_PROFILE 按活动类型写入确定性功率扫描偏置。
%   MANIFEST = APPLY_POWER_SWEEP_JNR_PROFILE(MANIFEST, CFG) 读取
%   CFG.EXPERIMENT.POWER_SWEEP，把普通闭集 manifest 中每个活动分量的
%   component_target_power_dB 从零偏置改为“期望有效JNR-公共标称JNR”。
%
%   本函数不改变样本顺序、组合、sample_seed、抽样权重或 imbalance_flags，
%   因而九个挡位可以构成严格配对实验。weak_component_mask 仅描述当前
%   样本内低于最高有效JNR的活动分量；单分量或同功率组合全部为 false。
%
%   输入必须来自 BUILD_BALANCED_MANIFEST，且六个启用代码必须被 fixed_codes
%   或 swept_codes 恰好覆盖。任何配置或表结构错误均在生成波形前拒绝。

    profile = validate_profile(cfg);
    validate_manifest(manifest);
    nominalJnr_dB = double(cfg.params.common.JNR_dB(1));

    for sampleIndex = 1:height(manifest)
        codes = manifest.active_type_codes{sampleIndex};
        if ~(iscell(codes) && isrow(codes) && ~isempty(codes) && ...
                all(cellfun(@is_valid_code, codes)))
            throw_invalid_manifest();
        end

        expectedJnr_dB = nan(1, numel(codes));
        for componentIndex = 1:numel(codes)
            code = codes{componentIndex};
            if ismember(code, profile.fixed_codes)
                expectedJnr_dB(componentIndex) = profile.fixed_JNR_dB;
            elseif ismember(code, profile.swept_codes)
                expectedJnr_dB(componentIndex) = profile.sweep_JNR_dB;
            else
                error('JammingPowerSweep:InvalidProfile', ...
                    '活动类型 %s 未被功率扫描 fixed_codes/swept_codes 覆盖。', ...
                    code);
            end
        end

        offsets_dB = expectedJnr_dB - nominalJnr_dB;
        weakMask = expectedJnr_dB < max(expectedJnr_dB);
        manifest.component_target_power_dB{sampleIndex} = offsets_dB;
        manifest.weak_component_mask{sampleIndex} = logical(weakMask);
    end
end

function profile = validate_profile(cfg)
%VALIDATE_PROFILE 校验保存于配置中的九挡功率剖面。

    if ~isstruct(cfg) || ~isscalar(cfg) || ...
            ~isfield(cfg, 'experiment') || ...
            ~isstruct(cfg.experiment) || ~isscalar(cfg.experiment) || ...
            ~isfield(cfg.experiment, 'power_sweep')
        throw_invalid_profile();
    end
    profile = cfg.experiment.power_sweep;
    required = {'fixed_codes', 'swept_codes', 'fixed_JNR_dB', ...
        'sweep_JNR_dB', 'standard_levels_dB', 'level_index', ...
        'paired_seeds'};
    if ~isstruct(profile) || ~isscalar(profile) || ...
            ~all(isfield(profile, required)) || ...
            ~isfield(cfg, 'params') || ~isfield(cfg.params, 'common') || ...
            ~isfield(cfg.params.common, 'JNR_dB')
        throw_invalid_profile();
    end

    fixedCodes = normalize_code_list(profile.fixed_codes);
    sweptCodes = normalize_code_list(profile.swept_codes);
    if ~isempty(intersect(fixedCodes, sweptCodes)) || ...
            numel(unique([fixedCodes, sweptCodes], 'stable')) ~= ...
            numel([fixedCodes, sweptCodes])
        throw_invalid_profile();
    end
    profile.fixed_codes = fixedCodes;
    profile.swept_codes = sweptCodes;

    numericFields = {'fixed_JNR_dB', 'sweep_JNR_dB', 'level_index'};
    for fieldIndex = 1:numel(numericFields)
        value = profile.(numericFields{fieldIndex});
        if ~(isnumeric(value) && isreal(value) && isscalar(value) && ...
                isfinite(value))
            throw_invalid_profile();
        end
        profile.(numericFields{fieldIndex}) = double(value);
    end
    levels = profile.standard_levels_dB;
    validLevels = isnumeric(levels) && isreal(levels) && isrow(levels) && ...
        isequal(double(levels), 10:5:50);
    validIndex = profile.level_index == round(profile.level_index) && ...
        profile.level_index >= 1 && profile.level_index <= numel(levels) && ...
        levels(profile.level_index) == profile.sweep_JNR_dB;
    validPaired = (islogical(profile.paired_seeds) || ...
        isnumeric(profile.paired_seeds)) && isscalar(profile.paired_seeds) && ...
        isfinite(double(profile.paired_seeds)) && ...
        ismember(double(profile.paired_seeds), [0, 1]);
    commonJnr = cfg.params.common.JNR_dB;
    validCommon = isnumeric(commonJnr) && isreal(commonJnr) && ...
        isequal(double(commonJnr), ...
        [profile.fixed_JNR_dB, profile.fixed_JNR_dB]);
    if ~(validLevels && validIndex && validPaired && ...
            logical(profile.paired_seeds) && validCommon)
        throw_invalid_profile();
    end

    if ~isfield(cfg, 'types') || ~isfield(cfg.types, 'enabled') || ...
            ~isequal(sort([fixedCodes, sweptCodes]), ...
            sort(cfg.types.enabled))
        throw_invalid_profile();
    end
end

function codes = normalize_code_list(value)
%NORMALIZE_CODE_LIST 规范非空、无重复的字符代码行单元。

    if ~(iscell(value) && isrow(value) && ~isempty(value) && ...
            all(cellfun(@is_valid_code, value)))
        throw_invalid_profile();
    end
    codes = value;
    if numel(unique(codes, 'stable')) ~= numel(codes)
        throw_invalid_profile();
    end
end

function validate_manifest(manifest)
%VALIDATE_MANIFEST 校验功率扫描要修改的 manifest 列及逐行长度。

    required = {'active_type_codes', 'combination_size', ...
        'component_target_power_dB', 'weak_component_mask', ...
        'sample_seed', 'imbalance_flags'};
    if ~istable(manifest) || isempty(manifest) || ...
            ~all(ismember(required, manifest.Properties.VariableNames))
        throw_invalid_manifest();
    end
    for sampleIndex = 1:height(manifest)
        codes = manifest.active_type_codes{sampleIndex};
        if numel(codes) ~= manifest.combination_size(sampleIndex)
            throw_invalid_manifest();
        end
    end
end

function valid = is_valid_code(value)
%IS_VALID_CODE 判断类型代码是否为非空字符行向量。

    valid = ischar(value) && isrow(value) && ~isempty(strtrim(value));
end

function throw_invalid_profile()
%THROW_INVALID_PROFILE 抛出稳定的功率扫描配置错误。

    error('JammingPowerSweep:InvalidProfile', ...
        'cfg.experiment.power_sweep 不是合法的六类九挡配对功率配置。');
end

function throw_invalid_manifest()
%THROW_INVALID_MANIFEST 抛出稳定的功率扫描 manifest 错误。

    error('JammingPowerSweep:InvalidManifest', ...
        '功率扫描 manifest 缺少必需列或逐行活动分量结构不一致。');
end
