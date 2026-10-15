function verification = verify_saved_closedset_split( ...
        outputPath, expectedManifest, cfg, schema)
%VERIFY_SAVED_CLOSEDSET_SPLIT 增强回载核对闭集划分 MAT 文件。
%   VERIFICATION = VERIFY_SAVED_CLOSEDSET_SPLIT(OUTPUTPATH,
%   EXPECTEDMANIFEST, CFG, SCHEMA) 在保存或复用文件前后执行完整一致性
%   核对。大型数组只通过 WHOS -FILE 检查变量存在性与尺寸，不载入
%   I/Q 数据；MATFILE 仅回载 manifest、config、parameter_schema 和
%   metadata 等小型变量。带功率验证剖面的文件还会回载分量级cell元数据
%   和参数矩阵，但绝不回载数GB的I/Q或legacy波形。九挡使用确定性
%   profile精确核对；全功率数据使用独立JNR范围和内部功率一致性核对。
%
%   核对项目包括：
%     iq_complex、可选 data_legacy、label_all、label_selected、
%     known_label、parameter_values 和 parameter_mask 的精确尺寸；
%     保存 manifest 与 EXPECTEDMANIFEST 的 ISEQUALN 全等关系；
%     顶层 config、parameter_schema 以及 metadata 内副本与期望值全等。
%   manifest 全等会同时覆盖 sample_seed、组合顺序、活动类型、功率偏置
%   和 split 等字段，不能只凭样本总数误复用旧实验文件。
%
%   任一文件缺失、损坏、维度错误或语义不匹配，均抛出稳定错误标识：
%       JammingDataset:SavedFileVerificationFailed

    stableError = 'JammingDataset:SavedFileVerificationFailed';
    try
        outputPath = normalize_text(outputPath, 'outputPath');
        validate_expected_inputs(expectedManifest, cfg, schema);
        expectedSampleCount = height(expectedManifest);
        expectedSampleLength = double(cfg.derived.sample_length);
        expectedParameterCount = height(schema);
        expectedLabelAllCount = numel(unique( ...
            schema.owner_code, 'stable'));
        expectedSelectedCount = numel(cfg.types.enabled);
        expectedKnownCount = numel(cfg.types.known);

        if ~isfile(outputPath)
            fail_verification('待核对的 MAT 文件不存在：%s。', outputPath);
        end

        % WHOS -FILE 只读取变量目录，避免因回载核对再次分配数 GB 波形。
        variables = whos('-file', outputPath);
        verify_variable_size(variables, 'iq_complex', ...
            [expectedSampleCount, expectedSampleLength]);
        if cfg.output.save_legacy
            verify_variable_size(variables, 'data_legacy', ...
                [expectedSampleCount, 2 * expectedSampleLength]);
        end
        verify_variable_size(variables, 'label_all', ...
            [expectedSampleCount, expectedLabelAllCount]);
        verify_variable_size(variables, 'label_selected', ...
            [expectedSampleCount, expectedSelectedCount]);
        verify_variable_size(variables, 'known_label', ...
            [expectedSampleCount, expectedKnownCount]);
        verify_variable_size(variables, 'parameter_values', ...
            [expectedSampleCount, expectedParameterCount]);
        verify_variable_size(variables, 'parameter_mask', ...
            [expectedSampleCount, expectedParameterCount]);
        require_variable(variables, 'manifest');
        require_variable(variables, 'config');
        require_variable(variables, 'parameter_schema');
        require_variable(variables, 'metadata');
        powerTruthMode = detect_power_truth_mode(cfg);
        powerTruthChecked = ~strcmp(powerTruthMode, 'none');
        if powerTruthChecked
            verify_variable_size(variables, ...
                'component_effective_JNR_dB', [expectedSampleCount, 1]);
            verify_variable_size(variables, ...
                'component_target_power_dB', [expectedSampleCount, 1]);
            verify_variable_size(variables, ...
                'component_measured_power_dB', [expectedSampleCount, 1]);
        end

        % 下列变量远小于波形，可安全回载并执行精确语义比较。
        matObject = matfile(outputPath);
        savedManifest = matObject.manifest;
        savedConfig = matObject.config;
        savedSchema = matObject.parameter_schema;
        savedMetadata = matObject.metadata;
        if ~isequaln(savedManifest, expectedManifest)
            fail_verification(['保存的 manifest 与本次期望 manifest 不一致，' ...
                '可能来自不同种子、组合顺序或配置。']);
        end
        if ~isequaln(savedConfig, cfg)
            fail_verification('保存的顶层 config 与本次配置不一致。');
        end
        if ~isequaln(savedSchema, schema)
            fail_verification('保存的 parameter_schema 与本次模式不一致。');
        end
        validate_metadata(savedMetadata, cfg, schema);
        switch powerTruthMode
            case 'power_sweep'
                verify_power_sweep_truth( ...
                    matObject, expectedManifest, cfg, schema);
            case 'independent_range'
                verify_full_power_truth( ...
                    matObject, expectedManifest, cfg, schema);
        end

        [splitName, patternCount, samplesPerPattern] = ...
            summarize_expected_manifest(expectedManifest, cfg);
        verification = struct( ...
            'performed', true, ...
            'file_path', outputPath, ...
            'split', splitName, ...
            'sample_count', expectedSampleCount, ...
            'sample_length', expectedSampleLength, ...
            'label_all_count', expectedLabelAllCount, ...
            'selected_label_count', expectedSelectedCount, ...
            'known_label_count', expectedKnownCount, ...
            'parameter_count', expectedParameterCount, ...
            'pattern_count', patternCount, ...
            'samples_per_pattern', samplesPerPattern, ...
            'manifest_equal', true, ...
            'config_equal', true, ...
            'metadata_equal', true, ...
            'power_truth_mode', powerTruthMode, ...
            'power_truth_checked', powerTruthChecked, ...
            'power_truth_equal', powerTruthChecked);
    catch exception
        if strcmp(exception.identifier, stableError)
            rethrow(exception);
        end
        error(stableError, '保存文件回载核对失败：%s', exception.message);
    end
end

function mode = detect_power_truth_mode(cfg)
%DETECT_POWER_TRUTH_MODE 识别确定性九挡、连续全功率或普通配置。

    mode = 'none';
    validExperiment = isstruct(cfg) && isscalar(cfg) && ...
        isfield(cfg, 'experiment') && ...
        isstruct(cfg.experiment) && isscalar(cfg.experiment);
    if ~validExperiment
        return;
    end
    if isfield(cfg.experiment, 'power_sweep')
        mode = 'power_sweep';
    elseif isfield(cfg.experiment, 'full_power')
        mode = 'independent_range';
    end
end

function verify_full_power_truth(matObject, manifest, cfg, schema)
%VERIFY_FULL_POWER_TRUTH 核对连续JNR范围及分量级功率内部一致性。
%   全功率JNR由各分量生成器在同一范围内独立抽样，无法像九挡一样从
%   manifest预先重建每个确切值。因此本函数回载最终有效JNR，核对其
%   范围、绝对目标功率、实测功率和参数矩阵中的类型JNR完全自洽。

    profile = cfg.experiment.full_power;
    validProfile = isstruct(profile) && isscalar(profile) && ...
        isfield(profile, 'JNR_range_dB') && ...
        isnumeric(profile.JNR_range_dB) && ...
        isreal(profile.JNR_range_dB) && ...
        isequal(size(profile.JNR_range_dB), [1, 2]) && ...
        all(isfinite(profile.JNR_range_dB)) && ...
        profile.JNR_range_dB(1) <= profile.JNR_range_dB(2);
    if ~validProfile
        fail_verification('全功率配置缺少合法JNR_range_dB。');
    end
    tolerance_dB = require_power_tolerance(cfg);
    noisePower_dB = 10 * log10(double(cfg.radar.noise_power));
    effectiveJnr = matObject.component_effective_JNR_dB;
    targetPower = matObject.component_target_power_dB;
    measuredPower = matObject.component_measured_power_dB;
    parameterValues = matObject.parameter_values;
    parameterMask = matObject.parameter_mask;
    if ~(iscell(effectiveJnr) && iscell(targetPower) && ...
            iscell(measuredPower))
        fail_verification('全功率分量级功率元数据必须为cell列。');
    end

    for sampleIndex = 1:height(manifest)
        codes = manifest.active_type_codes{sampleIndex};
        componentCount = manifest.combination_size(sampleIndex);
        actualJnr_dB = effectiveJnr{sampleIndex};
        if componentCount ~= numel(codes) || ...
                ~numeric_row_in_range( ...
                actualJnr_dB, profile.JNR_range_dB)
            fail_verification('第%d条有效JNR超出全功率范围。', ...
                sampleIndex);
        end
        expectedTarget_dB = noisePower_dB + actualJnr_dB;
        if ~numeric_row_matches(targetPower{sampleIndex}, ...
                expectedTarget_dB, tolerance_dB)
            fail_verification('第%d条目标功率与有效JNR不一致。', ...
                sampleIndex);
        end
        if ~numeric_row_matches(measuredPower{sampleIndex}, ...
                expectedTarget_dB, tolerance_dB)
            fail_verification('第%d条实测功率超出允许误差。', ...
                sampleIndex);
        end

        for componentIndex = 1:componentCount
            code = codes{componentIndex};
            jnrColumn = find(strcmp( ...
                schema.field_name, [code '.JNR_dB']));
            validParameter = numel(jnrColumn) == 1 && ...
                parameterMask(sampleIndex, jnrColumn) && ...
                isfinite(parameterValues(sampleIndex, jnrColumn)) && ...
                abs(parameterValues(sampleIndex, jnrColumn) - ...
                actualJnr_dB(componentIndex)) <= tolerance_dB;
            if ~validParameter
                fail_verification( ...
                    '第%d条类型%s的参数JNR真值不匹配。', ...
                    sampleIndex, code);
            end
        end
    end
end

function verify_power_sweep_truth(matObject, manifest, cfg, schema)
%VERIFY_POWER_SWEEP_TRUTH 逐活动分量核对JNR、功率和参数监督。

    profile = cfg.experiment.power_sweep;
    requiredProfileFields = {'fixed_codes', 'swept_codes', ...
        'fixed_JNR_dB', 'sweep_JNR_dB'};
    if ~isstruct(profile) || ~isscalar(profile) || ...
            ~all(isfield(profile, requiredProfileFields))
        fail_verification('保存配置缺少完整 power_sweep 功率剖面。');
    end
    tolerance_dB = require_power_tolerance(cfg);
    noisePower_dB = 10 * log10(double(cfg.radar.noise_power));

    effectiveJnr = matObject.component_effective_JNR_dB;
    targetPower = matObject.component_target_power_dB;
    measuredPower = matObject.component_measured_power_dB;
    parameterValues = matObject.parameter_values;
    parameterMask = matObject.parameter_mask;
    if ~(iscell(effectiveJnr) && iscell(targetPower) && ...
            iscell(measuredPower))
        fail_verification('功率扫描分量级功率元数据必须为 cell 列。');
    end

    for sampleIndex = 1:height(manifest)
        codes = manifest.active_type_codes{sampleIndex};
        componentCount = manifest.combination_size(sampleIndex);
        expectedJnr_dB = expected_jnr_for_codes(codes, profile);
        expectedOffsets_dB = expectedJnr_dB - profile.fixed_JNR_dB;
        expectedTarget_dB = noisePower_dB + expectedJnr_dB;
        expectedWeak = expectedJnr_dB < max(expectedJnr_dB);

        if componentCount ~= numel(codes) || ...
                ~numeric_row_matches( ...
                manifest.component_target_power_dB{sampleIndex}, ...
                expectedOffsets_dB, tolerance_dB) || ...
                ~isequal(manifest.weak_component_mask{sampleIndex}, ...
                logical(expectedWeak))
            fail_verification('第%d条 manifest 功率剖面与活动类型不一致。', ...
                sampleIndex);
        end
        if ~numeric_row_matches(effectiveJnr{sampleIndex}, ...
                expectedJnr_dB, tolerance_dB)
            fail_verification('第%d条有效JNR真值不匹配。', sampleIndex);
        end
        if ~numeric_row_matches(targetPower{sampleIndex}, ...
                expectedTarget_dB, tolerance_dB)
            fail_verification('第%d条目标功率真值不匹配。', sampleIndex);
        end
        if ~numeric_row_matches(measuredPower{sampleIndex}, ...
                expectedTarget_dB, tolerance_dB)
            fail_verification('第%d条实测功率超出允许误差。', sampleIndex);
        end

        for componentIndex = 1:componentCount
            jnrField = [codes{componentIndex} '.JNR_dB'];
            jnrColumn = find(strcmp(schema.field_name, jnrField));
            if numel(jnrColumn) ~= 1 || ...
                    ~parameterMask(sampleIndex, jnrColumn) || ...
                    ~isfinite(parameterValues(sampleIndex, jnrColumn)) || ...
                    abs(parameterValues(sampleIndex, jnrColumn) - ...
                    expectedJnr_dB(componentIndex)) > tolerance_dB
                fail_verification( ...
                    '第%d条类型%s的参数JNR真值不匹配。', ...
                    sampleIndex, codes{componentIndex});
            end
        end
    end
end

function tolerance_dB = require_power_tolerance(cfg)
%REQUIRE_POWER_TOLERANCE 返回非负有限功率回载容差。

    valid = isfield(cfg, 'output') && ...
        isfield(cfg.output, 'power_tolerance_dB') && ...
        isnumeric(cfg.output.power_tolerance_dB) && ...
        isreal(cfg.output.power_tolerance_dB) && ...
        isscalar(cfg.output.power_tolerance_dB) && ...
        isfinite(cfg.output.power_tolerance_dB) && ...
        cfg.output.power_tolerance_dB >= 0;
    if ~valid
        fail_verification('配置缺少合法power_tolerance_dB。');
    end
    tolerance_dB = double(cfg.output.power_tolerance_dB);
end

function expectedJnr_dB = expected_jnr_for_codes(codes, profile)
%EXPECTED_JNR_FOR_CODES 根据功率剖面映射活动代码的期望有效JNR。

    if ~(iscell(codes) && isrow(codes) && ~isempty(codes))
        fail_verification('manifest.active_type_codes 必须是非空行单元。');
    end
    expectedJnr_dB = nan(1, numel(codes));
    for componentIndex = 1:numel(codes)
        code = codes{componentIndex};
        if ismember(code, profile.fixed_codes)
            expectedJnr_dB(componentIndex) = profile.fixed_JNR_dB;
        elseif ismember(code, profile.swept_codes)
            expectedJnr_dB(componentIndex) = profile.sweep_JNR_dB;
        else
            fail_verification('活动类型%s未被功率扫描配置覆盖。', code);
        end
    end
end

function matches = numeric_row_matches(actual, expected, tolerance)
%NUMERIC_ROW_MATCHES 判断有限实数行是否在给定绝对误差内匹配。

    matches = isnumeric(actual) && isreal(actual) && isrow(actual) && ...
        numel(actual) == numel(expected) && all(isfinite(actual)) && ...
        all(abs(double(actual) - double(expected)) <= tolerance);
end

function matches = numeric_row_in_range(actual, limits)
%NUMERIC_ROW_IN_RANGE 判断有限实数行是否全部位于闭区间内。

    matches = isnumeric(actual) && isreal(actual) && isrow(actual) && ...
        ~isempty(actual) && all(isfinite(actual)) && ...
        all(double(actual) >= double(limits(1))) && ...
        all(double(actual) <= double(limits(2)));
end

function validate_expected_inputs(expectedManifest, cfg, schema)
%VALIDATE_EXPECTED_INPUTS 校验回载核对所需的期望对象。

    if ~istable(expectedManifest) || height(expectedManifest) < 1 || ...
            ~all(ismember({'split', 'combination_id', 'sample_seed'}, ...
            expectedManifest.Properties.VariableNames))
        fail_verification(['expectedManifest 必须是非空 table，且包含 ' ...
            'split、combination_id 和 sample_seed。']);
    end
    validCfg = isstruct(cfg) && isscalar(cfg) && ...
        isfield(cfg, 'derived') && isfield(cfg.derived, 'sample_length') && ...
        isfield(cfg, 'types') && isfield(cfg.types, 'enabled') && ...
        isfield(cfg.types, 'known') && isfield(cfg, 'output') && ...
        isfield(cfg.output, 'save_legacy');
    if ~validCfg
        fail_verification('cfg 必须是含派生尺寸、类型和输出字段的标准化配置。');
    end
    if ~istable(schema) || height(schema) < 1 || ...
            ~ismember('owner_code', schema.Properties.VariableNames)
        fail_verification('schema 必须是含 owner_code 的非空参数模式表。');
    end
end

function info = require_variable(variables, variableName)
%REQUIRE_VARIABLE 在 MAT 变量目录中定位必需变量。

    index = find(strcmp({variables.name}, variableName), 1);
    if isempty(index)
        fail_verification('MAT 文件缺少变量 %s。', variableName);
    end
    info = variables(index);
end

function verify_variable_size(variables, variableName, expectedSize)
%VERIFY_VARIABLE_SIZE 核对变量二维尺寸而不读取变量内容。

    info = require_variable(variables, variableName);
    actualSize = double(info.size);
    if ~isequal(actualSize, double(expectedSize))
        fail_verification('%s 尺寸应为 [%d,%d]，实际为 [%s]。', ...
            variableName, expectedSize(1), expectedSize(2), ...
            strtrim(sprintf('%d ', actualSize)));
    end
end

function validate_metadata(metadata, cfg, schema)
%VALIDATE_METADATA 核对 metadata 内嵌配置和参数模式副本。

    valid = isstruct(metadata) && isscalar(metadata) && ...
        isfield(metadata, 'config') && ...
        isfield(metadata, 'parameter_schema');
    if ~valid
        fail_verification('metadata 缺少 config 或 parameter_schema。');
    end
    if ~isequaln(metadata.config, cfg)
        fail_verification('metadata.config 与本次配置不一致。');
    end
    if ~isequaln(metadata.parameter_schema, schema)
        fail_verification('metadata.parameter_schema 与本次模式不一致。');
    end
end

function [splitName, patternCount, samplesPerPattern] = ...
        summarize_expected_manifest(manifest, cfg)
%SUMMARIZE_EXPECTED_MANIFEST 复核单划分和等量组合分布并返回摘要。

    splitValues = normalize_text_column(manifest.split, 'manifest.split');
    uniqueSplits = unique(splitValues, 'stable');
    if numel(uniqueSplits) ~= 1
        fail_verification('expectedManifest 必须只包含一个 split。');
    end
    splitName = uniqueSplits{1};
    combinationIds = normalize_text_column( ...
        manifest.combination_id, 'manifest.combination_id');
    [uniquePatterns, ~, groupIndex] = unique(combinationIds, 'sorted');
    patternCounts = accumarray(groupIndex, 1);
    uniqueCounts = unique(patternCounts);
    variableCounts = is_pure_label_longtail_config(cfg);
    if numel(uniqueCounts) ~= 1 && ~variableCounts
        fail_verification('expectedManifest 的各组合模式样本数必须相同。');
    end
    patternCount = numel(uniquePatterns);
    if variableCounts
        samplesPerPattern = [];
    else
        samplesPerPattern = uniqueCounts(1);
    end
end

function present = is_pure_label_longtail_config(cfg)
    present = isstruct(cfg) && isfield(cfg, 'experiment') && ...
        isstruct(cfg.experiment) && ...
        isfield(cfg.experiment, 'pure_label_longtail');
end

function values = normalize_text_column(value, path)
%NORMALIZE_TEXT_COLUMN 将 table 文本列统一为非空字符单元列。

    if iscell(value) && all(cellfun( ...
            @(item) ischar(item) && isrow(item) && ~isempty(item), value))
        values = reshape(value, [], 1);
    elseif isstring(value) && all(~ismissing(value))
        values = cellstr(reshape(value, [], 1));
    elseif iscategorical(value) && all(~isundefined(value))
        values = cellstr(reshape(value, [], 1));
    else
        fail_verification('%s 必须是无缺失的非空文本列。', path);
    end
end

function value = normalize_text(value, path)
%NORMALIZE_TEXT 将非空字符向量或 string 标量规范为字符行向量。

    if isstring(value) && isscalar(value) && ~ismissing(value)
        value = char(value);
    elseif ~(ischar(value) && isrow(value))
        fail_verification('%s 必须是非空文本标量。', path);
    end
    value = strtrim(value);
    if isempty(value)
        fail_verification('%s 不能为空。', path);
    end
end

function fail_verification(message, varargin)
%FAIL_VERIFICATION 使用统一错误标识终止回载核对。

    error('JammingDataset:SavedFileVerificationFailed', ...
        message, varargin{:});
end
