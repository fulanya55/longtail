function bundle = build_legacy_truth_bundle(label, legacyClassCodes, ...
        atomicTruth, sourceSampleIndex, metadata)
%BUILD_LEGACY_TRUTH_BUNDLE 构建历史组合数据的 v1 79 维真值包。
%   BUNDLE = BUILD_LEGACY_TRUTH_BUNDLE(LABEL, LEGACYCLASSCODES,
%   ATOMICTRUTH, SOURCESAMPLEINDEX, METADATA) 保留历史 LABEL 的行列和
%   数值类型，并将已迁移原子真值编码为当前 v1（10 类/79 维）参数模式。
%
%   历史组合脚本按组合块重复使用同一批 1...nBatch 原子波形。因此每个
%   组合输出行必须通过 SOURCESAMPLEINDEX 显式映射回原子真值行；禁止
%   使用组合文件全局行号直接索引 ATOMICTRUTH。
%
%   已迁移类型为 FMB、AMA、MFT 和 RangeFakeTarget。活动但未迁移类型
%   的参数保持 NaN/false，并在 truth_coverage 中说明原因。历史 MFT 和
%   RangeFakeTarget 没有独立 JNR，公共 arrival_delay 也已吸收到逐目标
%   最终时延，因此这些列保持 NaN/false。

    [registry, schema] = load_v1_schema();
    [label, legacyClassCodes, sourceSampleIndex, metadata] = ...
        validate_inputs(label, legacyClassCodes, atomicTruth, ...
        sourceSampleIndex, metadata, registry);
    sampleCount = size(label, 1);
    parameterValues = nan(sampleCount, height(schema));
    parameterMask = false(sampleCount, height(schema));
    supportedCodes = {'FMB', 'AMA', 'MFT', 'RangeFakeTarget'};

    for rowIndex = 1:sampleCount
        activeColumns = find(label(rowIndex, :) ~= 0);
        activeCodes = {};
        componentParams = {};
        for activeIndex = 1:numel(activeColumns)
            code = legacyClassCodes{activeColumns(activeIndex)};
            if ~ismember(code, supportedCodes) || ...
                    ~isfield(atomicTruth, code)
                continue;
            end
            params = legacy_truth_row_to_v1(code, atomicTruth.(code), ...
                sourceSampleIndex(rowIndex), registry);
            activeCodes{end + 1} = code; %#ok<AGROW>
            componentParams{end + 1} = params; %#ok<AGROW>
        end
        if ~isempty(activeCodes)
            [parameterValues(rowIndex, :), parameterMask(rowIndex, :)] = ...
                encode_parameter_targets( ...
                schema, registry, activeCodes, componentParams);
        end
    end

    coverageCodes = legacyClassCodes(:);
    migrated = false(numel(coverageCodes), 1);
    reason = cell(numel(coverageCodes), 1);
    for codeIndex = 1:numel(coverageCodes)
        code = coverageCodes{codeIndex};
        migrated(codeIndex) = ismember(code, supportedCodes) && ...
            isfield(atomicTruth, code);
        if migrated(codeIndex)
            reason{codeIndex} = 'migrated observable truth available';
        else
            reason{codeIndex} = ...
                'legacy generator has no migrated truth output';
        end
    end
    truthCoverage = table(coverageCodes, migrated, reason, ...
        'VariableNames', {'code', 'migrated', 'reason'});

    metadata.scientific_semantics_version = ...
        current_scientific_semantics_version();
    metadata.parameter_registry_version = 'v1';
    metadata.sample_count = sampleCount;
    metadata.truth_width = height(schema);
    metadata.corrected_legacy_physics = true;
    metadata.unmigrated_types = coverageCodes(~migrated).';
    metadata.generated_utc = char(datetime('now', 'TimeZone', 'UTC', ...
        'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));

    bundle = struct();
    bundle.parameter_values = parameterValues;
    bundle.parameter_mask = parameterMask;
    bundle.parameter_schema = schema;
    bundle.label = label;
    bundle.legacy_class_codes = legacyClassCodes;
    bundle.truth_coverage = truthCoverage;
    bundle.atomic_truth = atomicTruth;
    bundle.source_sample_index = sourceSampleIndex;
    bundle.metadata = metadata;
end

function [registry, schema] = load_v1_schema()
%LOAD_V1_SCHEMA 临时加入当前可配置程序路径并显式构建 v1 模式。

    jam9Dir = fileparts(mfilename('fullpath'));
    repoRoot = fileparts(jam9Dir);
    appRoot = fullfile(repoRoot, '可配置组合干扰数据生成');
    originalPath = path;
    cleanup = onCleanup(@() path(originalPath)); %#ok<NASGU>
    addpath(appRoot, fullfile(appRoot, 'core'), ...
        fullfile(appRoot, 'generators'));
    registry = build_jamming_registry(struct());
    schema = build_parameter_schema(registry);
    if numel(registry) ~= 10 || height(schema) ~= 79
        error('LegacyJamming:InvalidLegacyLabelMap', ...
            '历史旁车必须使用 v1 十类/79 维注册表。');
    end
end

function [label, codes, sourceIndex, metadata] = validate_inputs( ...
        label, codes, atomicTruth, sourceIndex, metadata, registry)
%VALIDATE_INPUTS 校验标签、类别映射、原子行索引和元数据。

    validLabelValues = islogical(label) || ...
        (isnumeric(label) && isreal(label) && ...
        all(ismember(label(:), [0, 1])));
    if ~ismatrix(label) || isempty(label) || ~validLabelValues
        error('LegacyJamming:InvalidTruthShape', ...
            'label 必须是非空 logical 或 0/1 二维矩阵。');
    end
    if isstring(codes) && ~any(ismissing(codes), 'all')
        codes = cellstr(codes(:).');
    elseif iscellstr(codes)
        codes = reshape(codes, 1, []);
    else
        error('LegacyJamming:InvalidLegacyLabelMap', ...
            'legacyClassCodes 必须是非缺失 string 数组或 cellstr。');
    end
    codes = cellfun(@strtrim, codes, 'UniformOutput', false);
    registryCodes = {registry.code};
    if size(label, 2) ~= numel(codes) || ...
            any(cellfun(@isempty, codes)) || ...
            numel(unique(codes)) ~= numel(codes) || ...
            ~all(ismember(codes, registryCodes))
        error('LegacyJamming:InvalidLegacyLabelMap', ...
            '标签列数、类别代码唯一性或 v1 类别映射不合法。');
    end
    if ~isstruct(atomicTruth) || ~isscalar(atomicTruth)
        error('LegacyJamming:InvalidTruthShape', ...
            'atomicTruth 必须是标量结构体。');
    end
    unsupportedAtomic = setdiff(fieldnames(atomicTruth), ...
        {'FMB'; 'AMA'; 'MFT'; 'RangeFakeTarget'});
    if ~isempty(unsupportedAtomic)
        error('LegacyJamming:InvalidLegacyLabelMap', ...
            'atomicTruth 含尚未定义旁车映射的类型 %s。', ...
            unsupportedAtomic{1});
    end

    sourceIndex = sourceIndex(:);
    if ~isnumeric(sourceIndex) || ~isreal(sourceIndex) || ...
            numel(sourceIndex) ~= size(label, 1) || ...
            any(~isfinite(sourceIndex)) || any(sourceIndex < 1) || ...
            any(sourceIndex ~= fix(sourceIndex))
        error('LegacyJamming:InvalidTruthShape', ...
            'sourceSampleIndex 必须是与 label 等行的有限正整数列。');
    end
    sourceIndex = double(sourceIndex);
    if ~isstruct(metadata) || ~isscalar(metadata) || ...
            ~all(isfield(metadata, ...
            {'legacy_entrypoint', 'legacy_main_file', 'nRepetition'}))
        error('LegacyJamming:InvalidTruthShape', ...
            'metadata 缺少历史入口、主文件或重复次数。');
    end
    if ~isnumeric(metadata.nRepetition) || ...
            ~isscalar(metadata.nRepetition) || ...
            metadata.nRepetition ~= 1
        error('LegacyJamming:TruthRequiresSingleRepetition', ...
            '历史组合真值要求 nRepetition 等于 1。');
    end
end

function params = legacy_truth_row_to_v1(code, truth, rowIndex, registry)
%LEGACY_TRUTH_ROW_TO_V1 将一个原子真值行映射为精确 v1 参数字段集合。

    registryIndex = find(strcmp({registry.code}, code));
    if numel(registryIndex) ~= 1
        error('LegacyJamming:InvalidLegacyLabelMap', ...
            '原子真值代码在 v1 注册表中不是唯一项。');
    end
    parameterNames = registry(registryIndex).parameter_names;
    params = struct();
    for parameterIndex = 1:numel(parameterNames)
        params.(parameterNames{parameterIndex}) = NaN;
    end

    switch code
        case {'FMB', 'AMA'}
            for parameterIndex = 1:numel(parameterNames)
                name = parameterNames{parameterIndex};
                params.(name) = require_finite_truth_scalar( ...
                    truth, name, rowIndex, code);
            end
        case 'MFT'
            scalarNames = {'num_targets', 'mean_delay_s', 'mean_spacing_s'};
            for nameIndex = 1:numel(scalarNames)
                name = scalarNames{nameIndex};
                params.(name) = require_truth_scalar( ...
                    truth, name, rowIndex, code);
            end
            [delays, amplitudes] = require_target_matrices( ...
                truth, rowIndex, 8, code, false);
            for slot = 1:8
                params.(sprintf('delay_%02d_s', slot)) = delays(slot);
                params.(sprintf('amplitude_%02d', slot)) = amplitudes(slot);
            end
        case 'RangeFakeTarget'
            scalarNames = {'num_targets', 'mean_delay_s'};
            for nameIndex = 1:numel(scalarNames)
                name = scalarNames{nameIndex};
                params.(name) = require_truth_scalar( ...
                    truth, name, rowIndex, code);
            end
            [delays, amplitudes, ranges] = require_target_matrices( ...
                truth, rowIndex, 4, code, true);
            for slot = 1:4
                params.(sprintf('delay_%02d_s', slot)) = delays(slot);
                params.(sprintf('range_%02d_m', slot)) = ranges(slot);
                params.(sprintf('amplitude_%02d', slot)) = amplitudes(slot);
            end
        otherwise
            error('LegacyJamming:InvalidLegacyLabelMap', ...
                '类型 %s 尚未定义历史旁车映射。', code);
    end
end

function value = require_finite_truth_scalar(truth, name, rowIndex, code)
%REQUIRE_FINITE_TRUTH_SCALAR 读取必须为有限值的逐样本标量真值。

    value = require_truth_scalar(truth, name, rowIndex, code);
    if ~isfinite(value)
        error('LegacyJamming:InvalidTruthShape', ...
            '%s.%s 必须是有限逐样本真值。', code, name);
    end
end

function value = require_truth_scalar(truth, name, rowIndex, code)
%REQUIRE_TRUTH_SCALAR 读取允许 NaN 但禁止 Inf 的逐样本标量真值。

    if ~isstruct(truth) || ~isscalar(truth) || ~isfield(truth, name)
        error('LegacyJamming:InvalidTruthShape', ...
            '%s 原子真值缺少字段 %s。', code, name);
    end
    column = truth.(name);
    if ~isnumeric(column) || ~isreal(column) || ...
            size(column, 2) ~= 1 || rowIndex > size(column, 1)
        error('LegacyJamming:InvalidTruthShape', ...
            '%s.%s 不是足够长的实数列。', code, name);
    end
    value = double(column(rowIndex));
    if ~isfinite(value) && ~isnan(value)
        error('LegacyJamming:InvalidTruthShape', ...
            '%s.%s 不能包含 Inf。', code, name);
    end
end

function [delays, amplitudes, ranges] = require_target_matrices( ...
        truth, rowIndex, capacity, code, requireRanges)
%REQUIRE_TARGET_MATRICES 读取固定容量目标时延、幅度和可选距离行。

    required = {'delay_s', 'amplitude'};
    if requireRanges
        required{end + 1} = 'range_m';
    end
    for fieldIndex = 1:numel(required)
        name = required{fieldIndex};
        if ~isfield(truth, name)
            error('LegacyJamming:InvalidTruthShape', ...
                '%s.%s 固定容量矩阵形状或数值非法。', code, name);
        end
        matrix = truth.(name);
        if ~isnumeric(matrix) || ~isreal(matrix) || ...
                size(matrix, 1) < rowIndex || ...
                size(matrix, 2) < capacity || ...
                any(isinf(matrix(rowIndex, 1:capacity)))
            error('LegacyJamming:InvalidTruthShape', ...
                '%s.%s 固定容量矩阵形状或数值非法。', code, name);
        end
    end
    delays = double(truth.delay_s(rowIndex, 1:capacity));
    amplitudes = double(truth.amplitude(rowIndex, 1:capacity));
    ranges = nan(1, capacity);
    if requireRanges
        ranges = double(truth.range_m(rowIndex, 1:capacity));
    end
end
