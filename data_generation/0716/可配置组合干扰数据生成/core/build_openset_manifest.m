function splits = build_openset_manifest(cfg, registry)
%BUILD_OPENSET_MANIFEST 构建开放集识别协议的四个确定性样本清单。
%   SPLITS = BUILD_OPENSET_MANIFEST(CFG, REGISTRY) 接收已由
%   NORMALIZE_AND_VALIDATE_CONFIG(CFG, 'openset') 标准化的配置和固定干扰
%   注册表，按顺序返回 train_known、calibration_known、
%   validation_mixed、test_mixed 四个字段。每个字段均为与
%   BUILD_BALANCED_MANIFEST 完全相同的 14 列 table。
%
%   训练和校准划分只包含 known 类型；混合验证和测试划分按
%   cfg.openset.mixed_ratios 的固定顺序生成四类样本：纯已知、纯未知、
%   已知加一个未知、已知加多个未知。比例使用最大余数法转换为精确行数，
%   同余数时保持上述类别顺序，然后用独立 mt19937ar 流打乱类别位置及
%   组合内容。活动类型始终按注册表 ID 排序，且不读取或修改全局随机状态。
%
%   active_type_codes、unknown_present 和 unknown_component_count 保存用于
%   评测的完整真值；开放集训练流程应只将 train_known 中的 known 标签视为
%   可见监督信息，不应把验证/测试中的未知类型名称泄露给训练算法。每个
%   非零划分继续继承普通构建器的百万行、uint32 种子和输出容量检查；零行
%   划分返回列名称及列类型完整的空表。
%
%   示例：
%       cfg = create_default_config();
%       cfg.types.enabled = {'FMB', 'MFT', 'ISRJ', 'SinFM', 'VFT'};
%       cfg.types.known = {'FMB', 'MFT', 'ISRJ'};
%       cfg.types.unknown = {'SinFM', 'VFT'};
%       cfg = normalize_and_validate_config(cfg, 'openset');
%       registry = build_jamming_registry(cfg);
%       splits = build_openset_manifest(cfg, registry);

    [knownIds, knownCodes, unknownIds, unknownCodes] = ...
        validate_and_resolve_partition(cfg.types, registry);
    splitCounts = validate_split_counts(cfg.openset);
    validationCategoryCounts = allocate_largest_remainder( ...
        splitCounts(3), cfg.openset.mixed_ratios);
    testCategoryCounts = allocate_largest_remainder( ...
        splitCounts(4), cfg.openset.mixed_ratios);
    requiredCategoryCounts = validationCategoryCounts + ...
        testCategoryCounts;
    validate_compositions(cfg, splitCounts, requiredCategoryCounts, ...
        numel(knownCodes), numel(unknownCodes));

    splits = struct();
    splits.train_known = build_known_split(cfg, registry, ...
        knownCodes, splitCounts(1), 'train');
    splits.calibration_known = build_known_split(cfg, registry, ...
        knownCodes, splitCounts(2), 'calibration');
    splits.validation_mixed = build_mixed_split(cfg, ...
        knownIds, knownCodes, unknownIds, unknownCodes, ...
        splitCounts(3), validationCategoryCounts, 'validation');
    splits.test_mixed = build_mixed_split(cfg, ...
        knownIds, knownCodes, unknownIds, unknownCodes, ...
        splitCounts(4), testCategoryCounts, 'test');
end

function [knownIds, knownCodes, unknownIds, unknownCodes] = ...
        validate_and_resolve_partition(types, registry)
%VALIDATE_AND_RESOLVE_PARTITION 防御性校验分区并按注册表 ID 排序。

    try
        enabled = normalize_code_list(types.enabled);
        known = normalize_code_list(types.known);
        unknown = normalize_code_list(types.unknown);
        registryCodes = {registry.code};
    catch
        throw_invalid_partition();
    end

    if ~isstruct(registry) || ~isfield(registry, 'id')
        throw_invalid_partition();
    end
    registryIds = zeros(1, numel(registry));
    for registryIndex = 1:numel(registry)
        registryId = registry(registryIndex).id;
        validRegistryId = isnumeric(registryId) && ...
            isscalar(registryId) && isreal(registryId) && ...
            isfinite(registryId) && registryId == fix(registryId) && ...
            registryId > 0;
        if ~validRegistryId
            throw_invalid_partition();
        end
        registryIds(registryIndex) = double(registryId);
    end

    allConfigured = [known, unknown];
    invalidDuplicates = numel(unique(enabled, 'stable')) ~= ...
        numel(enabled) || numel(unique(known, 'stable')) ~= ...
        numel(known) || numel(unique(unknown, 'stable')) ~= numel(unknown);
    invalidOverlap = ~isempty(intersect(known, unknown, 'stable'));
    invalidCoverage = ~isempty(setxor(enabled, allConfigured, 'stable'));
    invalidRegistry = numel(unique(registryCodes, 'stable')) ~= ...
        numel(registryCodes) || numel(unique(registryIds)) ~= ...
        numel(registryIds) || any(~ismember(enabled, registryCodes));
    if invalidDuplicates || invalidOverlap || invalidCoverage || ...
            invalidRegistry
        throw_invalid_partition();
    end

    [knownFound, knownLocations] = ismember(known, registryCodes);
    [unknownFound, unknownLocations] = ismember(unknown, registryCodes);
    if ~all(knownFound) || ~all(unknownFound)
        throw_invalid_partition();
    end
    knownIds = registryIds(knownLocations);
    unknownIds = registryIds(unknownLocations);
    [knownIds, knownOrder] = sort(knownIds);
    [unknownIds, unknownOrder] = sort(unknownIds);
    knownCodes = known(knownOrder);
    unknownCodes = unknown(unknownOrder);
end

function codes = normalize_code_list(codes)
%NORMALIZE_CODE_LIST 将已标准化类型列表再次收敛为行 cellstr。

    if isstring(codes) && all(~ismissing(codes), 'all')
        codes = cellstr(codes(:).');
    elseif iscellstr(codes)
        codes = reshape(codes, 1, []);
    elseif iscell(codes) && isempty(codes)
        codes = cell(1, 0);
    else
        throw_invalid_partition();
    end
    if any(cellfun(@isempty, codes))
        throw_invalid_partition();
    end
end

function throw_invalid_partition()
%THROW_INVALID_PARTITION 使用统一错误标识报告分区或映射问题。

    error('JammingOpenSet:InvalidPartition', ...
        ['cfg.types.enabled、known 和 unknown 必须形成互斥且完整的注册表' ...
        '类型分区。']);
end

function splitCounts = validate_split_counts(openset)
%VALIDATE_SPLIT_COUNTS 校验四个实际划分行数并允许稳健的零行输入。

    fieldNames = {'train_known', 'calibration_known', ...
        'validation_mixed', 'test_mixed'};
    splitCounts = zeros(1, numel(fieldNames));
    for fieldIndex = 1:numel(fieldNames)
        value = openset.(fieldNames{fieldIndex});
        valid = isnumeric(value) && isscalar(value) && isreal(value) && ...
            isfinite(value) && value == fix(value) && value >= 0 && ...
            value <= 1000000;
        if ~valid
            error('JammingManifest:SplitTooLarge', ...
                '每个开放集 manifest 的实际行数必须位于 0 到 1000000。');
        end
        splitCounts(fieldIndex) = double(value);
    end
end

function validate_compositions(cfg, splitCounts, requiredCategoryCounts, ...
        knownCount, unknownCount)
%VALIDATE_COMPOSITIONS 只对两个 mixed 划分实际分到行数的类别做预检。

    minimumComponents = double(cfg.dataset.min_components);
    maximumComponents = double(cfg.dataset.max_components);
    knownOnlyLower = max(1, minimumComponents);
    knownOnlyUpper = min(maximumComponents, knownCount);
    if any(splitCounts(1:2) > 0) && knownOnlyLower > knownOnlyUpper
        error('JammingOpenSet:InvalidComposition', ...
            'known-only 划分无法满足当前已知类型数和分量范围。');
    end

    physicalLowerBounds = [1, 1, 2, 3];
    lowerBounds = max(physicalLowerBounds, minimumComponents);
    upperBounds = [min(maximumComponents, knownCount), ...
        min(maximumComponents, unknownCount), ...
        min(maximumComponents, knownCount + 1), ...
        min(maximumComponents, knownCount + unknownCount)];
    poolsAvailable = [knownCount >= 1, unknownCount >= 1, ...
        knownCount >= 1 && unknownCount >= 1, ...
        knownCount >= 1 && unknownCount >= 2];
    categoryNames = {'纯已知', '纯未知', '已知加一个未知', ...
        '已知加多个未知'};

    for category = 1:4
        if requiredCategoryCounts(category) > 0 && ...
                (~poolsAvailable(category) || ...
                lowerBounds(category) > upperBounds(category))
            error('JammingOpenSet:InvalidComposition', ...
                ['%s mixed 类别无法满足当前分区、dataset.min_components ' ...
                '和 dataset.max_components。'], categoryNames{category});
        end
    end
end

function manifest = build_known_split(cfg, registry, knownCodes, ...
        sampleCount, splitName)
%BUILD_KNOWN_SPLIT 通过普通构建器生成仅含已知类型的协议划分。

    if sampleCount == 0
        manifest = create_empty_manifest();
        return;
    end
    temporaryCfg = cfg;
    temporaryCfg.types.enabled = knownCodes;
    temporaryCfg.types.known = knownCodes;
    temporaryCfg.types.unknown = cell(1, 0);
    temporaryCfg.dataset.combination_mode = 'random';
    temporaryCfg.dataset.num_samples = sampleCount;
    temporaryCfg.dataset.max_components = min( ...
        cfg.dataset.max_components, numel(knownCodes));
    if temporaryCfg.dataset.min_components > ...
            temporaryCfg.dataset.max_components
        error('JammingOpenSet:InvalidComposition', ...
            'known-only 划分的最小分量数超过可用已知类型数。');
    end
    manifest = build_balanced_manifest(temporaryCfg, registry, splitName);
end

function manifest = build_mixed_split(cfg, knownIds, knownCodes, ...
        unknownIds, unknownCodes, sampleCount, categoryCounts, splitName)
%BUILD_MIXED_SPLIT 从本地安全骨架直接填写四类开放集活动模式。

    [manifest, numericSeed, splitOffset] = ...
        create_safe_mixed_skeleton(cfg, sampleCount, splitName);
    if sampleCount == 0
        return;
    end

    streamSeed = mod(numericSeed + splitOffset + 2654435761, 2^32);
    stream = RandStream('mt19937ar', 'Seed', streamSeed);
    categories = repelem(1:4, categoryCounts).';
    if sampleCount > 1
        categories = categories(randperm(stream, sampleCount));
    end

    minimumComponents = double(cfg.dataset.min_components);
    maximumComponents = double(cfg.dataset.max_components);
    for sampleIndex = 1:sampleCount
        [ids, codes] = draw_open_set_pattern(stream, ...
            categories(sampleIndex), knownIds, knownCodes, ...
            unknownIds, unknownCodes, minimumComponents, ...
            maximumComponents);
        unknownCount = sum(ismember(codes, unknownCodes));
        componentCount = numel(ids);
        manifest.active_type_ids{sampleIndex} = ids;
        manifest.active_type_codes{sampleIndex} = codes;
        manifest.combination_id{sampleIndex} = strjoin(codes, '+');
        manifest.combination_size(sampleIndex) = componentCount;
        manifest.component_target_power_dB{sampleIndex} = ...
            zeros(1, componentCount);
        manifest.weak_component_mask{sampleIndex} = ...
            false(1, componentCount);
        manifest.component_sampling_weight(sampleIndex) = 1;
        manifest.pattern_sampling_weight(sampleIndex) = 1;
        manifest.imbalance_flags{sampleIndex} = ...
            logical([false, false, false]);
        manifest.unknown_present(sampleIndex) = unknownCount > 0;
        manifest.unknown_component_count(sampleIndex) = unknownCount;
    end
end

function [manifest, numericSeed, splitOffset] = ...
        create_safe_mixed_skeleton(cfg, sampleCount, splitName)
%CREATE_SAFE_MIXED_SKELETON 校验安全边界后直接预分配 14 列 table。

    validSampleCount = isnumeric(sampleCount) && isscalar(sampleCount) && ...
        isreal(sampleCount) && isfinite(sampleCount) && ...
        sampleCount == fix(sampleCount) && sampleCount >= 0 && ...
        sampleCount <= 1000000;
    if ~validSampleCount
        error('JammingManifest:SplitTooLarge', ...
            '每个开放集 manifest 的实际行数必须位于 0 到 1000000。');
    end
    sampleCount = double(sampleCount);

    switch splitName
        case 'validation'
            splitOffset = 2000000;
        case 'test'
            splitOffset = 3000000;
        otherwise
            error('JammingManifest:UnknownSplit', ...
                'mixed 划分仅允许 validation 或 test。');
    end

    maximumSeed = double(intmax('uint32'));
    validSeed = isnumeric(cfg.seed) && isscalar(cfg.seed) && ...
        isreal(cfg.seed) && isfinite(cfg.seed);
    if ~validSeed
        error('JammingManifest:InvalidSeed', ...
            'cfg.seed 必须是 uint32 取值域内的有限非负整数数值标量。');
    end
    numericSeed = double(cfg.seed);
    if numericSeed ~= fix(numericSeed) || numericSeed < 0 || ...
            numericSeed > maximumSeed
        error('JammingManifest:InvalidSeed', ...
            'cfg.seed 必须是 uint32 取值域内的有限非负整数数值标量。');
    end
    if numericSeed > maximumSeed - splitOffset - sampleCount
        error('JammingManifest:SeedOverflow', ...
            ['cfg.seed、划分偏移与实际 sample_id 的和不能超过 ' ...
            '2^32-1。']);
    end

    bytesPerPoint = 16 + 8 * double(cfg.output.save_legacy);
    estimatedBytes = sampleCount * double(cfg.derived.sample_length) * ...
        bytesPerPoint;
    if ~isfinite(estimatedBytes) || ...
            estimatedBytes > cfg.output.max_estimated_bytes
        error('JammingConfig:EstimatedSizeExceeded', ...
            ['按 manifest 实际 %d 行估算的输出大小为 %.0f 字节，超过 ' ...
            'cfg.output.max_estimated_bytes=%.0f 字节。'], ...
            sampleCount, estimatedBytes, cfg.output.max_estimated_bytes);
    end

    sampleId = (1:sampleCount)';
    splitColumn = repmat({splitName}, sampleCount, 1);
    sampleSeed = numericSeed + splitOffset + sampleId;
    activeTypeIds = cell(sampleCount, 1);
    activeTypeCodes = cell(sampleCount, 1);
    combinationId = cell(sampleCount, 1);
    combinationSize = zeros(sampleCount, 1);
    componentTargetPowerDb = cell(sampleCount, 1);
    weakComponentMask = cell(sampleCount, 1);
    componentSamplingWeight = ones(sampleCount, 1);
    patternSamplingWeight = ones(sampleCount, 1);
    imbalanceFlags = repmat({logical([false, false, false])}, ...
        sampleCount, 1);
    unknownPresent = false(sampleCount, 1);
    unknownComponentCount = zeros(sampleCount, 1);

    manifest = table(sampleId, splitColumn, sampleSeed, activeTypeIds, ...
        activeTypeCodes, combinationId, combinationSize, ...
        componentTargetPowerDb, weakComponentMask, ...
        componentSamplingWeight, patternSamplingWeight, imbalanceFlags, ...
        unknownPresent, unknownComponentCount, 'VariableNames', { ...
        'sample_id', 'split', 'sample_seed', 'active_type_ids', ...
        'active_type_codes', 'combination_id', 'combination_size', ...
        'component_target_power_dB', 'weak_component_mask', ...
        'component_sampling_weight', 'pattern_sampling_weight', ...
        'imbalance_flags', 'unknown_present', ...
        'unknown_component_count'});
end

function counts = allocate_largest_remainder(totalCount, ratios)
%ALLOCATE_LARGEST_REMAINDER 用稳定最大余数法得到精确的四类行数。

    rawCounts = double(totalCount) * double(ratios(:).');
    counts = floor(rawCounts);
    remaining = totalCount - sum(counts);
    fractionalParts = rawCounts - counts;
    sortKeys = [-fractionalParts(:), (1:4)'];
    [~, order] = sortrows(sortKeys, [1, 2]);
    counts(order(1:remaining)) = counts(order(1:remaining)) + 1;
end

function [ids, codes] = draw_open_set_pattern(stream, category, ...
        knownIds, knownCodes, unknownIds, unknownCodes, ...
        minimumComponents, maximumComponents)
%DRAW_OPEN_SET_PATTERN 按类别约束执行无放回类型抽样并规范排序。

    knownCount = numel(knownIds);
    unknownCount = numel(unknownIds);
    switch category
        case 1
            componentCount = draw_integer(stream, ...
                max(1, minimumComponents), ...
                min(maximumComponents, knownCount));
            selectedKnown = randperm(stream, knownCount, componentCount);
            ids = knownIds(selectedKnown);
            codes = knownCodes(selectedKnown);
        case 2
            componentCount = draw_integer(stream, ...
                max(1, minimumComponents), ...
                min(maximumComponents, unknownCount));
            selectedUnknown = randperm( ...
                stream, unknownCount, componentCount);
            ids = unknownIds(selectedUnknown);
            codes = unknownCodes(selectedUnknown);
        case 3
            componentCount = draw_integer(stream, ...
                max(2, minimumComponents), ...
                min(maximumComponents, knownCount + 1));
            selectedKnown = randperm( ...
                stream, knownCount, componentCount - 1);
            selectedUnknown = randperm(stream, unknownCount, 1);
            ids = [knownIds(selectedKnown), unknownIds(selectedUnknown)];
            codes = [knownCodes(selectedKnown), ...
                unknownCodes(selectedUnknown)];
        case 4
            componentCount = draw_integer(stream, ...
                max(3, minimumComponents), ...
                min(maximumComponents, knownCount + unknownCount));
            minimumUnknown = max(2, componentCount - knownCount);
            maximumUnknown = min(unknownCount, componentCount - 1);
            selectedUnknownCount = draw_integer( ...
                stream, minimumUnknown, maximumUnknown);
            selectedKnownCount = componentCount - selectedUnknownCount;
            selectedKnown = randperm( ...
                stream, knownCount, selectedKnownCount);
            selectedUnknown = randperm( ...
                stream, unknownCount, selectedUnknownCount);
            ids = [knownIds(selectedKnown), unknownIds(selectedUnknown)];
            codes = [knownCodes(selectedKnown), ...
                unknownCodes(selectedUnknown)];
        otherwise
            error('JammingOpenSet:InvalidComposition', ...
                '开放集类别编号必须位于 1 到 4。');
    end
    [ids, order] = sort(ids);
    codes = codes(order);
end

function value = draw_integer(stream, lowerBound, upperBound)
%DRAW_INTEGER 使用独立随机流在闭区间内均匀抽取一个整数。

    value = randi(stream, [lowerBound, upperBound], 1, 1);
end

function manifest = create_empty_manifest()
%CREATE_EMPTY_MANIFEST 创建与普通构建器列名称及列类型一致的空表。

    emptyDouble = zeros(0, 1);
    emptyCell = cell(0, 1);
    emptyLogical = false(0, 1);
    manifest = table(emptyDouble, emptyCell, emptyDouble, emptyCell, ...
        emptyCell, emptyCell, emptyDouble, emptyCell, emptyCell, ...
        emptyDouble, emptyDouble, emptyCell, emptyLogical, emptyDouble, ...
        'VariableNames', {'sample_id', 'split', 'sample_seed', ...
        'active_type_ids', 'active_type_codes', 'combination_id', ...
        'combination_size', 'component_target_power_dB', ...
        'weak_component_mask', 'component_sampling_weight', ...
        'pattern_sampling_weight', 'imbalance_flags', 'unknown_present', ...
        'unknown_component_count'});
end
