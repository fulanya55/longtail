function manifest = build_balanced_manifest(cfg, registry, split)
%BUILD_BALANCED_MANIFEST 构建普通组合干扰数据的确定性样本清单。
%   MANIFEST = BUILD_BALANCED_MANIFEST(CFG, REGISTRY, SPLIT) 根据已完成
%   NORMALIZE_AND_VALIDATE_CONFIG(CFG, 'balanced') 的配置 CFG、干扰注册表
%   REGISTRY 和数据划分 SPLIT，返回一张每行对应一个待生成样本的 table。
%   SPLIT 可为字符行向量或 string 标量，仅允许 train、calibration、
%   validation 和 test。
%
%   表列语义如下：sample_id 为划分内从 1 开始的顺序编号；split 为划分
%   名称；sample_seed 为样本级独立种子；active_type_ids 和
%   active_type_codes 为按注册表 ID 升序排列的活动干扰；combination_id
%   为用加号连接代码得到的规范组合标识；combination_size 为分量数量；
%   component_target_power_dB 为各分量相对目标功率（单位 dB）；
%   weak_component_mask 标记弱分量；component_sampling_weight 和
%   pattern_sampling_weight 分别为分量与组合抽样权重；imbalance_flags
%   依次表示分量、组合和功率不平衡是否启用；unknown_present 和
%   unknown_component_count 表示活动组合中是否含未知类及其数量。
%
%   random 模式使用独立 mt19937ar RandStream，种子为 cfg.seed 加固定
%   划分偏移，不读取或修改 MATLAB 全局随机状态；每个样本先均匀抽取
%   组合阶数，再从启用类型中无放回抽取分量。all 模式按组合阶数和
%   nchoosek 字典顺序枚举全部模式，并将每种模式重复
%   cfg.dataset.samples_per_pattern 次。基础种子和最终样本种子均限制在
%   uint32 域内；每个划分最多一百万行，并在分配 manifest 内存前按实际
%   行数复核预计波形字节数。本函数只构建元数据，不生成波形，也不修改
%   输入 CFG 或 REGISTRY。
%
%   示例：
%       cfg = create_default_config();
%       cfg = normalize_and_validate_config(cfg, 'balanced');
%       registry = build_jamming_registry(cfg);
%       manifest = build_balanced_manifest(cfg, registry, 'train');

    [splitName, splitOffset] = normalize_split(split);
    [enabledIds, enabledCodes] = resolve_enabled_types( ...
        cfg.types.enabled, registry);
    combinationMode = normalize_combination_mode( ...
        cfg.dataset.combination_mode);
    sampleCount = calculate_actual_sample_count( ...
        cfg.dataset, numel(enabledIds), combinationMode);
    validate_split_size(sampleCount);
    numericSeed = validate_seed(cfg.seed);
    validate_sample_seed_range(numericSeed, splitOffset, sampleCount);
    validate_actual_output_capacity(cfg, sampleCount);
    selectedIndices = select_combinations(cfg.dataset, ...
        numel(enabledIds), combinationMode, sampleCount, ...
        numericSeed + splitOffset);

    sampleId = (1:sampleCount)';
    sampleSeed = numericSeed + splitOffset + sampleId;
    splitColumn = repmat({splitName}, sampleCount, 1);

    activeTypeIds = cell(sampleCount, 1);
    activeTypeCodes = cell(sampleCount, 1);
    combinationId = cell(sampleCount, 1);
    combinationSize = zeros(sampleCount, 1);
    componentTargetPowerDb = cell(sampleCount, 1);
    weakComponentMask = cell(sampleCount, 1);
    unknownPresent = false(sampleCount, 1);
    unknownComponentCount = zeros(sampleCount, 1);
    unknownCodes = cfg.types.unknown;

    for sampleIndex = 1:sampleCount
        componentIndices = selectedIndices{sampleIndex};
        ids = enabledIds(componentIndices);
        codes = enabledCodes(componentIndices);
        componentCount = numel(componentIndices);
        unknownCount = sum(ismember(codes, unknownCodes));

        activeTypeIds{sampleIndex} = ids;
        activeTypeCodes{sampleIndex} = codes;
        combinationId{sampleIndex} = strjoin(codes, '+');
        combinationSize(sampleIndex) = componentCount;
        componentTargetPowerDb{sampleIndex} = zeros(1, componentCount);
        weakComponentMask{sampleIndex} = false(1, componentCount);
        unknownPresent(sampleIndex) = unknownCount > 0;
        unknownComponentCount(sampleIndex) = unknownCount;
    end

    componentSamplingWeight = ones(sampleCount, 1);
    patternSamplingWeight = ones(sampleCount, 1);
    imbalanceFlags = repmat({logical([false, false, false])}, ...
        sampleCount, 1);

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

function [splitName, splitOffset] = normalize_split(split)
%NORMALIZE_SPLIT 规范化划分文本并返回固定种子偏移。

    if isstring(split) && isscalar(split) && ~ismissing(split)
        splitName = char(split);
    elseif ischar(split) && isrow(split)
        splitName = split;
    else
        error('JammingManifest:UnknownSplit', ...
            ['split 必须是字符行向量或 string 标量，且取值为 train、' ...
            'calibration、validation 或 test。']);
    end

    validSplits = {'train', 'calibration', 'validation', 'test'};
    splitOffsets = [0, 1000000, 2000000, 3000000];
    splitIndex = find(strcmp(splitName, validSplits), 1);
    if isempty(splitIndex)
        error('JammingManifest:UnknownSplit', ...
            ['未知数据划分 "%s"；仅允许 train、calibration、' ...
            'validation 或 test。'], splitName);
    end
    splitOffset = splitOffsets(splitIndex);
end

function [enabledIds, enabledCodes] = resolve_enabled_types( ...
        configuredCodes, registry)
%RESOLVE_ENABLED_TYPES 按注册表 ID 顺序映射全部启用类型。

    registryCodes = {registry.code};
    missingCodes = setdiff(configuredCodes, registryCodes, 'stable');
    if ~isempty(missingCodes)
        error('JammingManifest:UnknownEnabledType', ...
            'cfg.types.enabled 包含注册表中不存在的类型：%s。', ...
            strjoin(missingCodes, ', '));
    end

    enabledMask = ismember(registryCodes, configuredCodes);
    enabledEntries = registry(enabledMask);
    enabledIds = double([enabledEntries.id]);
    enabledCodes = {enabledEntries.code};
    [enabledIds, order] = sort(enabledIds);
    enabledCodes = enabledCodes(order);
end

function combinationMode = normalize_combination_mode(combinationMode)
%NORMALIZE_COMBINATION_MODE 规范化并校验组合构建模式。

    validString = isstring(combinationMode) && ...
        isscalar(combinationMode) && ~ismissing(combinationMode);
    validChar = ischar(combinationMode) && isrow(combinationMode) && ...
        ~isempty(combinationMode);
    if validString
        combinationMode = char(combinationMode);
    elseif ~validChar
        error('JammingManifest:UnknownCombinationMode', ...
            ['cfg.dataset.combination_mode 必须是非空字符行向量或 ' ...
            'string 标量，且取值为 random 或 all。']);
    end

    combinationMode = lower(combinationMode);
    if ~ismember(combinationMode, {'random', 'all'})
        error('JammingManifest:UnknownCombinationMode', ...
            'cfg.dataset.combination_mode 仅允许 random 或 all。');
    end
end

function sampleCount = calculate_actual_sample_count(dataset, ...
        enabledCount, combinationMode)
%CALCULATE_ACTUAL_SAMPLE_COUNT 在任何大分配前计算 manifest 真实行数。

    if strcmp(combinationMode, 'random')
        sampleCount = dataset.num_samples;
        return;
    end

    patternCount = 0;
    for componentCount = dataset.min_components:dataset.max_components
        countAtCardinality = nchoosek(enabledCount, componentCount);
        if ~isfinite(countAtCardinality) || ...
                countAtCardinality ~= fix(countAtCardinality)
            error('JammingManifest:SplitTooLarge', ...
                'all 模式的组合模式总数不是有限整数。');
        end
        patternCount = patternCount + countAtCardinality;
    end

    if patternCount > flintmax / dataset.samples_per_pattern
        error('JammingManifest:SplitTooLarge', ...
            'all 模式的实际样本行数超过可精确表示范围。');
    end
    sampleCount = patternCount * dataset.samples_per_pattern;
end

function validate_split_size(sampleCount)
%VALIDATE_SPLIT_SIZE 保证固定百万级偏移下的相邻划分种子不重叠。

    maximumSplitSamples = 1000000;
    if ~isnumeric(sampleCount) || ~isscalar(sampleCount) || ...
            ~isreal(sampleCount) || ~isfinite(sampleCount) || ...
            sampleCount ~= fix(sampleCount) || sampleCount < 1 || ...
            sampleCount > maximumSplitSamples
        error('JammingManifest:SplitTooLarge', ...
            '每个 manifest 的实际样本行数必须位于 1 到 1000000。');
    end
end

function numericSeed = validate_seed(seed)
%VALIDATE_SEED 校验基础种子是 uint32 数值域内的实数整数标量。

    maximumSeed = double(intmax('uint32'));
    if ~isnumeric(seed) || ~isscalar(seed) || ~isreal(seed) || ...
            ~isfinite(seed)
        error('JammingManifest:InvalidSeed', ...
            'cfg.seed 必须是 uint32 取值域内的有限非负整数数值标量。');
    end

    numericSeed = double(seed);
    if numericSeed ~= fix(numericSeed) || numericSeed < 0 || ...
            numericSeed > maximumSeed
        error('JammingManifest:InvalidSeed', ...
            'cfg.seed 必须是 uint32 取值域内的有限非负整数数值标量。');
    end
end

function validate_sample_seed_range(numericSeed, splitOffset, sampleCount)
%VALIDATE_SAMPLE_SEED_RANGE 检查最终样本种子不超过 uint32 最大值。

    maximumSeed = double(intmax('uint32'));
    if numericSeed > maximumSeed - splitOffset - sampleCount
        error('JammingManifest:SeedOverflow', ...
            ['cfg.seed、划分偏移与实际 sample_id 的和不能超过 ' ...
            '2^32-1。']);
    end
end

function validate_actual_output_capacity(cfg, sampleCount)
%VALIDATE_ACTUAL_OUTPUT_CAPACITY 按真实行数复核预计波形输出字节数。

    bytesPerPoint = 16 + 8 * double(cfg.output.save_legacy);
    estimatedBytes = sampleCount * cfg.derived.sample_length * bytesPerPoint;
    if ~isfinite(estimatedBytes) || ...
            estimatedBytes > cfg.output.max_estimated_bytes
        error('JammingConfig:EstimatedSizeExceeded', ...
            ['按 manifest 实际 %d 行估算的输出大小为 %.0f 字节，超过 ' ...
            'cfg.output.max_estimated_bytes=%.0f 字节。'], ...
            sampleCount, estimatedBytes, cfg.output.max_estimated_bytes);
    end
end

function selectedIndices = select_combinations(dataset, enabledCount, ...
        combinationMode, sampleCount, streamSeed)
%SELECT_COMBINATIONS 在全部前置检查通过后预分配并填写类型下标行。

    minComponents = dataset.min_components;
    maxComponents = dataset.max_components;
    selectedIndices = cell(sampleCount, 1);
    switch combinationMode
        case 'random'
            stream = RandStream('mt19937ar', 'Seed', streamSeed);
            cardinalities = randi(stream, ...
                [minComponents, maxComponents], sampleCount, 1);
            for sampleIndex = 1:sampleCount
                selectedIndices{sampleIndex} = sort(randperm( ...
                    stream, enabledCount, cardinalities(sampleIndex)));
            end
        case 'all'
            sampleIndex = 0;
            for componentCount = minComponents:maxComponents
                combinations = nchoosek(1:enabledCount, componentCount);
                for patternIndex = 1:size(combinations, 1)
                    for repeatIndex = 1:dataset.samples_per_pattern
                        sampleIndex = sampleIndex + 1;
                        selectedIndices{sampleIndex} = ...
                            combinations(patternIndex, :);
                    end
                end
            end
    end
end
