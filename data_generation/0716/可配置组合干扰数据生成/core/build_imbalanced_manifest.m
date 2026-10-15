function manifest = build_imbalanced_manifest(cfg, registry, split)
%BUILD_IMBALANCED_MANIFEST 构建分量、组合和功率多维不均衡样本清单。
%   MANIFEST = BUILD_IMBALANCED_MANIFEST(CFG, REGISTRY, SPLIT) 接收已由
%   NORMALIZE_AND_VALIDATE_CONFIG(CFG, 'imbalanced') 标准化的配置、固定
%   干扰注册表和数据划分名称，返回与 BUILD_BALANCED_MANIFEST 完全相同的
%   14 列 table。sample_id、split 和 sample_seed 直接继承普通组合骨架，
%   因而沿用其划分、uint32 种子、实际输出容量和每划分一百万行安全约束。
%
%   分量不均衡使用 cfg.types.enabled 输入顺序对应的 component.weights，
%   逐步执行加权无放回抽样；component_sampling_weight 记录本次有序抽取
%   的条件概率乘积，最终活动分量仍按注册表 ID 规范排序。组合不均衡使用
%   min_components:max_components 对应的 cardinality_weights；若提供非空
%   pattern_weights，则直接按显式模式的规范化权重抽样，并忽略阶数和分量
%   权重。pattern_sampling_weight 记录实际采用的阶数或显式模式概率。
%
%   功率不均衡的 component_target_power_dB 单位为 dB，非弱分量严格为
%   0，弱分量在 relative_power_dB 闭区间内线性均匀抽样。random 策略在
%   每个组合内随机选择一个弱分量；rotate 按 sample_id 在组合位置间轮换；
%   fixed 使用 cfg.imbalance.power.fixed_component 指定的启用类型，仅当该
%   类型出现在当前样本中时标记为弱分量，否则该行全部为非弱分量。
%
%   三个 enable 全部关闭时，本函数直接返回普通组合 manifest。若仅启用
%   功率轴，则完整保留普通组合的活动模式和抽样权重，只写弱分量与功率
%   字段。需要重抽组合时，模式流使用 cfg.seed 加固定划分偏移，功率流
%   使用 uint32 域内确定性派生种子；两个独立 mt19937ar RandStream 互不
%   推进，也都不读取或修改 MATLAB 全局随机状态。
%
%   示例：
%       cfg = create_default_config();
%       cfg.types.enabled = {'FMB', 'MFT', 'ISRJ'};
%       cfg.types.known = cfg.types.enabled;
%       cfg.types.unknown = {};
%       cfg.imbalance.component.enable = true;
%       cfg.imbalance.component.weights = [10, 3, 1];
%       cfg.imbalance.power.enable = true;
%       cfg.imbalance.power.weak_component_policy = 'fixed';
%       cfg.imbalance.power.fixed_component = 'ISRJ';
%       cfg = normalize_and_validate_config(cfg, 'imbalanced');
%       registry = build_jamming_registry(cfg);
%       manifest = build_imbalanced_manifest(cfg, registry, 'train');

    componentEnabled = logical(cfg.imbalance.component.enable);
    combinationEnabled = logical(cfg.imbalance.combination.enable);
    powerEnabled = logical(cfg.imbalance.power.enable);

    if ~componentEnabled && ~combinationEnabled && ~powerEnabled
        manifest = build_balanced_manifest(cfg, registry, split);
        return;
    end

    % 普通构建器先完成全部规模、容量、种子和划分检查，并提供固定列类型。
    manifest = build_balanced_manifest(cfg, registry, split);
    sampleCount = height(manifest);
    splitOffset = manifest.sample_seed(1) - double(cfg.seed) - ...
        manifest.sample_id(1);
    patternSeed = double(cfg.seed) + splitOffset;
    powerSeed = mod(patternSeed + 2654435761, 2^32);
    if powerEnabled
        powerStream = RandStream('mt19937ar', 'Seed', powerSeed);
        [weakPolicy, fixedComponent] = validate_weak_policy( ...
            cfg.imbalance.power, cfg.types.enabled);
        relativePowerRange = double( ...
            cfg.imbalance.power.relative_power_dB);
    else
        powerStream = [];
        weakPolicy = '';
        fixedComponent = '';
        relativePowerRange = [0, 0];
    end

    imbalanceFlag = logical([componentEnabled, ...
        combinationEnabled, powerEnabled]);
    manifest.imbalance_flags = repmat({imbalanceFlag}, sampleCount, 1);

    if ~componentEnabled && ~combinationEnabled
        % power-only 路径只填写两列功率元数据，保留普通组合全部模式信息。
        manifest = apply_power_fields(manifest, powerEnabled, weakPolicy, ...
            fixedComponent, relativePowerRange, powerStream);
        return;
    end

    patternStream = RandStream('mt19937ar', 'Seed', patternSeed);

    registryIds = double([registry.id]);
    registryCodes = {registry.code};
    configuredCodes = cfg.types.enabled;
    [configuredFound, configuredLocations] = ismember( ...
        configuredCodes, registryCodes);
    if ~all(configuredFound)
        error('JammingManifest:UnknownEnabledType', ...
            'cfg.types.enabled 含无法映射到注册表的干扰类型。');
    end
    configuredIds = registryIds(configuredLocations);

    useManualPatterns = combinationEnabled && ...
        istable(cfg.imbalance.combination.pattern_weights) && ...
        ~isempty(cfg.imbalance.combination.pattern_weights);
    if useManualPatterns
        patternTable = cfg.imbalance.combination.pattern_weights;
        patternProbabilities = normalize_sampling_weights( ...
            patternTable.weight);
        [manualIds, manualCodes] = resolve_manual_patterns( ...
            patternTable.pattern_codes, registryIds, registryCodes);
    else
        patternProbabilities = [];
        manualIds = {};
        manualCodes = {};
    end

    componentWeights = double(cfg.imbalance.component.weights);
    minimumComponents = cfg.dataset.min_components;
    maximumComponents = cfg.dataset.max_components;
    allowedCardinalities = minimumComponents:maximumComponents;
    if combinationEnabled && ~useManualPatterns
        reachableCardinalityWeights = ...
            cfg.imbalance.combination.cardinality_weights( ...
            allowedCardinalities);
        cardinalityProbabilities = normalize_sampling_weights( ...
            reachableCardinalityWeights);
    else
        reachableCardinalityWeights = [];
        cardinalityProbabilities = [];
    end

    if componentEnabled && ~useManualPatterns
        validate_positive_component_candidates(componentWeights, ...
            allowedCardinalities, combinationEnabled, ...
            reachableCardinalityWeights);
    end

    componentSamplingWeight = ones(sampleCount, 1);
    patternSamplingWeight = ones(sampleCount, 1);

    for sampleIndex = 1:sampleCount
        if useManualPatterns
            patternIndex = draw_weighted_index( ...
                patternStream, patternProbabilities);
            ids = manualIds{patternIndex};
            codes = manualCodes{patternIndex};
            patternSamplingWeight(sampleIndex) = ...
                patternProbabilities(patternIndex);
        else
            if combinationEnabled
                cardinalityIndex = draw_weighted_index( ...
                    patternStream, cardinalityProbabilities);
                patternSamplingWeight(sampleIndex) = ...
                    cardinalityProbabilities(cardinalityIndex);
            else
                cardinalityIndex = randi( ...
                    patternStream, numel(allowedCardinalities));
            end
            componentCount = allowedCardinalities(cardinalityIndex);

            if componentEnabled
                [selectedPositions, selectionProbability] = ...
                    draw_weighted_without_replacement( ...
                    patternStream, componentWeights, componentCount);
                componentSamplingWeight(sampleIndex) = ...
                    selectionProbability;
            else
                selectedPositions = randperm( ...
                    patternStream, numel(configuredCodes), componentCount);
            end
            ids = configuredIds(selectedPositions);
            codes = configuredCodes(selectedPositions);
            [ids, canonicalOrder] = sort(ids);
            codes = codes(canonicalOrder);
        end

        componentCount = numel(ids);
        unknownCount = sum(ismember(codes, cfg.types.unknown));

        manifest.active_type_ids{sampleIndex} = ids;
        manifest.active_type_codes{sampleIndex} = codes;
        manifest.combination_id{sampleIndex} = strjoin(codes, '+');
        manifest.combination_size(sampleIndex) = componentCount;
        manifest.unknown_present(sampleIndex) = unknownCount > 0;
        manifest.unknown_component_count(sampleIndex) = unknownCount;
    end

    manifest.component_sampling_weight = componentSamplingWeight;
    manifest.pattern_sampling_weight = patternSamplingWeight;
    manifest = apply_power_fields(manifest, powerEnabled, weakPolicy, ...
        fixedComponent, relativePowerRange, powerStream);
end

function manifest = apply_power_fields(manifest, powerEnabled, policy, ...
        fixedComponent, relativePowerRange, powerStream)
%APPLY_POWER_FIELDS 仅按最终活动组合填写弱分量掩码和相对目标功率。

    for sampleIndex = 1:height(manifest)
        codes = manifest.active_type_codes{sampleIndex};
        componentCount = numel(codes);
        weakMask = select_weak_components(policy, fixedComponent, codes, ...
            manifest.sample_id(sampleIndex), powerStream, powerEnabled);
        targetPowerDb = zeros(1, componentCount);
        if any(weakMask)
            targetPowerDb(weakMask) = relativePowerRange(1) + ...
                diff(relativePowerRange) * rand( ...
                powerStream, 1, nnz(weakMask));
        end
        manifest.component_target_power_dB{sampleIndex} = targetPowerDb;
        manifest.weak_component_mask{sampleIndex} = weakMask;
    end
end

function validate_positive_component_candidates(weights, cardinalities, ...
        combinationEnabled, cardinalityWeights)
%VALIDATE_POSITIVE_COMPONENT_CANDIDATES 循环前校验可达阶数的正权重容量。

    if combinationEnabled
        reachableCardinalities = cardinalities(cardinalityWeights > 0);
    else
        reachableCardinalities = cardinalities;
    end
    if isempty(reachableCardinalities) || ...
            nnz(weights > 0) < max(reachableCardinalities)
        error('JammingManifest:InvalidSamplingWeights', ...
            '正 component 权重候选数量不足以生成全部可达组合阶数。');
    end
end

function probabilities = normalize_sampling_weights(weights)
%NORMALIZE_SAMPLING_WEIGHTS 稳定归一化有限非负权重并防御空或全零输入。

    if ~isnumeric(weights) || ~isreal(weights) || isempty(weights)
        error('JammingManifest:InvalidSamplingWeights', ...
            '抽样权重必须是非空实数数值向量。');
    end
    weights = reshape(double(weights), 1, []);
    if any(~isfinite(weights)) || any(weights < 0) || ~any(weights > 0)
        error('JammingManifest:InvalidSamplingWeights', ...
            '抽样权重必须有限、非负，且至少一个权重大于零。');
    end
    scaledWeights = weights / max(weights);
    totalWeight = sum(scaledWeights);
    if ~isfinite(totalWeight) || totalWeight <= 0
        error('JammingManifest:InvalidSamplingWeights', ...
            '抽样权重无法归一化为有效概率。');
    end
    probabilities = scaledWeights / totalWeight;
end

function index = draw_weighted_index(stream, weights)
%DRAW_WEIGHTED_INDEX 用累计概率和独立随机流返回一个有效权重下标。

    probabilities = reshape(double(weights), 1, []);
    if isempty(probabilities) || any(~isfinite(probabilities)) || ...
            any(probabilities < 0) || ~any(probabilities > 0)
        error('JammingManifest:InvalidSamplingWeights', ...
            '没有可供抽样的有效正权重候选项。');
    end
    cumulativeProbabilities = cumsum(probabilities);
    randomValue = rand(stream);
    index = find(randomValue < cumulativeProbabilities, 1, 'first');
    if isempty(index)
        % 浮点累计和可能略小于一，末端回退到最后一个正权重下标。
        index = find(probabilities > 0, 1, 'last');
    end
    if isempty(index)
        error('JammingManifest:InvalidSamplingWeights', ...
            '没有可供抽样的正权重候选项。');
    end
end

function [selectedPositions, probabilityProduct] = ...
        draw_weighted_without_replacement(stream, weights, drawCount)
%DRAW_WEIGHTED_WITHOUT_REPLACEMENT 序贯执行加权无放回抽样。

    availablePositions = 1:numel(weights);
    selectedPositions = zeros(1, drawCount);
    probabilityProduct = 1;
    for drawIndex = 1:drawCount
        if isempty(availablePositions)
            error('JammingManifest:InvalidSamplingWeights', ...
                '加权无放回抽样的候选数量不足。');
        end
        availableProbabilities = normalize_sampling_weights( ...
            weights(availablePositions));
        localIndex = draw_weighted_index(stream, availableProbabilities);
        selectedPositions(drawIndex) = availablePositions(localIndex);
        probabilityProduct = probabilityProduct * ...
            availableProbabilities(localIndex);
        availablePositions(localIndex) = [];
    end
end

function [patternIds, patternCodes] = resolve_manual_patterns( ...
        rawPatternCodes, registryIds, registryCodes)
%RESOLVE_MANUAL_PATTERNS 将显式组合文本映射并按注册表 ID 规范排序。

    patternCount = pattern_row_count(rawPatternCodes);
    patternIds = cell(patternCount, 1);
    patternCodes = cell(patternCount, 1);
    for patternIndex = 1:patternCount
        patternText = pattern_text_at(rawPatternCodes, patternIndex);
        codes = strsplit(patternText, '+', 'CollapseDelimiters', false);
        codes = cellfun(@strtrim, codes, 'UniformOutput', false);
        [found, locations] = ismember(codes, registryCodes);
        if isempty(codes) || ~all(found)
            error('JammingManifest:InvalidSamplingWeights', ...
                '显式组合模式无法映射到注册表。');
        end
        ids = registryIds(locations);
        [ids, canonicalOrder] = sort(ids);
        patternIds{patternIndex} = ids;
        patternCodes{patternIndex} = codes(canonicalOrder);
    end
end

function rowCount = pattern_row_count(rawPatternCodes)
%PATTERN_ROW_COUNT 返回显式模式文本列的行数。

    if ischar(rawPatternCodes)
        rowCount = size(rawPatternCodes, 1);
    else
        rowCount = numel(rawPatternCodes);
    end
end

function patternText = pattern_text_at(rawPatternCodes, rowIndex)
%PATTERN_TEXT_AT 从 char、string 或 cell 文本列中读取一个模式。

    if ischar(rawPatternCodes)
        patternText = strtrim(rawPatternCodes(rowIndex, :));
    elseif isstring(rawPatternCodes)
        patternText = char(rawPatternCodes(rowIndex));
    else
        patternText = rawPatternCodes{rowIndex};
        if isstring(patternText)
            patternText = char(patternText);
        end
    end
end

function [policy, fixedComponent] = validate_weak_policy(powerCfg, ...
        enabledCodes)
%VALIDATE_WEAK_POLICY 规范化弱分量策略并校验 fixed 类型。

    rawPolicy = powerCfg.weak_component_policy;
    if isstring(rawPolicy) && isscalar(rawPolicy) && ~ismissing(rawPolicy)
        policy = char(rawPolicy);
    elseif ischar(rawPolicy) && isrow(rawPolicy) && ~isempty(rawPolicy)
        policy = rawPolicy;
    else
        error('JammingManifest:InvalidWeakComponentPolicy', ...
            'weak_component_policy 必须是 random、rotate 或 fixed 文本标量。');
    end
    policy = lower(policy);
    if ~ismember(policy, {'random', 'rotate', 'fixed'})
        error('JammingManifest:InvalidWeakComponentPolicy', ...
            'weak_component_policy 仅允许 random、rotate 或 fixed。');
    end

    fixedComponent = '';
    if ~strcmp(policy, 'fixed')
        return;
    end
    if ~isfield(powerCfg, 'fixed_component')
        error('JammingManifest:InvalidWeakComponentPolicy', ...
            'fixed 策略必须提供 cfg.imbalance.power.fixed_component。');
    end
    fixedComponent = powerCfg.fixed_component;
    if isstring(fixedComponent) && isscalar(fixedComponent) && ...
            ~ismissing(fixedComponent)
        fixedComponent = char(fixedComponent);
    end
    if ~(ischar(fixedComponent) && isrow(fixedComponent) && ...
            ~isempty(fixedComponent) && ismember(fixedComponent, enabledCodes))
        error('JammingManifest:InvalidWeakComponentPolicy', ...
            'fixed_component 必须是 cfg.types.enabled 中的干扰代码。');
    end
end

function weakMask = select_weak_components(policy, fixedComponent, codes, ...
        sampleId, stream, powerEnabled)
%SELECT_WEAK_COMPONENTS 按功率轴策略生成一行逻辑弱分量掩码。

    componentCount = numel(codes);
    weakMask = false(1, componentCount);
    if ~powerEnabled
        return;
    end
    switch policy
        case 'random'
            weakMask(randi(stream, componentCount)) = true;
        case 'rotate'
            weakMask(mod(sampleId - 1, componentCount) + 1) = true;
        case 'fixed'
            weakMask = strcmp(codes, fixedComponent);
        otherwise
            error('JammingManifest:InvalidWeakComponentPolicy', ...
                '无法执行未知弱分量策略。');
    end
end
