function stats = print_dataset_statistics(bundle, registry)
%PRINT_DATASET_STATISTICS 打印并返回组合干扰数据集统计摘要。
%   STATS = PRINT_DATASET_STATISTICS(BUNDLE, REGISTRY) 统计十类分量边际、
%   组合阶数、组合模式、弱分量数量、最终相对功率分位数，以及每个划分的
%   样本数和未知样本数。函数不修改数据，也不依赖 Statistics Toolbox。

    if ~isstruct(bundle) || ~isscalar(bundle) || ...
            ~all(isfield(bundle, {'label_all', 'manifest', ...
            'weak_component_mask', 'relative_power_dB'})) || ...
            ~istable(bundle.manifest)
        error('JammingDataset:InvalidStatisticsInput', ...
            'bundle 缺少统计所需字段。');
    end
    if ~isstruct(registry) || numel(registry) ~= size(bundle.label_all, 2)
        error('JammingDataset:InvalidStatisticsInput', ...
            'registry 数量必须与 label_all 列数一致。');
    end

    componentCounts = sum(double(bundle.label_all), 1);
    stats.component_counts = table(reshape({registry.code}, [], 1), ...
        componentCounts(:), 'VariableNames', {'type_code', 'count'});

    cardinalities = bundle.manifest.combination_size;
    uniqueCardinalities = unique(cardinalities, 'sorted');
    cardinalityCounts = arrayfun(@(value) sum(cardinalities == value), ...
        uniqueCardinalities);
    stats.cardinality_counts = table(uniqueCardinalities(:), ...
        cardinalityCounts(:), 'VariableNames', ...
        {'combination_size', 'count'});

    patterns = bundle.manifest.combination_id;
    [uniquePatterns, ~, patternGroup] = unique(patterns, 'stable');
    patternCounts = accumarray(patternGroup, 1);
    stats.pattern_counts = table(uniquePatterns(:), patternCounts(:), ...
        'VariableNames', {'combination_id', 'count'});

    stats.weak_component_count = sum(cellfun(@nnz, ...
        bundle.weak_component_mask));
    relativeValues = [bundle.relative_power_dB{:}];
    stats.relative_power_quantile_levels = [0, 25, 50, 75, 100];
    stats.relative_power_quantiles_dB = simple_percentiles( ...
        relativeValues, stats.relative_power_quantile_levels);

    splitNames = bundle.manifest.split;
    [uniqueSplits, ~, splitGroup] = unique(splitNames, 'stable');
    splitCounts = accumarray(splitGroup, 1);
    unknownCounts = accumarray(splitGroup, ...
        double(bundle.manifest.unknown_present));
    stats.open_set_split_counts = table(uniqueSplits(:), ...
        splitCounts(:), unknownCounts(:), 'VariableNames', ...
        {'split', 'sample_count', 'unknown_sample_count'});

    fprintf('\n分量边际计数：\n');
    disp(stats.component_counts);
    fprintf('组合阶数计数：\n');
    disp(stats.cardinality_counts);
    fprintf('组合模式计数：\n');
    disp(stats.pattern_counts);
    fprintf('弱分量总数：%d\n', stats.weak_component_count);
    fprintf('最终相对功率分位数 dB [0 25 50 75 100]：\n');
    disp(stats.relative_power_quantiles_dB);
    fprintf('开放集/划分计数：\n');
    disp(stats.open_set_split_counts);
end

function values = simple_percentiles(data, levels)
%SIMPLE_PERCENTILES 用排序和线性插值计算百分位数，不依赖附加工具箱。

    data = sort(double(data(isfinite(data))));
    if isempty(data)
        values = nan(size(levels));
        return;
    end
    values = zeros(size(levels));
    for index = 1:numel(levels)
        position = 1 + (numel(data) - 1) * levels(index) / 100;
        lowerIndex = floor(position);
        upperIndex = ceil(position);
        fraction = position - lowerIndex;
        values(index) = data(lowerIndex) * (1 - fraction) + ...
            data(upperIndex) * fraction;
    end
end
