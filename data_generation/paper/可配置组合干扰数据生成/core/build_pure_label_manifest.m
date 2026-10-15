function manifest = build_pure_label_manifest(cfg, registry, split)
%BUILD_PURE_LABEL_MANIFEST Build an exact-count six-class label-LT manifest.
%   The count vector is supplied by cfg.experiment.pure_label_longtail and
%   has one entry for each mask 1:63 (bit 1..6 = FMB..ISRJ).  No waveform or
%   physical parameter is changed here; only the number of rows per label
%   combination is changed.  Train/test use the existing split seed offsets.

    validate_inputs(cfg, registry, split);
    profile = cfg.experiment.pure_label_longtail;
    splitName = char(split);
    if strcmp(splitName, 'train')
        counts = profile.counts_train;
    elseif strcmp(splitName, 'test')
        counts = profile.counts_test;
    else
        error('JammingManifest:PureLabelSplit', ...
            'pure label long-tail 仅支持 train 或 test。');
    end
    counts = double(reshape(counts, 1, []));
    if numel(counts) ~= 63 || any(~isfinite(counts)) || ...
            any(counts < 0) || any(counts ~= fix(counts)) || ...
            sum(counts) < 1
        error('JammingManifest:PureLabelCounts', ...
            'counts_train/counts_test 必须是长度63的非负整数向量且总数>0。');
    end
    minPerPattern = double(profile.min_per_pattern);
    active = counts > 0;
    if any(counts(active) < minPerPattern)
        error('JammingManifest:PureLabelCounts', ...
            '所有保留组合的样本数必须不低于 min_per_pattern。');
    end

    splitOffsets = struct('train', 0, 'test', 3000000);
    splitOffset = splitOffsets.(splitName);
    sampleCount = sum(counts);
    if sampleCount > 1000000
        error('JammingManifest:SplitTooLarge', ...
            '纯标签长尾 manifest 行数不能超过1000000。');
    end
    numericSeed = double(cfg.seed);
    if numericSeed < 0 || numericSeed ~= fix(numericSeed) || ...
            numericSeed > double(intmax('uint32')) - splitOffset - sampleCount
        error('JammingManifest:SeedOverflow', ...
            'cfg.seed、划分偏移和 sample_id 超出 uint32 种子域。');
    end
    validate_output_capacity(cfg, sampleCount);

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
    componentSamplingWeight = zeros(sampleCount, 1);
    patternSamplingWeight = zeros(sampleCount, 1);
    % Keep the legacy scenario flags false so the existing balanced physical
    % validator remains applicable.  The authoritative axis declaration is
    % cfg.experiment.pure_label_longtail (stored in config/metadata).
    imbalanceFlags = repmat({logical([false, false, false])}, sampleCount, 1);

    row = 0;
    registryCodes = {registry.code};
    enabledIds = zeros(1, 6);
    for enabledIndex = 1:6
        location = find(strcmp(registryCodes, cfg.types.enabled{enabledIndex}), 1);
        if isempty(location)
            error('JammingManifest:PureLabelConfig', ...
                '六类类型无法映射到 registry。');
        end
        enabledIds(enabledIndex) = registry(location).id;
    end
    for mask = 1:63
        curCount = counts(mask);
        if curCount == 0
            continue;
        end
        localIds = find(bitget(mask, 1:6));
        ids = enabledIds(localIds);
        codes = cfg.types.enabled(localIds);
        for repeatIndex = 1:curCount
            row = row + 1;
            activeTypeIds{row} = ids;
            activeTypeCodes{row} = codes;
            combinationId{row} = strjoin(codes, '+');
            combinationSize(row) = numel(ids);
            componentTargetPowerDb{row} = zeros(1, numel(ids));
            weakComponentMask{row} = false(1, numel(ids));
            componentSamplingWeight(row) = 1;
            patternSamplingWeight(row) = curCount / sampleCount;
        end
    end
    if row ~= sampleCount
        error('JammingManifest:PureLabelCounts', ...
            '计数向量展开后的行数与总数不一致。');
    end
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

function validate_inputs(cfg, registry, split)
    if ~isstruct(cfg) || ~isfield(cfg, 'experiment') || ...
            ~isfield(cfg.experiment, 'pure_label_longtail')
        error('JammingManifest:PureLabelConfig', ...
            '缺少 cfg.experiment.pure_label_longtail。');
    end
    if ~(ischar(split) && isrow(split))
        error('JammingManifest:PureLabelSplit', 'split 必须是字符行向量。');
    end
    if ~strcmp(split, 'train') && ~strcmp(split, 'test')
        error('JammingManifest:PureLabelSplit', ...
            'split 仅允许 train 或 test。');
    end
    enabled = cfg.types.enabled;
    allowedAma = {'FMB','AMA','SawFM','SinFM','MFT','ISRJ'};
    allowedVft = {'FMB','SawFM','SinFM','MFT','VFT','ISRJ'};
    if numel(enabled) ~= 6 || ...
            ~(isequal(sort(enabled), sort(allowedAma)) || ...
              isequal(sort(enabled), sort(allowedVft)))
        error('JammingManifest:PureLabelConfig', ...
            '纯标签六类入口要求使用 AMA 版或 VFT 版六类集合。');
    end
    if ~isstruct(registry) || numel(registry) < 6
        error('JammingManifest:PureLabelConfig', 'registry 不完整。');
    end
end

function validate_output_capacity(cfg, sampleCount)
    if ~isfield(cfg, 'output') || ~isfield(cfg.output, 'max_estimated_bytes')
        return;
    end
    bytesPerPoint = 16 + 8 * double(cfg.output.save_legacy);
    estimate = double(sampleCount) * double(cfg.derived.sample_length) * bytesPerPoint;
    if estimate > double(cfg.output.max_estimated_bytes)
        error('JammingManifest:OutputTooLarge', ...
            '估算输出 %.0f 字节超过 max_estimated_bytes。', estimate);
    end
end
