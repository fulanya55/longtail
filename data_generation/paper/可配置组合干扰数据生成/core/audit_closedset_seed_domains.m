function audit = audit_closedset_seed_domains(configs, names)
%AUDIT_CLOSEDSET_SEED_DOMAINS 核对多个实验train/test种子域无交集。
%   AUDIT = AUDIT_CLOSEDSET_SEED_DOMAINS(CONFIGS, NAMES) 对每个代表配置
%   重新构建train和test manifest，并对全部种子向量执行两两INTERSECT。
%   九挡实验只需传入任意一个挡位代表配置，因为run_power_sweep_levels
%   已独立证明挡位间sample_seed完全一致。任一交集立即抛出稳定错误，
%   通过时返回8域等可读签名，不保留大型manifest。

    validate_inputs(configs, names);
    domainCount = 2 * numel(configs);
    vectors = cell(domainCount, 1);
    labels = cell(domainCount, 1);
    counts = zeros(domainCount, 1);
    firstSeeds = zeros(domainCount, 1);
    lastSeeds = zeros(domainCount, 1);
    cursor = 0;

    for configIndex = 1:numel(configs)
        cfg = normalize_and_validate_config( ...
            configs{configIndex}, 'balanced');
        registry = build_jamming_registry(cfg);
        for splitCell = {'train', 'test'}
            splitName = splitCell{1};
            manifest = build_closedset_manifest(cfg, registry, splitName);
            cursor = cursor + 1;
            vectors{cursor} = double(manifest.sample_seed(:));
            labels{cursor} = sprintf( ...
                '%s:%s', names{configIndex}, splitName);
            counts(cursor) = numel(vectors{cursor});
            firstSeeds(cursor) = vectors{cursor}(1);
            lastSeeds(cursor) = vectors{cursor}(end);
            clear manifest
        end
    end

    for left = 1:domainCount
        for right = left + 1:domainCount
            if ~isempty(intersect(vectors{left}, vectors{right}))
                error('JammingSubset:SeedDomainOverlap', ...
                    '%s与%s的sample_seed发生交集。', ...
                    labels{left}, labels{right});
            end
        end
    end

    audit = struct();
    audit.verified = true;
    audit.domain_count = domainCount;
    audit.labels = labels;
    audit.counts = counts;
    audit.first_seeds = firstSeeds;
    audit.last_seeds = lastSeeds;
end

function validate_inputs(configs, names)
%VALIDATE_INPUTS 校验配置cell和一一对应的非空名称。

    validConfigs = iscell(configs) && ~isempty(configs) && ...
        all(cellfun(@(cfg) isstruct(cfg) && isscalar(cfg), configs));
    validNames = iscell(names) && numel(names) == numel(configs) && ...
        all(cellfun(@(name) ischar(name) && isrow(name) && ...
        ~isempty(strtrim(name)), names));
    if ~(validConfigs && validNames)
        error('JammingSubset:InvalidSeedAuditInput', ...
            'configs与names必须是等长的有效配置和名称cell。');
    end
end
