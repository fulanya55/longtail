function bundle = generate_dataset_from_manifest( ...
        cfg, registry, schema, manifest)
%GENERATE_DATASET_FROM_MANIFEST 按 manifest 批量生成数据集内存包。
%   BUNDLE = GENERATE_DATASET_FROM_MANIFEST(CFG, REGISTRY, SCHEMA,
%   MANIFEST) 构造一次参考 LFM 信号，逐行调用 COMPOSE_JAMMING_SAMPLE，
%   并把定长字段预分配为数值矩阵，把随组合阶数变化的分量功率、种子和
%   弱分量信息保存为 N×1 单元数组。MANIFEST 原表完整保留在 BUNDLE 中。
%
%   BUNDLE.iq_complex 为 N×sample_length complex double；若
%   cfg.output.save_legacy=true，同时生成旧程序兼容的
%   data_legacy=[real(iq_complex), imag(iq_complex)]。metadata 保存标准化
%   配置、无函数句柄的注册表快照、参数模式、MATLAB 版本和 UTC 时间。

    validate_inputs(cfg, registry, schema, manifest);
    sampleCount = height(manifest);
    sampleLength = cfg.derived.sample_length;
    parameterCount = height(schema);
    registryCount = numel(registry);
    enabledCount = sum(ismember({registry.code}, cfg.types.enabled));
    knownCount = sum(ismember({registry.code}, cfg.types.known));
    reference = build_reference_signal(cfg);

    bundle = struct();
    bundle.iq_complex = complex(zeros(sampleCount, sampleLength));
    bundle.label_all = false(sampleCount, registryCount);
    bundle.label_selected = false(sampleCount, enabledCount);
    bundle.known_label = false(sampleCount, knownCount);
    bundle.unknown_present = false(sampleCount, 1);
    bundle.unknown_component_count = zeros(sampleCount, 1);
    bundle.parameter_values = nan(sampleCount, parameterCount);
    bundle.parameter_mask = false(sampleCount, parameterCount);
    bundle.component_power_offset_dB = cell(sampleCount, 1);
    bundle.component_target_power_dB = cell(sampleCount, 1);
    bundle.component_measured_power_dB = cell(sampleCount, 1);
    bundle.relative_power_dB = cell(sampleCount, 1);
    bundle.weak_component_mask = cell(sampleCount, 1);
    bundle.component_nominal_JNR_dB = cell(sampleCount, 1);
    bundle.component_effective_JNR_dB = cell(sampleCount, 1);
    bundle.component_nominal_amplitude = cell(sampleCount, 1);
    bundle.component_effective_amplitude = cell(sampleCount, 1);
    bundle.component_seeds = cell(sampleCount, 1);
    bundle.receiver_noise_seed = zeros(sampleCount, 1);
    bundle.target_present = false(sampleCount, 1);
    bundle.target_delay_s = nan(sampleCount, 1);
    bundle.target_range_m = nan(sampleCount, 1);
    bundle.target_target_power_dB = nan(sampleCount, 1);
    bundle.target_measured_power_dB = nan(sampleCount, 1);

    for sampleIndex = 1:sampleCount
        sample = compose_jamming_sample(cfg, registry, schema, ...
            manifest(sampleIndex, :), reference);
        bundle.iq_complex(sampleIndex, :) = sample.iq_complex;
        bundle.label_all(sampleIndex, :) = sample.label_all;
        bundle.label_selected(sampleIndex, :) = sample.label_selected;
        bundle.known_label(sampleIndex, :) = sample.known_label;
        bundle.unknown_present(sampleIndex) = sample.unknown_present;
        bundle.unknown_component_count(sampleIndex) = ...
            sample.unknown_component_count;
        bundle.parameter_values(sampleIndex, :) = sample.parameter_values;
        bundle.parameter_mask(sampleIndex, :) = sample.parameter_mask;
        bundle.component_power_offset_dB{sampleIndex} = ...
            sample.component_power_offset_dB;
        bundle.component_target_power_dB{sampleIndex} = ...
            sample.component_target_power_dB;
        bundle.component_measured_power_dB{sampleIndex} = ...
            sample.component_measured_power_dB;
        bundle.relative_power_dB{sampleIndex} = sample.relative_power_dB;
        bundle.weak_component_mask{sampleIndex} = ...
            sample.weak_component_mask;
        bundle.component_nominal_JNR_dB{sampleIndex} = ...
            sample.component_nominal_JNR_dB;
        bundle.component_effective_JNR_dB{sampleIndex} = ...
            sample.component_effective_JNR_dB;
        bundle.component_nominal_amplitude{sampleIndex} = ...
            sample.component_nominal_amplitude;
        bundle.component_effective_amplitude{sampleIndex} = ...
            sample.component_effective_amplitude;
        bundle.component_seeds{sampleIndex} = sample.component_seeds;
        bundle.receiver_noise_seed(sampleIndex) = ...
            sample.receiver_noise_seed;
        bundle.target_present(sampleIndex) = sample.target_present;
        bundle.target_delay_s(sampleIndex) = sample.target_delay_s;
        bundle.target_range_m(sampleIndex) = sample.target_range_m;
        bundle.target_target_power_dB(sampleIndex) = ...
            sample.target_target_power_dB;
        bundle.target_measured_power_dB(sampleIndex) = ...
            sample.target_measured_power_dB;
    end

    if cfg.output.save_legacy
        bundle.data_legacy = [real(bundle.iq_complex), ...
            imag(bundle.iq_complex)];
    end
    bundle.manifest = manifest;
    bundle.metadata = struct( ...
        'config', cfg, ...
        'registry_snapshot', registry_snapshot(registry), ...
        'parameter_schema', schema, ...
        'scientific_semantics_version', ...
        current_scientific_semantics_version(), ...
        'matlab_version', version, ...
        'generated_utc', utc_timestamp());
end

function validate_inputs(cfg, registry, schema, manifest)
%VALIDATE_INPUTS 防御性检查批量生成所需的定长信息。

    if ~isstruct(cfg) || ~isscalar(cfg) || ~isfield(cfg, 'derived') || ...
            ~isfield(cfg.derived, 'sample_length')
        error('JammingDataset:InvalidInput', ...
            'cfg 必须是已标准化且含 derived.sample_length 的标量结构体。');
    end
    if ~isstruct(registry) || isempty(registry) || ...
            ~all(isfield(registry, {'code', 'generator'}))
        error('JammingDataset:InvalidInput', 'registry 不是合法注册表。');
    end
    if ~istable(schema) || height(schema) < 1 || ...
            ~all(ismember({'field_name', 'owner_code'}, ...
            schema.Properties.VariableNames))
        error('JammingDataset:InvalidInput', 'schema 不是合法参数模式表。');
    end
    if ~istable(manifest)
        error('JammingDataset:InvalidInput', 'manifest 必须是 table。');
    end
end

function snapshot = registry_snapshot(registry)
%REGISTRY_SNAPSHOT 生成可移植、无函数句柄的注册表快照。

    generatorNames = arrayfun(@(entry) func2str(entry.generator), ...
        registry, 'UniformOutput', false);
    snapshot = rmfield(registry, 'generator');
    for index = 1:numel(snapshot)
        snapshot(index).generator_name = generatorNames{index};
    end
end

function timestamp = utc_timestamp()
%UTC_TIMESTAMP 返回含时区的 ISO 风格 UTC 生成时间文本。

    nowUtc = datetime('now', 'TimeZone', 'UTC', ...
        'Format', "yyyy-MM-dd'T'HH:mm:ss.SSSXXX");
    timestamp = char(nowUtc);
end
