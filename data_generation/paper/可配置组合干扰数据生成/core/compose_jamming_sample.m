function sample = compose_jamming_sample( ...
        cfg, registry, schema, manifestRow, reference)
%COMPOSE_JAMMING_SAMPLE 合成一个组合干扰样本并生成多任务监督真值。
%   SAMPLE = COMPOSE_JAMMING_SAMPLE(CFG, REGISTRY, SCHEMA, MANIFESTROW)
%   接收已经标准化的配置、一致的 v1/v2 注册表、固定参数模式和恰好一行
%   manifest。函数为每个活动类型派生互不混用的局部 RandStream，生成
%   不含接收机噪声的干净分量，施加真实功率规则，叠加可选目标回波，
%   最后只加入一次复高斯接收机背景噪声。
%
%   manifestRow.component_target_power_dB 在三个 manifest builder 中表示
%   相对功率偏置，不是绝对目标功率。对所有声明 JNR_dB 的类型，最终
%   JNR = 生成器采样 JNR + 相对偏置，绝对目标功率再加噪声功率 dB；
%   JNR 参数监督回写为最终有效值。Traction 的 amplitude 是唯一全局
%   幅度监督，禁止另采 JNR 或绝对功率归一化；偏置仅换算为线性幅度
%   增益并并入最终 amplitude 真值。
%
%   可选第五参数 REFERENCE 可由批量数据集生成器构造一次后复用。SAMPLE
%   同时返回全注册表/启用类/已知类标签、参数 values/mask、原始功率偏置、
%   绝对目标和实测功率、最终样本内相对功率、弱分量标记、可复现种子及
%   manifest 标识。所有波形均为 1×cfg.derived.sample_length complex double。

    row = validate_manifest_row(cfg, registry, manifestRow);
    if nargin < 5
        reference = build_reference_signal(cfg);
    end

    componentCount = row.combination_size;
    sampleLength = cfg.derived.sample_length;
    noisePower_dB = 10 * log10(cfg.radar.noise_power);
    componentSignals = complex(zeros(componentCount, sampleLength));
    componentParams = cell(1, componentCount);
    componentSeeds = zeros(1, componentCount);
    targetPower_dB = nan(1, componentCount);
    measuredPower_dB = nan(1, componentCount);
    nominalJnr_dB = nan(1, componentCount);
    effectiveJnr_dB = nan(1, componentCount);
    nominalAmplitude = nan(1, componentCount);
    effectiveAmplitude = nan(1, componentCount);

    for componentIndex = 1:componentCount
        typeId = row.active_type_ids(componentIndex);
        code = row.active_type_codes{componentIndex};
        componentSeeds(componentIndex) = derive_substream_seed( ...
            row.sample_seed, 1, typeId);
        stream = RandStream('mt19937ar', ...
            'Seed', componentSeeds(componentIndex));
        [rawSignal, params] = generate_single_jamming( ...
            code, cfg, registry, stream, reference);
        offset_dB = row.component_power_offset_dB(componentIndex);

        if strcmp(code, 'Traction')
            nominalAmplitude(componentIndex) = params.amplitude;
            amplitudeGain = 10 ^ (offset_dB / 20);
            if ~isfinite(amplitudeGain) || amplitudeGain <= 0
                throw_invalid_power( ...
                    'Traction 相对功率偏置不能转换为有限正幅度增益。');
            end
            rawPower_dB = measure_power_dB(rawSignal);
            scaledSignal = rawSignal * amplitudeGain;
            if ~all(isfinite(real(scaledSignal))) || ...
                    ~all(isfinite(imag(scaledSignal)))
                throw_invalid_power('Traction 偏置后的波形含非有限值。');
            end
            params.amplitude = params.amplitude * amplitudeGain;
            effectiveAmplitude(componentIndex) = params.amplitude;
            targetPower_dB(componentIndex) = rawPower_dB + offset_dB;
            measuredPower_dB(componentIndex) = ...
                measure_power_dB(scaledSignal);
            effectiveJnr_dB(componentIndex) = ...
                targetPower_dB(componentIndex) - noisePower_dB;
        else
            if ~isfield(params, 'JNR_dB') || ...
                    ~isfinite(params.JNR_dB)
                throw_invalid_power( ...
                    sprintf('类型 %s 缺少有限 JNR_dB 真值。', code));
            end
            nominalJnr_dB(componentIndex) = double(params.JNR_dB);
            effectiveJnr_dB(componentIndex) = ...
                nominalJnr_dB(componentIndex) + offset_dB;
            targetPower_dB(componentIndex) = noisePower_dB + ...
                effectiveJnr_dB(componentIndex);
            [scaledSignal, measuredPower_dB(componentIndex)] = ...
                calibrate_component_power(rawSignal, ...
                targetPower_dB(componentIndex));
            params.JNR_dB = effectiveJnr_dB(componentIndex);
        end
        validate_observable_component_params(code, params, cfg, ...
            nominalAmplitude(componentIndex), ...
            effectiveAmplitude(componentIndex), offset_dB);
        componentSignals(componentIndex, :) = scaledSignal;
        componentParams{componentIndex} = params;
    end

    cleanJamming = sum(componentSignals, 1);
    [targetSignal, targetMeta] = build_optional_target(cfg, reference, ...
        noisePower_dB);
    receiverNoiseSeed = derive_substream_seed(row.sample_seed, 2, 0);
    noiseStream = RandStream('mt19937ar', 'Seed', receiverNoiseSeed);
    receiverNoise = sqrt(cfg.radar.noise_power / 2) * ( ...
        randn(noiseStream, 1, sampleLength) + ...
        1i * randn(noiseStream, 1, sampleLength));
    iqComplex = cleanJamming + targetSignal + receiverNoise;
    iqComplex = complex(double(real(iqComplex)), double(imag(iqComplex)));

    [parameterValues, parameterMask] = encode_parameter_targets( ...
        schema, registry, row.active_type_codes, componentParams);
    registryCodes = {registry.code};
    labelAll = false(1, numel(registry));
    labelAll(row.active_type_ids) = true;
    enabledMask = ismember(registryCodes, cfg.types.enabled);
    knownMask = ismember(registryCodes, cfg.types.known);

    sample = struct();
    sample.iq_complex = iqComplex;
    sample.label_all = labelAll;
    sample.label_selected = labelAll(enabledMask);
    sample.known_label = labelAll(knownMask);
    sample.unknown_present = row.unknown_present;
    sample.unknown_component_count = row.unknown_component_count;
    sample.parameter_values = parameterValues;
    sample.parameter_mask = parameterMask;
    sample.active_type_ids = row.active_type_ids;
    sample.active_type_codes = row.active_type_codes;
    sample.component_power_offset_dB = row.component_power_offset_dB;
    sample.component_target_power_dB = targetPower_dB;
    sample.component_measured_power_dB = measuredPower_dB;
    sample.relative_power_dB = targetPower_dB - max(targetPower_dB);
    sample.weak_component_mask = row.weak_component_mask;
    sample.component_nominal_JNR_dB = nominalJnr_dB;
    sample.component_effective_JNR_dB = effectiveJnr_dB;
    sample.component_nominal_amplitude = nominalAmplitude;
    sample.component_effective_amplitude = effectiveAmplitude;
    sample.component_seeds = componentSeeds;
    sample.receiver_noise_seed = receiverNoiseSeed;
    sample.target_present = targetMeta.present;
    sample.target_delay_s = targetMeta.delay_s;
    sample.target_range_m = targetMeta.range_m;
    sample.target_target_power_dB = targetMeta.target_power_dB;
    sample.target_measured_power_dB = targetMeta.measured_power_dB;
    sample.sample_id = row.sample_id;
    sample.sample_seed = row.sample_seed;
    sample.split = row.split;
    sample.combination_id = row.combination_id;
    sample.combination_size = componentCount;
    sample.component_sampling_weight = row.component_sampling_weight;
    sample.pattern_sampling_weight = row.pattern_sampling_weight;
    sample.imbalance_flags = row.imbalance_flags;
end

function row = validate_manifest_row(cfg, registry, manifestRow)
%VALIDATE_MANIFEST_ROW 校验一行 manifest 的列、类型映射和长度一致性。

    requiredColumns = {'sample_id', 'split', 'sample_seed', ...
        'active_type_ids', 'active_type_codes', 'combination_id', ...
        'combination_size', 'component_target_power_dB', ...
        'weak_component_mask', 'component_sampling_weight', ...
        'pattern_sampling_weight', 'imbalance_flags', ...
        'unknown_present', 'unknown_component_count'};
    if ~istable(manifestRow) || height(manifestRow) ~= 1 || ...
            ~all(ismember(requiredColumns, ...
            manifestRow.Properties.VariableNames))
        throw_invalid_manifest('manifestRow 必须是含规定14列的一行 table。');
    end

    row.sample_id = numeric_integer_scalar(manifestRow.sample_id(1), ...
        1, flintmax, 'sample_id');
    row.sample_seed = numeric_integer_scalar(manifestRow.sample_seed(1), ...
        0, double(intmax('uint32')), 'sample_seed');
    row.split = text_scalar(manifestRow.split{1}, 'split');
    row.active_type_ids = numeric_row(manifestRow.active_type_ids{1}, ...
        'active_type_ids');
    row.active_type_codes = text_list( ...
        manifestRow.active_type_codes{1}, 'active_type_codes');
    row.combination_id = text_scalar( ...
        manifestRow.combination_id{1}, 'combination_id');
    row.combination_size = numeric_integer_scalar( ...
        manifestRow.combination_size(1), 1, numel(registry), ...
        'combination_size');
    row.component_power_offset_dB = numeric_row( ...
        manifestRow.component_target_power_dB{1}, ...
        'component_target_power_dB');
    row.weak_component_mask = logical_row( ...
        manifestRow.weak_component_mask{1}, 'weak_component_mask');
    row.component_sampling_weight = positive_scalar( ...
        manifestRow.component_sampling_weight(1), ...
        'component_sampling_weight');
    row.pattern_sampling_weight = positive_scalar( ...
        manifestRow.pattern_sampling_weight(1), ...
        'pattern_sampling_weight');
    row.imbalance_flags = logical_row( ...
        manifestRow.imbalance_flags{1}, 'imbalance_flags');
    row.unknown_present = logical_scalar( ...
        manifestRow.unknown_present(1), 'unknown_present');
    row.unknown_component_count = numeric_integer_scalar( ...
        manifestRow.unknown_component_count(1), 0, numel(registry), ...
        'unknown_component_count');

    componentCount = row.combination_size;
    if numel(row.active_type_ids) ~= componentCount || ...
            numel(row.active_type_codes) ~= componentCount || ...
            numel(row.component_power_offset_dB) ~= componentCount || ...
            numel(row.weak_component_mask) ~= componentCount || ...
            numel(row.imbalance_flags) ~= 3
        throw_invalid_manifest('活动类型、功率、弱分量或标志列长度不一致。');
    end
    if any(row.active_type_ids ~= floor(row.active_type_ids)) || ...
            any(row.active_type_ids < 1) || ...
            any(row.active_type_ids > numel(registry))
        throw_invalid_manifest( ...
            'active_type_ids 必须是注册表范围内的正整数 ID。');
    end
    if componentCount < cfg.dataset.min_components || ...
            componentCount > cfg.dataset.max_components
        throw_invalid_manifest( ...
            'combination_size 必须位于配置的 min/max_components 范围内。');
    end
    if numel(unique(row.active_type_ids, 'stable')) ~= componentCount || ...
            numel(unique(row.active_type_codes, 'stable')) ~= componentCount
        throw_invalid_manifest('活动类型 ID 和代码不能重复。');
    end
    if any(diff(row.active_type_ids) <= 0)
        throw_invalid_manifest('活动类型必须按注册表 ID 严格升序排列。');
    end
    registryCodes = {registry.code};
    if ~isequal(row.active_type_codes, ...
            registryCodes(row.active_type_ids))
        throw_invalid_manifest('active_type_ids 与 active_type_codes 不匹配。');
    end
    if ~all(ismember(row.active_type_codes, cfg.types.enabled))
        throw_invalid_manifest('活动类型必须全部属于 cfg.types.enabled。');
    end
    if ~strcmp(row.combination_id, strjoin(row.active_type_codes, '+'))
        throw_invalid_manifest('combination_id 与活动类型代码不一致。');
    end
    unknownCount = sum(ismember( ...
        row.active_type_codes, cfg.types.unknown));
    if row.unknown_component_count ~= unknownCount || ...
            row.unknown_present ~= (unknownCount > 0)
        throw_invalid_manifest('known/unknown 元数据与活动类型不一致。');
    end
end

function [targetSignal, meta] = build_optional_target( ...
        cfg, reference, noisePower_dB)
%BUILD_OPTIONAL_TARGET 构造零边界延迟并按目标 SNR 标定的真实目标回波。

    targetSignal = complex(zeros(1, cfg.derived.sample_length));
    meta = struct('present', false, 'delay_s', NaN, 'range_m', NaN, ...
        'target_power_dB', NaN, 'measured_power_dB', NaN);
    if ~cfg.radar.add_target
        return;
    end

    delaySamples = round(2 * cfg.radar.target_range_m / ...
        cfg.radar.c_mps * cfg.radar.fs_Hz);
    if ~isfinite(delaySamples) || delaySamples < 0 || ...
            delaySamples + cfg.derived.samples_per_pulse > ...
            cfg.derived.samples_per_pri
        error('JammingComposition:InvalidTargetDelay', ...
            ['真实目标的双程时延加脉宽必须完整位于一个 PRI 内，' ...
            '且不得循环回绕。']);
    end
    delaySamples = double(delaySamples);
    targetSignal(delaySamples + 1:end) = ...
        reference.train(1:end - delaySamples);
    targetPower_dB = noisePower_dB + cfg.radar.target_SNR_dB;
    [targetSignal, measuredPower_dB] = calibrate_component_power( ...
        targetSignal, targetPower_dB);
    realizedDelay_s = delaySamples / cfg.radar.fs_Hz;
    meta.present = true;
    meta.delay_s = realizedDelay_s;
    meta.range_m = cfg.radar.c_mps * realizedDelay_s / 2;
    meta.target_power_dB = targetPower_dB;
    meta.measured_power_dB = measuredPower_dB;
end

function seed = derive_substream_seed(sampleSeed, roleCode, identityCode)
%DERIVE_SUBSTREAM_SEED 用 uint64 盐值派生稳定且与组合阶数无关的种子。
%   roleCode=1 表示单分量生成器，identityCode 使用类型 ID；roleCode=2
%   表示接收机噪声，identityCode 固定为 0。映射在 uint32 模数下对不同
%   sampleSeed 保持一一对应，避免简单 sample_seed+1000*位置 的跨样本碰撞。

    modulus = uint64(4294967296);
    mixed = mod(uint64(sampleSeed) + ...
        uint64(roleCode) * uint64(2246822519) + ...
        uint64(identityCode) * uint64(2654435761), modulus);
    seed = double(mixed);
end

function power_dB = measure_power_dB(signal)
%MEASURE_POWER_DB 通过尺度归一化稳定计算复波形平均功率分贝。

    componentScale = max([max(abs(real(signal))), ...
        max(abs(imag(signal)))]);
    if ~isfinite(componentScale) || componentScale <= 0
        throw_invalid_power('波形必须具有有限正 RMS。');
    end
    normalized = signal / componentScale;
    rmsValue = sqrt(mean(abs(normalized) .^ 2));
    if ~isfinite(rmsValue) || rmsValue <= 0
        throw_invalid_power('波形必须具有有限正 RMS。');
    end
    power_dB = 20 * log10(componentScale) + 20 * log10(rmsValue);
end

function value = numeric_row(value, name)
%NUMERIC_ROW 校验有限实数数值向量并转换为 double 行向量。

    if ~isnumeric(value) || ~isreal(value) || ~isvector(value) || ...
            isempty(value) || ~all(isfinite(value(:)))
        throw_invalid_manifest([name ' 必须是非空有限实数数值向量。']);
    end
    value = reshape(double(value), 1, []);
end

function value = logical_row(value, name)
%LOGICAL_ROW 校验逻辑或0/1数值向量并转换为 logical 行向量。

    if islogical(value) && isvector(value) && ~isempty(value)
        value = reshape(value, 1, []);
    elseif isnumeric(value) && isreal(value) && isvector(value) && ...
            ~isempty(value) && all(ismember(value(:), [0, 1]))
        value = logical(reshape(value, 1, []));
    else
        throw_invalid_manifest([name ' 必须是非空 logical 或0/1向量。']);
    end
end

function value = logical_scalar(value, name)
%LOGICAL_SCALAR 校验一个逻辑或0/1标量。

    parsed = logical_row(value, name);
    if numel(parsed) ~= 1
        throw_invalid_manifest([name ' 必须是逻辑标量。']);
    end
    value = parsed(1);
end

function value = numeric_integer_scalar(value, lower, upper, name)
%NUMERIC_INTEGER_SCALAR 校验给定闭区间内的有限整数标量。

    if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ...
            ~isfinite(value) || value ~= floor(value) || ...
            value < lower || value > upper
        throw_invalid_manifest(sprintf( ...
            '%s 必须是 [%g,%g] 内的有限整数标量。', name, lower, upper));
    end
    value = double(value);
end

function value = positive_scalar(value, name)
%POSITIVE_SCALAR 校验有限正实数标量。

    if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ...
            ~isfinite(value) || value <= 0
        throw_invalid_manifest([name ' 必须是有限正实数标量。']);
    end
    value = double(value);
end

function value = text_scalar(value, name)
%TEXT_SCALAR 校验字符行向量或非缺失 string 标量。

    if isstring(value) && isscalar(value) && ~ismissing(value)
        value = char(value);
    elseif ~(ischar(value) && isrow(value))
        throw_invalid_manifest([name ' 必须是文本标量。']);
    end
    value = strtrim(value);
    if isempty(value)
        throw_invalid_manifest([name ' 不能为空。']);
    end
end

function values = text_list(value, name)
%TEXT_LIST 校验非空 cellstr 或 string 数组并转换为 cellstr 行向量。

    if isstring(value) && ~any(ismissing(value), 'all')
        values = cellstr(value(:).');
    elseif iscellstr(value)
        values = reshape(value, 1, []);
    else
        throw_invalid_manifest([name ' 必须是非空 cellstr 或 string 数组。']);
    end
    values = cellfun(@strtrim, values, 'UniformOutput', false);
    if isempty(values) || any(cellfun(@isempty, values))
        throw_invalid_manifest([name ' 不能为空或含空类型代码。']);
    end
end

function validate_observable_component_params(code, params, cfg, ...
        nominalAmplitude, effectiveAmplitude, offset_dB)
%VALIDATE_OBSERVABLE_COMPONENT_PARAMS 在编码前检查分量物理真值不变量。
%   该检查位于功率标定之后，因此 Traction.params.amplitude 已表示最终
%   有效幅度，其他含 JNR 类型的 params.JNR_dB 已表示最终有效 JNR。

    if ismember(code, {'FMB', 'AMA'})
        expectedFactor = params.interference_bandwidth_Hz / ...
            cfg.radar.bandwidth_Hz;
        if ~nearly_equal(params.bandwidth_factor, expectedFactor)
            throw_observable_truth_mismatch( ...
                [code ' 带宽因子与实际干扰带宽不一致。']);
        end
    elseif ismember(code, {'MFT', 'RangeFakeTarget'})
        delayNames = fieldnames(params);
        delayNames = delayNames(~cellfun('isempty', regexp( ...
            delayNames, '^delay_[0-9][0-9]_s$', 'once')));
        delays = cellfun(@(name) params.(name), delayNames);
        activeDelays = delays(isfinite(delays));
        if ~isnan(params.arrival_delay_s) || ...
                any(diff(activeDelays) <= 0)
            throw_observable_truth_mismatch( ...
                [code ' 最终目标时延不可辨识或存在碰撞。']);
        end
    elseif strcmp(code, 'Traction')
        expectedAmplitude = nominalAmplitude * 10 ^ (offset_dB / 20);
        if ~isfinite(expectedAmplitude) || expectedAmplitude <= 0 || ...
                ~nearly_equal(effectiveAmplitude, expectedAmplitude) || ...
                ~nearly_equal(params.amplitude, effectiveAmplitude)
            throw_observable_truth_mismatch( ...
                'Traction 最终幅度与相对功率偏置不一致。');
        end
    end
end

function matches = nearly_equal(actual, expected)
%NEARLY_EQUAL 使用尺度相关浮点容差比较两个有限标量。

    tolerance = 128 * eps(max([1, abs(actual), abs(expected)]));
    matches = isfinite(actual) && isfinite(expected) && ...
        abs(actual - expected) <= tolerance;
end

function throw_invalid_manifest(message)
%THROW_INVALID_MANIFEST 抛出统一 manifest 行错误标识。

    error('JammingComposition:InvalidManifestRow', '%s', message);
end

function throw_invalid_power(message)
%THROW_INVALID_POWER 抛出统一组合功率错误标识。

    error('JammingComposition:InvalidPower', '%s', message);
end

function throw_observable_truth_mismatch(message)
%THROW_OBSERVABLE_TRUTH_MISMATCH 抛出可观测真值不一致错误。

    error('JammingDataset:ObservableTruthMismatch', '%s', message);
end
