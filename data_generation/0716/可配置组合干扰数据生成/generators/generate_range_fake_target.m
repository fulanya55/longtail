function [signal, params] = generate_range_fake_target( ...
        cfg, stream, reference)
%GENERATE_RANGE_FAKE_TARGET 生成距离假目标干扰及定长容量真值。
%   [SIGNAL, PARAMS] = GENERATE_RANGE_FAKE_TARGET(CFG, STREAM, REFERENCE)
%   使用标量 RandStream STREAM 从配置整数范围采样目标数，在量化后的
%   相对时延样点槽中无放回选择目标，并采样相对幅度；公共到达时延量化
%   后加到每个相对时延，最终按可观测总时延升序重排目标对。每个目标由
%   相对幅度乘以循环移位后的参考脉冲串表示，距离严格按最终总时延的
%   range=c*delay/2 换算。CFG 必须已经标准化。
%
%   SIGNAL 为 1×sample_length complex double 干净干扰，不含接收机背景
%   噪声且不按 JNR_dB 缩放。PARAMS 中 delay 和 mean_delay 单位为 s，
%   range 单位为 m，amplitude 为无量纲相对幅度；固定输出四个 delay、
%   range 和 amplitude 槽，未使用槽为 NaN。arrival_delay_s 保留字段但
%   返回 NaN，表示公共到达时延已吸收到逐目标最终时延中，不独立监督。
%
%   示例：
%       reference = build_reference_signal(cfg);
%       stream = RandStream('mt19937ar', 'Seed', 9);
%       [signal, truth] = generate_range_fake_target( ...
%           cfg, stream, reference);

    capacity = 4;
    fs_Hz = cfg.radar.fs_Hz;
    validate_fixed_capacity(cfg.params.RangeFakeTarget, ...
        capacity, 'RangeFakeTarget');
    validate_delay_range(cfg.params.RangeFakeTarget.target_delay_s, ...
        cfg.radar.pri_s, 'RangeFakeTarget target_delay_s');
    validate_positive_range( ...
        cfg.params.RangeFakeTarget.target_relative_amplitude, ...
        'RangeFakeTarget target_relative_amplitude');
    validate_delay_range(cfg.params.common.arrival_delay_s, ...
        cfg.radar.pri_s, 'RangeFakeTarget arrival_delay_s');
    validate_delay_identifiability(cfg.params.RangeFakeTarget, ...
        cfg.params.common.arrival_delay_s, cfg.radar, 'RangeFakeTarget');
    JNR_dB = sample_range(cfg.params.common.JNR_dB, stream);
    numTargets = sample_integer( ...
        cfg.params.RangeFakeTarget.num_targets, stream);
    if numTargets > capacity
        throw_invalid_physical_parameter( ...
            'RangeFakeTarget 的 num_targets 不能超过固定真值容量 4。');
    end
    arrivalDelaySamples = round(sample_range( ...
        cfg.params.common.arrival_delay_s, stream) * fs_Hz);
    rawRelativeDelaySamples = sample_unique_delay_samples( ...
        cfg.params.RangeFakeTarget.target_delay_s, numTargets, fs_Hz, ...
        stream, 'RangeFakeTarget');
    rawAmplitudes = sample_vector( ...
        cfg.params.RangeFakeTarget.target_relative_amplitude, ...
        numTargets, stream);
    [relativeDelaySamples, order] = sort(rawRelativeDelaySamples);
    delaySamples = relativeDelaySamples + arrivalDelaySamples;
    delays_s = delaySamples / fs_Hz;
    amplitudes = rawAmplitudes(order);
    ranges_m = cfg.radar.c_mps * delays_s / 2;

    signal = complex(zeros(size(reference.train)));
    for targetIndex = 1:numTargets
        signal = signal + amplitudes(targetIndex) * circshift( ...
            reference.train, [0, delaySamples(targetIndex)]);
    end
    signal = complex(real(signal), imag(signal));

    params = struct('JNR_dB', JNR_dB, ...
        'num_targets', numTargets, 'mean_delay_s', mean(delays_s));
    for targetIndex = 1:capacity
        fieldName = sprintf('delay_%02d_s', targetIndex);
        if targetIndex <= numTargets
            params.(fieldName) = delays_s(targetIndex);
        else
            params.(fieldName) = NaN;
        end
    end
    for targetIndex = 1:capacity
        fieldName = sprintf('range_%02d_m', targetIndex);
        if targetIndex <= numTargets
            params.(fieldName) = ranges_m(targetIndex);
        else
            params.(fieldName) = NaN;
        end
    end
    for targetIndex = 1:capacity
        fieldName = sprintf('amplitude_%02d', targetIndex);
        if targetIndex <= numTargets
            params.(fieldName) = amplitudes(targetIndex);
        else
            params.(fieldName) = NaN;
        end
    end
    % 公共到达时延已体现在 delay_XX_s/range_XX_m 中，不再提供不可独立
    % 辨识的冗余真值；NaN 也将使后续参数掩码自动关闭该槽。
    params.arrival_delay_s = NaN;
end

function validate_delay_identifiability(params, arrivalRange_s, radar, name)
%VALIDATE_DELAY_IDENTIFIABILITY 防御性校验唯一离散槽和 PRI 内完整支持。

    fs_Hz = radar.fs_Hz;
    minimumDelaySamples = round(params.target_delay_s(1) * fs_Hz);
    maximumDelaySamples = round(params.target_delay_s(2) * fs_Hz);
    availableSlotCount = maximumDelaySamples - minimumDelaySamples + 1;
    if availableSlotCount < params.num_targets(2)
        throw_invalid_physical_parameter(sprintf( ...
            '%s 量化时延槽数 %d 少于 num_targets 上界 %d。', ...
            name, availableSlotCount, params.num_targets(2)));
    end

    worstEnd_s = params.target_delay_s(2) + arrivalRange_s(2) + ...
        radar.pulse_width_s;
    samplesPerPri = round(radar.pri_s * fs_Hz);
    samplesPerPulse = round(radar.pulse_width_s * fs_Hz);
    maximumArrivalSamples = round(arrivalRange_s(2) * fs_Hz);
    worstEndSamples = maximumDelaySamples + maximumArrivalSamples + ...
        samplesPerPulse;
    tolerance_s = 32 * eps(max([1, radar.pri_s, worstEnd_s]));
    if worstEnd_s > radar.pri_s + tolerance_s || ...
            worstEndSamples > samplesPerPri
        throw_invalid_physical_parameter([name ...
            ' 最大总时延加脉宽必须完整位于同一 PRI 内。']);
    end
end

function delaySamples = sample_unique_delay_samples( ...
        range_s, count, fs_Hz, stream, name)
%SAMPLE_UNIQUE_DELAY_SAMPLES 在量化整数样点槽中无放回均匀抽样。

    minimumSample = round(range_s(1) * fs_Hz);
    maximumSample = round(range_s(2) * fs_Hz);
    slotCount = maximumSample - minimumSample + 1;
    if slotCount < count
        throw_invalid_physical_parameter(sprintf( ...
            '%s 可用唯一样点时延槽不足，无法生成 %d 个目标。', name, count));
    end
    selectedOffsets = randperm(stream, slotCount, count) - 1;
    delaySamples = minimumSample + selectedOffsets;
end

function validate_fixed_capacity(params, capacity, name)
%VALIDATE_FIXED_CAPACITY 防御性校验固定真值容量和目标数范围。

    range = params.num_targets;
    validRange = isnumeric(range) && isreal(range) && ...
        isequal(size(range), [1, 2]) && all(isfinite(range)) && ...
        all(range > 0) && all(range == fix(range)) && ...
        range(1) <= range(2) && range(2) <= capacity;
    validCapacity = isnumeric(params.max_targets) && ...
        isreal(params.max_targets) && isscalar(params.max_targets) && ...
        isfinite(params.max_targets) && params.max_targets == capacity;
    if ~(validRange && validCapacity)
        throw_invalid_physical_parameter(sprintf( ...
            '%s 固定容量必须为 %d，num_targets 不能越界。', ...
            name, capacity));
    end
end

function validate_delay_range(range, pri_s, name)
%VALIDATE_DELAY_RANGE 防御性校验非负且小于 PRI 的时延。

    if ~is_valid_range(range) || range(1) < 0 || range(2) >= pri_s
        throw_invalid_physical_parameter( ...
            [name ' 必须满足下界>=0且上界<PRI。']);
    end
end

function validate_positive_range(range, name)
%VALIDATE_POSITIVE_RANGE 防御性校验严格正的连续范围。

    if ~is_valid_range(range) || range(1) <= 0
        throw_invalid_physical_parameter([name ' 必须为有限正数范围。']);
    end
end

function valid = is_valid_range(range)
%IS_VALID_RANGE 判断值是否为有限有序 1×2 实数范围。

    valid = isnumeric(range) && isreal(range) && ...
        isequal(size(range), [1, 2]) && all(isfinite(range)) && ...
        range(1) <= range(2);
end

function throw_invalid_physical_parameter(message)
%THROW_INVALID_PHYSICAL_PARAMETER 抛出统一生成器物理边界标识。

    error('JammingGenerator:InvalidPhysicalParameter', '%s', message);
end

function value = sample_range(range, stream)
%SAMPLE_RANGE 使用局部随机流在闭区间表示的连续范围内均匀采样。

    value = range(1) + (range(2) - range(1)) * rand(stream);
end

function value = sample_integer(range, stream)
%SAMPLE_INTEGER 使用局部随机流从含端点的整数范围内均匀采样。

    value = randi(stream, double(range), 1, 1);
end

function values = sample_vector(range, count, stream)
%SAMPLE_VECTOR 使用局部随机流均匀采样指定数量的连续参数。

    values = range(1) + (range(2) - range(1)) * ...
        rand(stream, 1, count);
end
