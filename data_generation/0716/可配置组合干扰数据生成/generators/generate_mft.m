function [signal, params] = generate_mft(cfg, stream, reference)
%GENERATE_MFT 生成多假目标（MFT）干扰及定长容量真值。
%   [SIGNAL, PARAMS] = GENERATE_MFT(CFG, STREAM, REFERENCE) 使用标量
%   RandStream STREAM 从配置整数范围采样目标数，在量化后的相对时延
%   样点槽中无放回选择目标，并采样相对幅度；公共到达时延量化后加到
%   每个目标相对时延，最终按可观测总时延升序重排目标对。每个目标由
%   相对幅度乘以循环移位后的参考脉冲串表示。CFG 必须已经标准化。
%
%   SIGNAL 为 1×sample_length complex double 干净干扰，不含接收机背景
%   噪声且不按 JNR_dB 缩放。PARAMS 中目标数为计数，delay、mean_delay
%   和 mean_spacing 单位为 s，amplitude 为无量纲相对幅度；固定输出
%   delay_01_s...delay_08_s 与 amplitude_01...amplitude_08，未使用槽为
%   NaN。delay 字段记录“相对目标时延+公共到达时延”的最终可观测总
%   时延；arrival_delay_s 返回 NaN，表示该公共量已吸收到逐目标时延，
%   不应再作为独立监督目标。单目标时 mean_spacing_s 为 NaN。
%
%   示例：
%       reference = build_reference_signal(cfg);
%       stream = RandStream('mt19937ar', 'Seed', 6);
%       [signal, truth] = generate_mft(cfg, stream, reference);

    capacity = 8;
    fs_Hz = cfg.radar.fs_Hz;
    validate_fixed_capacity(cfg.params.MFT, capacity, 'MFT');
    validate_delay_range(cfg.params.MFT.target_delay_s, ...
        cfg.radar.pri_s, 'MFT target_delay_s');
    validate_positive_range(cfg.params.MFT.target_relative_amplitude, ...
        'MFT target_relative_amplitude');
    validate_delay_range(cfg.params.common.arrival_delay_s, ...
        cfg.radar.pri_s, 'MFT arrival_delay_s');
    validate_delay_identifiability(cfg.params.MFT, ...
        cfg.params.common.arrival_delay_s, cfg.radar, 'MFT');
    JNR_dB = sample_range(cfg.params.common.JNR_dB, stream);
    numTargets = sample_integer(cfg.params.MFT.num_targets, stream);
    if numTargets > capacity
        throw_invalid_physical_parameter( ...
            'MFT 的 num_targets 不能超过固定真值容量 8。');
    end
    arrivalDelaySamples = round(sample_range( ...
        cfg.params.common.arrival_delay_s, stream) * fs_Hz);
    rawRelativeDelaySamples = sample_unique_delay_samples( ...
        cfg.params.MFT.target_delay_s, numTargets, fs_Hz, stream, 'MFT');
    rawAmplitudes = sample_vector( ...
        cfg.params.MFT.target_relative_amplitude, numTargets, stream);
    [relativeDelaySamples, order] = sort(rawRelativeDelaySamples);
    delaySamples = relativeDelaySamples + arrivalDelaySamples;
    delays_s = delaySamples / fs_Hz;
    amplitudes = rawAmplitudes(order);

    signal = complex(zeros(size(reference.train)));
    for targetIndex = 1:numTargets
        signal = signal + amplitudes(targetIndex) * circshift( ...
            reference.train, [0, delaySamples(targetIndex)]);
    end
    signal = complex(real(signal), imag(signal));

    meanDelay_s = mean(delays_s);
    if numTargets == 1
        meanSpacing_s = NaN;
    else
        meanSpacing_s = mean(diff(delays_s));
    end
    params = struct('JNR_dB', JNR_dB, ...
        'num_targets', numTargets, 'mean_delay_s', meanDelay_s, ...
        'mean_spacing_s', meanSpacing_s);
    for targetIndex = 1:capacity
        fieldName = sprintf('delay_%02d_s', targetIndex);
        if targetIndex <= numTargets
            params.(fieldName) = delays_s(targetIndex);
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
    % 公共到达时延已包含在每个 delay_XX_s 中，单独输出会形成不可辨识的
    % 冗余监督，因此保留注册字段但明确置为 NaN。
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
