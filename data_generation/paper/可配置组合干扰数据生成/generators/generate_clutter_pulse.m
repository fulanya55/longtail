function [signal, params] = generate_clutter_pulse(cfg, stream, reference)
%GENERATE_CLUTTER_PULSE 生成杂乱脉冲干扰及三十二槽详细真值。
%   [SIGNAL, PARAMS] = GENERATE_CLUTTER_PULSE(CFG, STREAM, REFERENCE) 在每个
%   PRI 内生成 3--8 个随机门控零中心参考 LFM 片段。independent 模式逐 PRI
%   独立抽样布局，repeated 模式只抽样首 PRI 并在后续 PRI 精确复用脉内
%   起点、宽度、幅度、频偏和相位。脉冲允许重叠并按 complex double 相干
%   求和；占空比使用占用样点并集计算，重叠样点只计一次。
%
%   公共到达时延量化到整数样点，并在完整干扰记录形成后执行零边界平移。
%   每个脉冲的起点抽样预留该时延，因此平移后仍完整位于原 PRI 内，尾部
%   不会截断或循环回卷。所有随机量只消耗传入的标量 RandStream；输出不
%   含接收机背景噪声，也不在本函数内按 JNR 缩放。
%
%   PARAMS 包含九个汇总字段和五组三十二槽详细字段。start_XX_s 是加入
%   公共时延后的记录绝对时间，槽顺序固定为 PRI 主序、PRI 内起点升序；
%   未使用槽为 NaN。固定容量为每 PRI 8 个、每条记录 32 个。物理边界
%   错误使用 JammingGenerator:InvalidPhysicalParameter，非法模式使用
%   JammingGenerator:UnsupportedMode。

    capacityPerPri = 8;
    totalCapacity = 32;
    validate_stream(stream);
    clutter = cfg.params.ClutterPulse;
    mode = normalize_mode(clutter.mode);
    validate_clutter_config( ...
        clutter, cfg, capacityPerPri, totalCapacity);
    validate_real_range(cfg.params.common.JNR_dB, ...
        'ClutterPulse JNR_dB');
    validate_delay_range(cfg.params.common.arrival_delay_s, ...
        cfg.radar.pri_s, 'ClutterPulse arrival_delay_s');
    maximumWidthSamples = round( ...
        clutter.pulse_width_s(2) * cfg.radar.fs_Hz);
    maximumArrivalSamples = round( ...
        cfg.params.common.arrival_delay_s(2) * cfg.radar.fs_Hz);
    if maximumWidthSamples + maximumArrivalSamples > ...
            cfg.derived.samples_per_pri
        throw_invalid_physical_parameter( ...
            'ClutterPulse 最大脉宽与最大到达时延不能完整位于 PRI 内。');
    end

    centered = build_centered_reference_signal(cfg, reference);
    validate_frequency_support(clutter.frequency_offset_Hz, ...
        centered.support_Hz, cfg.radar.fs_Hz);

    fs_Hz = cfg.radar.fs_Hz;
    samplesPerPri = cfg.derived.samples_per_pri;
    sampleLength = cfg.derived.sample_length;
    arrivalDelaySamples = round(sample_range( ...
        cfg.params.common.arrival_delay_s, stream) * fs_Hz);
    arrivalDelay_s = arrivalDelaySamples / fs_Hz;
    JNR_dB = sample_range(cfg.params.common.JNR_dB, stream);

    startSamples = nan(1, totalCapacity);
    widthSamples = nan(1, totalCapacity);
    amplitudes = nan(1, totalCapacity);
    frequencies_Hz = nan(1, totalCapacity);
    phases_rad = nan(1, totalCapacity);
    cleanSignal = complex(zeros(1, sampleLength));
    cleanOccupied = false(1, sampleLength);
    slotIndex = 0;

    if strcmp(mode, 'repeated')
        repeatedCount = sample_integer(clutter.pulses_per_pri, stream);
        repeatedLayout = sample_layout(clutter, repeatedCount, ...
            arrivalDelaySamples, cfg, stream);
        modeCode = 1;
    else
        repeatedLayout = struct([]);
        modeCode = 0;
    end

    for priIndex = 1:cfg.radar.num_pulses
        if modeCode == 1
            layout = repeatedLayout;
        else
            pulseCount = sample_integer(clutter.pulses_per_pri, stream);
            layout = sample_layout(clutter, pulseCount, ...
                arrivalDelaySamples, cfg, stream);
        end
        priOffsetSamples = (priIndex - 1) * samplesPerPri;
        for localIndex = 1:numel(layout)
            slotIndex = slotIndex + 1;
            width = layout(localIndex).width_samples;
            localStart = layout(localIndex).start_samples;
            cleanStart = priOffsetSamples + localStart;
            indices = cleanStart + (1:width);
            localTime_s = (0:width - 1) / fs_Hz;
            gatedReference = centered.pulse(1:width) .* exp(1i * ( ...
                2 * pi * layout(localIndex).frequency_Hz * localTime_s + ...
                layout(localIndex).phase_rad));
            cleanSignal(indices) = cleanSignal(indices) + ...
                layout(localIndex).amplitude * gatedReference;
            cleanOccupied(indices) = true;

            startSamples(slotIndex) = cleanStart + arrivalDelaySamples;
            widthSamples(slotIndex) = width;
            amplitudes(slotIndex) = layout(localIndex).amplitude;
            frequencies_Hz(slotIndex) = layout(localIndex).frequency_Hz;
            phases_rad(slotIndex) = layout(localIndex).phase_rad;
        end
    end

    signal = delay_with_zeros(cleanSignal, arrivalDelaySamples);
    occupied = delay_logical_with_zeros( ...
        cleanOccupied, arrivalDelaySamples);
    signal = complex(real(signal), imag(signal));
    totalPulseCount = slotIndex;
    activeSlots = 1:totalPulseCount;

    params = struct( ...
        'JNR_dB', JNR_dB, ...
        'mode_code', modeCode, ...
        'total_pulse_count', totalPulseCount, ...
        'mean_width_s', mean(widthSamples(activeSlots)) / fs_Hz, ...
        'duty_cycle', nnz(occupied) / sampleLength, ...
        'mean_amplitude', mean(amplitudes(activeSlots)), ...
        'minimum_frequency_Hz', min(frequencies_Hz(activeSlots)), ...
        'maximum_frequency_Hz', max(frequencies_Hz(activeSlots)), ...
        'arrival_delay_s', arrivalDelay_s);
    for index = 1:totalCapacity
        params.(sprintf('start_%02d_s', index)) = ...
            startSamples(index) / fs_Hz;
        params.(sprintf('width_%02d_s', index)) = ...
            widthSamples(index) / fs_Hz;
        params.(sprintf('amplitude_%02d', index)) = amplitudes(index);
        params.(sprintf('frequency_%02d_Hz', index)) = ...
            frequencies_Hz(index);
        params.(sprintf('phase_%02d_rad', index)) = phases_rad(index);
    end
end

function layout = sample_layout( ...
        clutter, count, arrivalDelaySamples, cfg, stream)
%SAMPLE_LAYOUT 抽样一个 PRI 内的布局，并按量化起点升序返回。

    fs_Hz = cfg.radar.fs_Hz;
    samplesPerPri = cfg.derived.samples_per_pri;
    layout = repmat(struct( ...
        'start_samples', 0, ...
        'width_samples', 0, ...
        'amplitude', 0, ...
        'frequency_Hz', 0, ...
        'phase_rad', 0), 1, count);
    for pulseIndex = 1:count
        requestedWidth_s = sample_range(clutter.pulse_width_s, stream);
        width = max(1, round(requestedWidth_s * fs_Hz));
        latestStart = samplesPerPri - width - arrivalDelaySamples;
        if width > cfg.derived.samples_per_pulse || latestStart < 0
            throw_invalid_physical_parameter( ...
                'ClutterPulse 脉宽与公共到达时延无法完整位于 PRI 内。');
        end
        layout(pulseIndex).start_samples = ...
            randi(stream, latestStart + 1, 1, 1) - 1;
        layout(pulseIndex).width_samples = width;
        layout(pulseIndex).amplitude = sample_range( ...
            clutter.relative_amplitude, stream);
        layout(pulseIndex).frequency_Hz = sample_range( ...
            clutter.frequency_offset_Hz, stream);
        layout(pulseIndex).phase_rad = 2 * pi * rand(stream);
    end
    [~, order] = sort([layout.start_samples]);
    layout = layout(order);
end

function validate_clutter_config( ...
        clutter, cfg, capacityPerPri, totalCapacity)
%VALIDATE_CLUTTER_CONFIG 防御性校验计数、固定容量、脉宽和幅度。

    required = {'pulses_per_pri', 'pulse_width_s', ...
        'relative_amplitude', 'frequency_offset_Hz', 'mode', ...
        'max_pulses_per_pri', 'max_total_pulses'};
    if ~isstruct(clutter) || ~isscalar(clutter) || ...
            ~all(isfield(clutter, required))
        throw_invalid_physical_parameter( ...
            'ClutterPulse 参数结构缺少必需字段。');
    end
    validCount = is_integer_range(clutter.pulses_per_pri) && ...
        clutter.pulses_per_pri(1) >= 3 && ...
        clutter.pulses_per_pri(2) <= capacityPerPri;
    validPerPriCapacity = is_exact_integer( ...
        clutter.max_pulses_per_pri, capacityPerPri);
    validTotalCapacity = is_exact_integer( ...
        clutter.max_total_pulses, totalCapacity);
    requiredSlots = cfg.radar.num_pulses * clutter.pulses_per_pri(2);
    if ~(validCount && validPerPriCapacity && validTotalCapacity) || ...
            requiredSlots > totalCapacity
        throw_invalid_physical_parameter( ...
            'ClutterPulse 必须满足每 PRI 3--8 个、容量 8/32 和总槽约束。');
    end
    validate_positive_range(clutter.pulse_width_s, ...
        'ClutterPulse pulse_width_s');
    maximumWidthSamples = round( ...
        clutter.pulse_width_s(2) * cfg.radar.fs_Hz);
    if maximumWidthSamples < 1 || ...
            maximumWidthSamples > cfg.derived.samples_per_pulse
        throw_invalid_physical_parameter( ...
            'ClutterPulse 最大脉宽必须至少一个样点且不超过参考脉宽。');
    end
    validate_positive_range(clutter.relative_amplitude, ...
        'ClutterPulse relative_amplitude');
    validate_real_range(clutter.frequency_offset_Hz, ...
        'ClutterPulse frequency_offset_Hz');
end

function validate_frequency_support(offsetRange_Hz, support_Hz, fs_Hz)
%VALIDATE_FREQUENCY_SUPPORT 保证全部可抽频偏的完整参考支撑严格无混叠。

    guard_Hz = fs_Hz / 2 - max(eps(fs_Hz / 2), fs_Hz * 1e-12);
    minimum_Hz = offsetRange_Hz(1) + support_Hz(1);
    maximum_Hz = offsetRange_Hz(2) + support_Hz(2);
    if minimum_Hz <= -guard_Hz || maximum_Hz >= guard_Hz
        throw_invalid_physical_parameter( ...
            'ClutterPulse 频偏范围使参考支撑触及或越过 Nyquist guard。');
    end
end

function mode = normalize_mode(value)
%NORMALIZE_MODE 规范 independent/repeated 模式文本。

    if isstring(value) && isscalar(value) && ~ismissing(value)
        value = char(value);
    elseif ~(ischar(value) && isrow(value))
        throw_unsupported_mode();
    end
    mode = lower(strtrim(value));
    if ~ismember(mode, {'independent', 'repeated'})
        throw_unsupported_mode();
    end
end

function validate_stream(stream)
%VALIDATE_STREAM 要求生成器只使用调用者提供的标量随机流。

    if ~(isa(stream, 'RandStream') && isscalar(stream))
        error('JammingGenerator:InvalidStream', ...
            'stream 必须是标量 RandStream。');
    end
end

function validate_delay_range(range, pri_s, name)
%VALIDATE_DELAY_RANGE 校验非负且严格小于 PRI 的到达时延范围。

    if ~is_valid_range(range) || range(1) < 0 || range(2) >= pri_s
        throw_invalid_physical_parameter( ...
            [name ' 必须满足下界>=0且上界<PRI。']);
    end
end

function validate_positive_range(range, name)
%VALIDATE_POSITIVE_RANGE 校验严格正的有限有序范围。

    if ~is_valid_range(range) || range(1) <= 0
        throw_invalid_physical_parameter([name ' 必须为有限正数范围。']);
    end
end

function validate_real_range(range, name)
%VALIDATE_REAL_RANGE 校验允许负数的有限有序范围。

    if ~is_valid_range(range)
        throw_invalid_physical_parameter([name ' 必须为有限有序范围。']);
    end
end

function valid = is_valid_range(range)
%IS_VALID_RANGE 判断值是否为有限有序 1×2 实数范围。

    valid = isnumeric(range) && isreal(range) && ...
        isequal(size(range), [1, 2]) && all(isfinite(range)) && ...
        range(1) <= range(2);
end

function valid = is_integer_range(range)
%IS_INTEGER_RANGE 判断值是否为有限递增正整数范围。

    valid = is_valid_range(range) && all(range > 0) && ...
        all(range == fix(range));
end

function valid = is_exact_integer(value, expected)
%IS_EXACT_INTEGER 判断固定容量是否等于指定有限正整数。

    valid = isnumeric(value) && isreal(value) && isscalar(value) && ...
        isfinite(value) && value == expected;
end

function value = sample_range(range, stream)
%SAMPLE_RANGE 使用局部随机流在连续范围内均匀抽样。

    value = range(1) + (range(2) - range(1)) * rand(stream);
end

function value = sample_integer(range, stream)
%SAMPLE_INTEGER 使用局部随机流从含端点整数范围抽样。

    value = randi(stream, double(range), 1, 1);
end

function delayed = delay_with_zeros(signal, delaySamples)
%DELAY_WITH_ZEROS 对复数记录执行非循环整数样点时延。

    delayed = complex(zeros(size(signal)));
    if delaySamples < numel(signal)
        delayed(delaySamples + 1:end) = signal(1:end - delaySamples);
    end
end

function delayed = delay_logical_with_zeros(mask, delaySamples)
%DELAY_LOGICAL_WITH_ZEROS 对占用掩码执行同一零边界时延。

    delayed = false(size(mask));
    if delaySamples < numel(mask)
        delayed(delaySamples + 1:end) = mask(1:end - delaySamples);
    end
end

function throw_invalid_physical_parameter(message)
%THROW_INVALID_PHYSICAL_PARAMETER 抛出统一物理边界错误。

    error('JammingGenerator:InvalidPhysicalParameter', '%s', message);
end

function throw_unsupported_mode()
%THROW_UNSUPPORTED_MODE 抛出稳定的布局模式错误。

    error('JammingGenerator:UnsupportedMode', ...
        'ClutterPulse mode 必须为 independent 或 repeated。');
end
