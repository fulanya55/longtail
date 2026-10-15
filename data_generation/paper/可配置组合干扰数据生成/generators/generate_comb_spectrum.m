function [signal, params] = generate_comb_spectrum(cfg, stream, reference)
%GENERATE_COMB_SPECTRUM 生成梳状谱干扰及八齿定长物理真值。
%   [SIGNAL, PARAMS] = GENERATE_COMB_SPECTRUM(CFG, STREAM, REFERENCE) 先把
%   现有参考 LFM 搬移到零中心复基带，再生成若干具有等间隔频移的相干
%   LFM 副本并求和。STREAM 必须是标量 RandStream；本函数不使用全局随机
%   状态。REFERENCE 由 BUILD_REFERENCE_SIGNAL 构造，输出 SIGNAL 为不含
%   接收机噪声、未按 JNR 缩放的 1×sample_length complex double 行向量。
%
%   num_teeth 为 3--8 的整数；spacing_Hz 和 center_offset_Hz 共同决定每
%   个齿的零中心基带偏移。equal 模式把所有活动齿幅度固定为 1，random
%   模式从 amplitude 范围独立抽样；每齿相位独立均匀分布在 [0,2*pi)。
%   生成前根据参考离散频率支撑解析收缩可行齿间隔和中心区间，生成后再次
%   检查所有平移支撑严格位于 Nyquist guard 内，绝不静默混叠。
%
%   公共到达时延量化到整数样点后使用零边界平移，记录尾部不会循环回卷。
%   PARAMS 保存 JNR、齿数、实际齿间隔、中心、模式代码、实际时延以及
%   frequency_01_Hz...08、amplitude_01...08、phase_01_rad...08；未使用
%   槽固定为 NaN。物理定义域错误统一使用
%   JammingGenerator:InvalidPhysicalParameter，非法模式使用
%   JammingGenerator:UnsupportedMode。

    capacity = 8;
    validate_stream(stream);
    comb = cfg.params.CombSpectrum;
    validate_comb_config(comb, capacity);
    validate_real_range(cfg.params.common.JNR_dB, ...
        'CombSpectrum JNR_dB');
    validate_delay_range(cfg.params.common.arrival_delay_s, ...
        cfg.radar.pri_s, 'CombSpectrum arrival_delay_s');

    centered = build_centered_reference_signal(cfg, reference);
    fs_Hz = cfg.radar.fs_Hz;
    guard_Hz = fs_Hz / 2 - max( ...
        eps(fs_Hz / 2), fs_Hz * 1e-12);
    mode = normalize_mode(comb.amplitude_mode);

    JNR_dB = sample_range(cfg.params.common.JNR_dB, stream);
    numTeeth = sample_integer(comb.num_teeth, stream);
    [spacing_Hz, centerOffset_Hz] = sample_feasible_comb( ...
        comb, numTeeth, centered.support_Hz, guard_Hz, stream);
    toothIndices = (1:numTeeth) - (numTeeth + 1) / 2;
    frequencies_Hz = centerOffset_Hz + toothIndices * spacing_Hz;
    validate_realized_support( ...
        frequencies_Hz, centered.support_Hz, guard_Hz);

    if strcmp(mode, 'equal')
        amplitudes = ones(1, numTeeth);
        modeCode = 0;
    else
        amplitudes = sample_vector(comb.amplitude, numTeeth, stream);
        modeCode = 1;
    end
    phases_rad = 2 * pi * rand(stream, 1, numTeeth);

    signal = complex(zeros(size(centered.train)));
    for toothIndex = 1:numTeeth
        modulation = exp(1i * (2 * pi * frequencies_Hz(toothIndex) * ...
            reference.time_train_s + phases_rad(toothIndex)));
        signal = signal + amplitudes(toothIndex) * ...
            centered.train .* modulation;
    end

    arrivalDelaySamples = round(sample_range( ...
        cfg.params.common.arrival_delay_s, stream) * fs_Hz);
    validate_delayed_support(arrivalDelaySamples, cfg);
    signal = delay_with_zeros(signal, arrivalDelaySamples);
    signal = complex(real(signal), imag(signal));
    arrivalDelay_s = arrivalDelaySamples / fs_Hz;

    params = struct( ...
        'JNR_dB', JNR_dB, ...
        'num_teeth', numTeeth, ...
        'spacing_Hz', spacing_Hz, ...
        'center_offset_Hz', centerOffset_Hz, ...
        'amplitude_mode_code', modeCode, ...
        'arrival_delay_s', arrivalDelay_s);
    for toothIndex = 1:capacity
        if toothIndex <= numTeeth
            frequencyValue = frequencies_Hz(toothIndex);
            amplitudeValue = amplitudes(toothIndex);
            phaseValue = phases_rad(toothIndex);
        else
            frequencyValue = NaN;
            amplitudeValue = NaN;
            phaseValue = NaN;
        end
        params.(sprintf('frequency_%02d_Hz', toothIndex)) = ...
            frequencyValue;
        params.(sprintf('amplitude_%02d', toothIndex)) = amplitudeValue;
        params.(sprintf('phase_%02d_rad', toothIndex)) = phaseValue;
    end
end

function [spacing_Hz, centerOffset_Hz] = sample_feasible_comb( ...
        comb, numTeeth, support_Hz, guard_Hz, stream)
%SAMPLE_FEASIBLE_COMB 解析求出存在严格无混叠中心值的齿间隔区间。

    outerIndex = (numTeeth - 1) / 2;
    spacingLower_Hz = comb.spacing_Hz(1);
    spacingUpper_Hz = min([comb.spacing_Hz(2), ...
        (comb.center_offset_Hz(2) + guard_Hz + support_Hz(1)) / ...
        outerIndex, ...
        (guard_Hz - support_Hz(2) - comb.center_offset_Hz(1)) / ...
        outerIndex]);
    if ~isfinite(spacingUpper_Hz) || ...
            spacingUpper_Hz < spacingLower_Hz
        throw_invalid_physical_parameter( ...
            'CombSpectrum 没有可行的齿间隔和中心偏移组合。');
    end
    spacing_Hz = sample_range( ...
        [spacingLower_Hz, spacingUpper_Hz], stream);
    lowerCenter_Hz = max(comb.center_offset_Hz(1), ...
        -guard_Hz - support_Hz(1) + outerIndex * spacing_Hz);
    upperCenter_Hz = min(comb.center_offset_Hz(2), ...
        guard_Hz - support_Hz(2) - outerIndex * spacing_Hz);
    if ~isfinite(lowerCenter_Hz) || ~isfinite(upperCenter_Hz) || ...
            lowerCenter_Hz > upperCenter_Hz
        throw_invalid_physical_parameter( ...
            'CombSpectrum 没有可行的中心偏移。');
    end
    centerOffset_Hz = sample_range( ...
        [lowerCenter_Hz, upperCenter_Hz], stream);
end

function validate_comb_config(comb, capacity)
%VALIDATE_COMB_CONFIG 防御性校验齿数、容量和连续参数范围。

    required = {'num_teeth', 'spacing_Hz', 'center_offset_Hz', ...
        'amplitude', 'amplitude_mode', 'max_teeth'};
    if ~isstruct(comb) || ~isscalar(comb) || ...
            ~all(isfield(comb, required))
        throw_invalid_physical_parameter( ...
            'CombSpectrum 参数结构缺少必需字段。');
    end
    validCount = is_integer_range(comb.num_teeth) && ...
        comb.num_teeth(1) >= 3 && comb.num_teeth(2) <= capacity;
    validCapacity = isnumeric(comb.max_teeth) && ...
        isreal(comb.max_teeth) && isscalar(comb.max_teeth) && ...
        isfinite(comb.max_teeth) && comb.max_teeth == capacity;
    if ~(validCount && validCapacity)
        throw_invalid_physical_parameter( ...
            'CombSpectrum 齿数必须为 3--8，固定容量必须为 8。');
    end
    validate_positive_range(comb.spacing_Hz, ...
        'CombSpectrum spacing_Hz');
    validate_real_range(comb.center_offset_Hz, ...
        'CombSpectrum center_offset_Hz');
    validate_positive_range(comb.amplitude, ...
        'CombSpectrum amplitude');
end

function validate_realized_support(frequencies_Hz, support_Hz, guard_Hz)
%VALIDATE_REALIZED_SUPPORT 检查最外齿的完整参考支撑严格位于 guard 内。

    minimum_Hz = min(frequencies_Hz) + support_Hz(1);
    maximum_Hz = max(frequencies_Hz) + support_Hz(2);
    if minimum_Hz <= -guard_Hz || maximum_Hz >= guard_Hz
        throw_invalid_physical_parameter( ...
            'CombSpectrum 实际齿支撑触及或越过 Nyquist guard。');
    end
end

function validate_delayed_support(delaySamples, cfg)
%VALIDATE_DELAYED_SUPPORT 保证最后一个参考脉冲经时延后仍完整位于记录内。

    lastSupportSample = (cfg.radar.num_pulses - 1) * ...
        cfg.derived.samples_per_pri + cfg.derived.samples_per_pulse;
    if delaySamples < 0 || ...
            lastSupportSample + delaySamples > cfg.derived.sample_length
        throw_invalid_physical_parameter( ...
            'CombSpectrum 公共到达时延会截断记录末端有效支撑。');
    end
end

function mode = normalize_mode(value)
%NORMALIZE_MODE 规范幅度模式并返回小写字符代码。

    if isstring(value) && isscalar(value) && ~ismissing(value)
        value = char(value);
    elseif ~(ischar(value) && isrow(value))
        throw_unsupported_mode();
    end
    mode = lower(strtrim(value));
    if ~ismember(mode, {'equal', 'random'})
        throw_unsupported_mode();
    end
end

function validate_stream(stream)
%VALIDATE_STREAM 要求所有随机量只消耗调用者提供的标量随机流。

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

function value = sample_range(range, stream)
%SAMPLE_RANGE 使用局部随机流在闭区间表示的连续范围内均匀抽样。

    value = range(1) + (range(2) - range(1)) * rand(stream);
end

function value = sample_integer(range, stream)
%SAMPLE_INTEGER 使用局部随机流在含端点整数范围内均匀抽样。

    value = randi(stream, double(range), 1, 1);
end

function values = sample_vector(range, count, stream)
%SAMPLE_VECTOR 独立均匀抽样指定数量的连续参数。

    values = range(1) + (range(2) - range(1)) * ...
        rand(stream, 1, count);
end

function delayed = delay_with_zeros(signal, delaySamples)
%DELAY_WITH_ZEROS 对完整记录施加非循环整数样点时延。

    delayed = complex(zeros(size(signal)));
    if delaySamples < numel(signal)
        delayed(delaySamples + 1:end) = signal(1:end - delaySamples);
    end
end

function throw_invalid_physical_parameter(message)
%THROW_INVALID_PHYSICAL_PARAMETER 抛出统一物理边界错误。

    error('JammingGenerator:InvalidPhysicalParameter', '%s', message);
end

function throw_unsupported_mode()
%THROW_UNSUPPORTED_MODE 抛出稳定的幅度模式错误。

    error('JammingGenerator:UnsupportedMode', ...
        'CombSpectrum amplitude_mode 必须为 equal 或 random。');
end
