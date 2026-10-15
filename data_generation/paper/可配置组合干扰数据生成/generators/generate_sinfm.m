function [signal, params] = generate_sinfm(cfg, stream, reference)
%GENERATE_SINFM 生成正弦调频（SinFM）干扰及物理真值。
%   [SIGNAL, PARAMS] = GENERATE_SINFM(CFG, STREAM, REFERENCE) 使用标量
%   RandStream STREAM 采样中心频率、频偏和调制频率，在参考训练时间轴
%   上构造 f0+deviation*cos(2*pi*fm*t) 的瞬时频率，并逐样点积分得到
%   单位幅度复包络。CFG 必须已经标准化；REFERENCE 的时间单位为秒。
%
%   SIGNAL 为 1×sample_length complex double 干净干扰，不含接收机背景
%   噪声且不按 JNR_dB 缩放。PARAMS 中三个频率量单位为 Hz，调制周期和
%   到达时延单位为 s，JNR_dB 单位为 dB。
%
%   示例：
%       reference = build_reference_signal(cfg);
%       stream = RandStream('mt19937ar', 'Seed', 4);
%       [signal, truth] = generate_sinfm(cfg, stream, reference);

    fs_Hz = cfg.radar.fs_Hz;
    nyquistGuard_Hz = fs_Hz / 2 - max( ...
        eps(fs_Hz / 2), fs_Hz * 1e-12);
    validate_nonnegative_range(cfg.params.SinFM.frequency_deviation_Hz, ...
        'SinFM frequency_deviation_Hz');
    validate_positive_range(cfg.params.SinFM.modulation_frequency_Hz, ...
        'SinFM modulation_frequency_Hz');
    validate_delay_range(cfg.params.common.arrival_delay_s, ...
        cfg.radar.pri_s, 'SinFM arrival_delay_s');
    JNR_dB = sample_range(cfg.params.common.JNR_dB, stream);
    deviationRange = cfg.params.SinFM.frequency_deviation_Hz;
    centerGuard_Hz = nyquistGuard_Hz - deviationRange(1);
    centerFrequency_Hz = sample_center_frequency( ...
        cfg.params.SinFM.center_frequency_Hz, centerGuard_Hz, stream);
    supportedUpper_Hz = min(deviationRange(2), ...
        nyquistGuard_Hz - abs(centerFrequency_Hz));
    if deviationRange(1) > supportedUpper_Hz
        throw_invalid_physical_parameter( ...
            ['SinFM frequency_deviation_Hz 下界超过当前中心频率的 ' ...
            'Nyquist 剩余支持。']);
    end
    frequencyDeviation_Hz = sample_range( ...
        [deviationRange(1), supportedUpper_Hz], stream);
    modulationFrequency_Hz = sample_range( ...
        cfg.params.SinFM.modulation_frequency_Hz, stream);
    arrivalDelaySamples = round(sample_range( ...
        cfg.params.common.arrival_delay_s, stream) * fs_Hz);
    arrivalDelay_s = arrivalDelaySamples / fs_Hz;
    modulationPeriod_s = 1 / modulationFrequency_Hz;

    instantaneousFrequency_Hz = centerFrequency_Hz + ...
        frequencyDeviation_Hz * cos(2 * pi * ...
        modulationFrequency_Hz * reference.time_train_s);
    phase = 2 * pi * cumsum(instantaneousFrequency_Hz) / fs_Hz;
    signal = exp(1i * phase);
    signal = apply_arrival_delay(signal, arrivalDelaySamples);
    signal = complex(real(signal), imag(signal));

    params = struct('JNR_dB', JNR_dB, ...
        'center_frequency_Hz', centerFrequency_Hz, ...
        'frequency_deviation_Hz', frequencyDeviation_Hz, ...
        'modulation_frequency_Hz', modulationFrequency_Hz, ...
        'modulation_period_s', modulationPeriod_s, ...
        'arrival_delay_s', arrivalDelay_s);
end

function centerFrequency_Hz = sample_center_frequency(range, guard_Hz, stream)
%SAMPLE_CENTER_FREQUENCY 在具有最小频偏余量的严格域内采样。

    if ~is_valid_range(range)
        throw_invalid_physical_parameter('SinFM center_frequency_Hz 范围非法。');
    end
    supportedRange = [max(range(1), -guard_Hz), ...
        min(range(2), guard_Hz)];
    if supportedRange(1) > supportedRange(2)
        throw_invalid_physical_parameter( ...
            'SinFM center_frequency_Hz 必须位于 Nyquist 区间内。');
    end
    centerFrequency_Hz = sample_range(supportedRange, stream);
end

function validate_positive_range(range, name)
%VALIDATE_POSITIVE_RANGE 防御性校验严格正的连续范围。

    if ~is_valid_range(range) || range(1) <= 0
        throw_invalid_physical_parameter([name ' 必须为有限正数范围。']);
    end
end

function validate_nonnegative_range(range, name)
%VALIDATE_NONNEGATIVE_RANGE 防御性校验非负的连续范围。

    if ~is_valid_range(range) || range(1) < 0
        throw_invalid_physical_parameter([name ' 必须为有限非负范围。']);
    end
end

function validate_delay_range(range, pri_s, name)
%VALIDATE_DELAY_RANGE 防御性校验非负且小于 PRI 的到达时延。

    if ~is_valid_range(range) || range(1) < 0 || range(2) >= pri_s
        throw_invalid_physical_parameter( ...
            [name ' 必须满足下界>=0且上界<PRI。']);
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

function signal = apply_arrival_delay(signal, delaySamples)
%APPLY_ARRIVAL_DELAY 按已量化整数样点对完整波形循环移位一次。

    signal = circshift(signal, [0, delaySamples]);
end
