function [signal, params] = generate_vft(cfg, stream, reference)
%GENERATE_VFT 生成速度假目标（VFT）干扰及物理真值。
%   [SIGNAL, PARAMS] = GENERATE_VFT(CFG, STREAM, REFERENCE) 使用标量
%   RandStream STREAM 采样多普勒频率 fd，将参考脉冲串逐样点乘以
%   exp(1i*2*pi*fd*t)，并按 v=fd*c/(2*fc) 换算径向速度。CFG 必须已经
%   标准化；REFERENCE 提供参考脉冲串和以秒计的训练时间轴。
%
%   SIGNAL 为 1×sample_length complex double 干净干扰，不含接收机背景
%   噪声且不按 JNR_dB 缩放。PARAMS 中 doppler_Hz 单位为 Hz，velocity_mps
%   单位为 m/s，arrival_delay_s 单位为 s，JNR_dB 单位为 dB。
%
%   示例：
%       reference = build_reference_signal(cfg);
%       stream = RandStream('mt19937ar', 'Seed', 7);
%       [signal, truth] = generate_vft(cfg, stream, reference);

    fs_Hz = cfg.radar.fs_Hz;
    nyquistGuard_Hz = fs_Hz / 2 - max( ...
        eps(fs_Hz / 2), fs_Hz * 1e-12);
    validate_delay_range(cfg.params.common.arrival_delay_s, ...
        cfg.radar.pri_s, 'VFT arrival_delay_s');
    JNR_dB = sample_range(cfg.params.common.JNR_dB, stream);
    [referenceMinimum_Hz, referenceMaximum_Hz] = ...
        reference_lfm_frequency_support(cfg.derived.samples_per_pulse, ...
        cfg.derived.chirp_rate_Hz_per_s, fs_Hz);
    supportedDopplerRange_Hz = [ ...
        -nyquistGuard_Hz - referenceMinimum_Hz, ...
        nyquistGuard_Hz - referenceMaximum_Hz];
    doppler_Hz = sample_supported_frequency( ...
        cfg.params.VFT.doppler_Hz, supportedDopplerRange_Hz, stream);
    arrivalDelaySamples = round(sample_range( ...
        cfg.params.common.arrival_delay_s, stream) * fs_Hz);
    arrivalDelay_s = arrivalDelaySamples / fs_Hz;
    velocity_mps = doppler_Hz * cfg.radar.c_mps / ...
        (2 * cfg.radar.carrier_Hz);

    signal = reference.train .* exp(1i * 2 * pi * ...
        doppler_Hz * reference.time_train_s);
    signal = apply_arrival_delay(signal, arrivalDelaySamples);
    signal = complex(real(signal), imag(signal));

    params = struct('JNR_dB', JNR_dB, ...
        'doppler_Hz', doppler_Hz, 'velocity_mps', velocity_mps, ...
        'arrival_delay_s', arrivalDelay_s);
end

function validate_delay_range(range, pri_s, name)
%VALIDATE_DELAY_RANGE 防御性校验非负且小于 PRI 的到达时延。

    validRange = isnumeric(range) && isreal(range) && ...
        isequal(size(range), [1, 2]) && all(isfinite(range)) && ...
        range(1) <= range(2);
    if ~validRange || range(1) < 0 || range(2) >= pri_s
        error('JammingGenerator:InvalidPhysicalParameter', ...
            '%s 必须满足下界>=0且上界<PRI。', name);
    end
end

function value = sample_range(range, stream)
%SAMPLE_RANGE 使用局部随机流在闭区间表示的连续范围内均匀采样。

    value = range(1) + (range(2) - range(1)) * rand(stream);
end

function [minimum_Hz, maximum_Hz] = reference_lfm_frequency_support( ...
        samplesPerPulse, chirpRate_Hz_per_s, fs_Hz)
%REFERENCE_LFM_FREQUENCY_SUPPORT 返回相邻样点相位差对应的 LFM 支持。

    if samplesPerPulse < 2
        minimum_Hz = 0;
        maximum_Hz = 0;
        return;
    end
    minimum_Hz = chirpRate_Hz_per_s / (2 * fs_Hz);
    maximum_Hz = chirpRate_Hz_per_s * (2 * samplesPerPulse - 3) / ...
        (2 * fs_Hz);
end

function value = sample_supported_frequency(range, supportedRange, stream)
%SAMPLE_SUPPORTED_FREQUENCY 从配置与参考感知支持域的交集中采样。

    validRange = isnumeric(range) && isreal(range) && ...
        isequal(size(range), [1, 2]) && all(isfinite(range)) && ...
        range(1) <= range(2);
    if ~validRange
        error('JammingGenerator:InvalidPhysicalParameter', ...
            'VFT doppler_Hz 必须是有限有序范围。');
    end
    intersection = [max(range(1), supportedRange(1)), ...
        min(range(2), supportedRange(2))];
    if supportedRange(1) > supportedRange(2) || ...
            intersection(1) > intersection(2)
        error('JammingGenerator:InvalidPhysicalParameter', ...
            'VFT doppler_Hz 与参考波形的严格无混叠支持域没有交集。');
    end
    value = sample_range(intersection, stream);
end

function signal = apply_arrival_delay(signal, delaySamples)
%APPLY_ARRIVAL_DELAY 按已量化整数样点对完整波形循环移位一次。

    signal = circshift(signal, [0, delaySamples]);
end
