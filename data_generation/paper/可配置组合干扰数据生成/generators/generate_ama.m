function [signal, params] = generate_ama(cfg, stream, reference)
%GENERATE_AMA 生成噪声调幅（AMA）干扰及物理真值。
%   [SIGNAL, PARAMS] = GENERATE_AMA(CFG, STREAM, REFERENCE) 使用标量
%   RandStream STREAM 产生实高斯白噪声，以实际干扰带宽进行 FFT 域低通
%   着色和单位 RMS 归一化，构造 1+modulation_index*noise 的幅度包络，
%   再乘中心频率复载波。CFG 必须已经标准化；REFERENCE 仅提供总长度和
%   训练时间轴。
%
%   SIGNAL 为 1×sample_length complex double 干净波形，不含接收机背景
%   噪声且不按 JNR_dB 缩放。PARAMS 中 bandwidth_factor 是实际干扰
%   带宽与雷达带宽之比，而非可能被裁剪的配置请求值；调制指数无量纲，
%   频率及实际带宽单位为 Hz，到达时延单位为 s，JNR_dB 单位为 dB。
%
%   示例：
%       reference = build_reference_signal(cfg);
%       stream = RandStream('mt19937ar', 'Seed', 2);
%       [signal, truth] = generate_ama(cfg, stream, reference);

    fs_Hz = cfg.radar.fs_Hz;
    nyquistGuard_Hz = fs_Hz / 2 - max( ...
        eps(fs_Hz / 2), fs_Hz * 1e-12);
    validate_positive_range(cfg.params.AMA.bandwidth_factor, ...
        'AMA bandwidth_factor');
    validate_delay_range(cfg.params.common.arrival_delay_s, ...
        cfg.radar.pri_s, 'AMA arrival_delay_s');
    JNR_dB = sample_range(cfg.params.common.JNR_dB, stream);
    requestedBandwidthFactor = sample_range( ...
        cfg.params.AMA.bandwidth_factor, stream);
    centerFrequency_Hz = sample_center_frequency( ...
        cfg.params.AMA.center_frequency_Hz, nyquistGuard_Hz, stream);
    modulationIndex = sample_range( ...
        cfg.params.AMA.modulation_index, stream);
    arrivalDelaySamples = round(sample_range( ...
        cfg.params.common.arrival_delay_s, stream) * fs_Hz);
    arrivalDelay_s = arrivalDelaySamples / fs_Hz;
    requestedBandwidth_Hz = requestedBandwidthFactor * ...
        cfg.radar.bandwidth_Hz;
    interferenceBandwidth_Hz = min(requestedBandwidth_Hz, ...
        2 * (nyquistGuard_Hz - abs(centerFrequency_Hz)));
    if ~isfinite(interferenceBandwidth_Hz) || ...
            interferenceBandwidth_Hz <= 0
        throw_invalid_physical_parameter( ...
            'AMA 实际带宽必须是有限正数。');
    end
    whiteNoise = randn(stream, 1, cfg.derived.sample_length);
    [coloredNoise, realizedBandwidth_Hz] = lowpass_real_noise( ...
        whiteNoise, interferenceBandwidth_Hz, fs_Hz);
    % 带宽真值采用 FFT 掩膜实际选中的离散支持宽度；落入同一掩膜的
    % 连续请求因此共享同一波形和监督标签。
    realizedBandwidthFactor = realizedBandwidth_Hz / ...
        cfg.radar.bandwidth_Hz;
    envelope = 1 + modulationIndex * coloredNoise;
    carrier = exp(1i * 2 * pi * centerFrequency_Hz * ...
        reference.time_train_s);
    signal = envelope .* carrier;
    signal = apply_arrival_delay(signal, arrivalDelaySamples);
    signal = complex(real(signal), imag(signal));

    params = struct('JNR_dB', JNR_dB, ...
        'bandwidth_factor', realizedBandwidthFactor, ...
        'interference_bandwidth_Hz', realizedBandwidth_Hz, ...
        'center_frequency_Hz', centerFrequency_Hz, ...
        'modulation_index', modulationIndex, ...
        'arrival_delay_s', arrivalDelay_s);
end

function centerFrequency_Hz = sample_center_frequency(range, guard_Hz, stream)
%SAMPLE_CENTER_FREQUENCY 从完全位于严格 Nyquist guard 内的范围采样。

    if ~is_valid_range(range)
        throw_invalid_physical_parameter('AMA center_frequency_Hz 范围非法。');
    end
    if guard_Hz <= 0 || range(1) <= -guard_Hz || range(2) >= guard_Hz
        throw_invalid_physical_parameter( ...
            'AMA center_frequency_Hz 必须严格位于 Nyquist guard 内。');
    end
    centerFrequency_Hz = sample_range(range, stream);
    if abs(centerFrequency_Hz) >= guard_Hz
        throw_invalid_physical_parameter( ...
            'AMA center_frequency_Hz 必须严格位于 Nyquist guard 内。');
    end
end

function validate_positive_range(range, name)
%VALIDATE_POSITIVE_RANGE 防御性校验严格正的连续范围。

    if ~is_valid_range(range) || range(1) <= 0
        throw_invalid_physical_parameter([name ' 必须为有限正数范围。']);
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

function [coloredNoise, realizedBandwidth_Hz] = lowpass_real_noise( ...
        whiteNoise, bandwidth_Hz, fs_Hz)
%LOWPASS_REAL_NOISE 产生实带限噪声并返回 FFT 掩膜的离散实现带宽。

    sampleLength = numel(whiteNoise);
    frequency_Hz = (-floor(sampleLength / 2): ...
        ceil(sampleLength / 2) - 1) * (fs_Hz / sampleLength);
    spectrum = fftshift(fft(whiteNoise));
    mask = abs(frequency_Hz) <= bandwidth_Hz / 2;
    selectedNonzeroFrequencies_Hz = frequency_Hz(mask & frequency_Hz ~= 0);
    if isempty(selectedNonzeroFrequencies_Hz)
        throw_invalid_physical_parameter( ...
            'AMA 离散带宽必须至少包含一对非零共轭 FFT 频点。');
    end
    realizedBandwidth_Hz = max(selectedNonzeroFrequencies_Hz) - ...
        min(selectedNonzeroFrequencies_Hz);
    if ~isfinite(realizedBandwidth_Hz) || realizedBandwidth_Hz <= 0
        throw_invalid_physical_parameter('AMA 离散实现带宽必须为有限正数。');
    end
    coloredNoise = real(ifft(ifftshift(spectrum .* mask)));
    coloredNoise = coloredNoise - mean(coloredNoise);
    rmsValue = sqrt(mean(coloredNoise .^ 2));
    if isfinite(rmsValue) && rmsValue > 0
        coloredNoise = coloredNoise / rmsValue;
    else
        throw_invalid_physical_parameter( ...
            'AMA 带限噪声去均值后必须具有有限正能量。');
    end
end

function signal = apply_arrival_delay(signal, delaySamples)
%APPLY_ARRIVAL_DELAY 按已量化整数样点对完整波形循环移位一次。

    signal = circshift(signal, [0, delaySamples]);
end
