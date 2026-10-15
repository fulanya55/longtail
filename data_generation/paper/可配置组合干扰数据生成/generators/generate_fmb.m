function [signal, params] = generate_fmb(cfg, stream, reference)
%GENERATE_FMB 生成阻塞式噪声调频（FMB）干扰及物理真值。
%   [SIGNAL, PARAMS] = GENERATE_FMB(CFG, STREAM, REFERENCE) 使用标量
%   RandStream STREAM 产生实高斯白噪声，在 FFT 域用实际干扰带宽低通
%   着色并按最大绝对值归一化，再令瞬时频率为中心频率与离散实现半带
%   宽倍着色噪声之和，对频率逐样点积分得到单位幅度复包络。CFG 必须
%   已经标准化；
%   REFERENCE 是 dispatcher 构造的参考信号，本模型只使用其总长度。
%
%   SIGNAL 为 1×sample_length complex double 干净干扰波形，不含接收机
%   背景噪声且不按 JNR_dB 缩放。PARAMS 的 bandwidth_factor 是实际
%   干扰带宽与雷达带宽之比，而非可能被裁剪的配置请求值；频率和带宽
%   单位为 Hz，到达时延单位为 s，JNR_dB 单位为 dB。实际带宽限制为
%   min(请求因子*radar bandwidth,
%   2*(Nyquist-abs(center_frequency)))，从而不产生离散时间混叠。
%
%   示例：
%       reference = build_reference_signal(cfg);
%       stream = RandStream('mt19937ar', 'Seed', 1);
%       [signal, truth] = generate_fmb(cfg, stream, reference);

    sampleLength = numel(reference.train);
    fs_Hz = cfg.radar.fs_Hz;
    nyquistGuard_Hz = fs_Hz / 2 - max( ...
        eps(fs_Hz / 2), fs_Hz * 1e-12);
    validate_positive_range(cfg.params.FMB.bandwidth_factor, ...
        'FMB bandwidth_factor');
    validate_delay_range(cfg.params.common.arrival_delay_s, ...
        cfg.radar.pri_s, 'FMB arrival_delay_s');
    JNR_dB = sample_range(cfg.params.common.JNR_dB, stream);
    requestedBandwidthFactor = sample_range( ...
        cfg.params.FMB.bandwidth_factor, stream);
    centerFrequency_Hz = sample_center_frequency( ...
        cfg.params.FMB.center_frequency_Hz, nyquistGuard_Hz, stream);
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
            'FMB 实际带宽必须是有限正数。');
    end
    whiteNoise = randn(stream, 1, sampleLength);
    [coloredNoise, realizedBandwidth_Hz] = lowpass_real_noise( ...
        whiteNoise, interferenceBandwidth_Hz, fs_Hz);
    % 监督标签和调频幅度都必须使用 FFT 掩膜真正选中的离散支持宽度。
    % 因而，落入同一掩膜的连续请求不会产生不可辨识的不同标签或波形。
    realizedBandwidthFactor = realizedBandwidth_Hz / ...
        cfg.radar.bandwidth_Hz;
    instantaneousFrequency_Hz = centerFrequency_Hz + ...
        (realizedBandwidth_Hz / 2) * coloredNoise;
    phase = 2 * pi * cumsum(instantaneousFrequency_Hz) / fs_Hz;
    signal = exp(1i * phase);
    signal = apply_arrival_delay(signal, arrivalDelaySamples);
    signal = complex(real(signal), imag(signal));

    params = struct('JNR_dB', JNR_dB, ...
        'bandwidth_factor', realizedBandwidthFactor, ...
        'interference_bandwidth_Hz', realizedBandwidth_Hz, ...
        'center_frequency_Hz', centerFrequency_Hz, ...
        'arrival_delay_s', arrivalDelay_s);
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
            'FMB 离散带宽必须至少包含一对非零共轭 FFT 频点。');
    end
    realizedBandwidth_Hz = max(selectedNonzeroFrequencies_Hz) - ...
        min(selectedNonzeroFrequencies_Hz);
    if ~isfinite(realizedBandwidth_Hz) || realizedBandwidth_Hz <= 0
        throw_invalid_physical_parameter('FMB 离散实现带宽必须为有限正数。');
    end
    coloredNoise = real(ifft(ifftshift(spectrum .* mask)));
    coloredNoise = coloredNoise - mean(coloredNoise);
    maxAbsValue = max(abs(coloredNoise));
    if isfinite(maxAbsValue) && maxAbsValue > 0
        coloredNoise = coloredNoise / maxAbsValue;
    else
        throw_invalid_physical_parameter( ...
            'FMB 带限噪声去均值后必须具有有限正能量。');
    end
end

function centerFrequency_Hz = sample_center_frequency(range, guard_Hz, stream)
%SAMPLE_CENTER_FREQUENCY 从完全位于严格 Nyquist guard 内的范围采样。

    if ~is_valid_range(range)
        throw_invalid_physical_parameter('FMB center_frequency_Hz 范围非法。');
    end
    if guard_Hz <= 0 || range(1) <= -guard_Hz || range(2) >= guard_Hz
        throw_invalid_physical_parameter( ...
            'FMB center_frequency_Hz 必须满足 abs(center)<Nyquist。');
    end
    centerFrequency_Hz = sample_range(range, stream);
    if abs(centerFrequency_Hz) >= guard_Hz
        throw_invalid_physical_parameter( ...
            'FMB center_frequency_Hz 必须满足 abs(center)<Nyquist。');
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

function signal = apply_arrival_delay(signal, delaySamples)
%APPLY_ARRIVAL_DELAY 按已量化整数样点对完整波形循环移位一次。

    signal = circshift(signal, [0, delaySamples]);
end
