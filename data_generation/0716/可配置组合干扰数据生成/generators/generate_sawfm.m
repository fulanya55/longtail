function [signal, params] = generate_sawfm(cfg, stream, reference)
%GENERATE_SAWFM 生成锯齿波扫频（SawFM）干扰及物理真值。
%   [SIGNAL, PARAMS] = GENERATE_SAWFM(CFG, STREAM, REFERENCE) 使用标量
%   RandStream STREAM 采样扫频带宽 B 和周期 T，在参考训练时间轴上构造
%   -B/2+B*mod(t,T)/T 的锯齿瞬时频率，并逐样点积分生成单位幅度复包络。
%   CFG 必须已经标准化；REFERENCE 提供以秒计的完整训练时间轴。
%
%   SIGNAL 为 1×sample_length complex double 干净干扰，不含接收机背景
%   噪声且不按 JNR_dB 缩放。PARAMS 中带宽单位为 Hz，扫频斜率单位为
%   Hz/s，周期和到达时延单位为 s，JNR_dB 单位为 dB。
%
%   示例：
%       reference = build_reference_signal(cfg);
%       stream = RandStream('mt19937ar', 'Seed', 3);
%       [signal, truth] = generate_sawfm(cfg, stream, reference);

    fs_Hz = cfg.radar.fs_Hz;
    nyquistGuard_Hz = fs_Hz / 2 - max( ...
        eps(fs_Hz / 2), fs_Hz * 1e-12);
    validate_positive_range(cfg.params.SawFM.sweep_bandwidth_Hz, ...
        'SawFM sweep_bandwidth_Hz');
    validate_positive_range(cfg.params.SawFM.sweep_period_s, ...
        'SawFM sweep_period_s');
    validate_delay_range(cfg.params.common.arrival_delay_s, ...
        cfg.radar.pri_s, 'SawFM arrival_delay_s');
    JNR_dB = sample_range(cfg.params.common.JNR_dB, stream);
    requestedSweepBandwidth_Hz = sample_range( ...
        cfg.params.SawFM.sweep_bandwidth_Hz, stream);
    sweepBandwidth_Hz = min(requestedSweepBandwidth_Hz, ...
        2 * nyquistGuard_Hz);
    sweepPeriod_s = sample_range( ...
        cfg.params.SawFM.sweep_period_s, stream);
    arrivalDelaySamples = round(sample_range( ...
        cfg.params.common.arrival_delay_s, stream) * fs_Hz);
    arrivalDelay_s = arrivalDelaySamples / fs_Hz;
    sweepSlope_Hz_per_s = sweepBandwidth_Hz / sweepPeriod_s;

    instantaneousFrequency_Hz = -sweepBandwidth_Hz / 2 + ...
        sweepBandwidth_Hz * mod(reference.time_train_s, ...
        sweepPeriod_s) / sweepPeriod_s;
    phase = 2 * pi * cumsum(instantaneousFrequency_Hz) / fs_Hz;
    signal = exp(1i * phase);
    signal = apply_arrival_delay(signal, arrivalDelaySamples);
    signal = complex(real(signal), imag(signal));

    params = struct('JNR_dB', JNR_dB, ...
        'sweep_bandwidth_Hz', sweepBandwidth_Hz, ...
        'sweep_slope_Hz_per_s', sweepSlope_Hz_per_s, ...
        'sweep_period_s', sweepPeriod_s, ...
        'arrival_delay_s', arrivalDelay_s);
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

function signal = apply_arrival_delay(signal, delaySamples)
%APPLY_ARRIVAL_DELAY 按已量化整数样点对完整波形循环移位一次。

    signal = circshift(signal, [0, delaySamples]);
end
