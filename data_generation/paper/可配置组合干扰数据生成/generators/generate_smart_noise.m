function [signal, params] = generate_smart_noise(cfg, stream, reference)
%GENERATE_SMART_NOISE 生成智能噪声（SN）采样转发干扰及物理真值。
%   [SIGNAL, PARAMS] = GENERATE_SMART_NOISE(CFG, STREAM, REFERENCE) 使用
%   标量 RandStream STREAM 采样截取宽度和整数副本数，以“开启一个截取
%   宽度、关闭 replicas 个截取宽度”的周期窗采样参考脉冲串；随后将
%   r=0...replicas 的采样片段分别延迟 r 个截取宽度并累加。每一份转发
%   片段使用 STREAM 产生的独立均匀随机初相，用于表达智能噪声干扰机的
%   相位扰动，而不是接收机背景噪声。CFG 必须已经标准化。
%
%   SIGNAL 为 1×sample_length complex double 干净干扰，不额外叠加背景
%   噪声且不按 JNR_dB 缩放。PARAMS 中 sample_width_s、sample_period_s
%   和 arrival_delay_s 单位为 s，num_replicas 为整数计数，JNR_dB 为 dB；
%   sample_period_s=(1+num_replicas)*sample_width_s。
%
%   示例：
%       reference = build_reference_signal(cfg);
%       stream = RandStream('mt19937ar', 'Seed', 5);
%       [signal, truth] = generate_smart_noise(cfg, stream, reference);

    fs_Hz = cfg.radar.fs_Hz;
    validate_positive_range(cfg.params.SN.sample_width_s, ...
        'SN sample_width_s');
    validate_delay_range(cfg.params.common.arrival_delay_s, ...
        cfg.radar.pri_s, 'SN arrival_delay_s');
    JNR_dB = sample_range(cfg.params.common.JNR_dB, stream);
    requestedSampleWidth_s = sample_range( ...
        cfg.params.SN.sample_width_s, stream);
    widthSamples = max(1, round(requestedSampleWidth_s * fs_Hz));
    sampleWidth_s = widthSamples / fs_Hz;
    numReplicas = sample_integer(cfg.params.SN.num_replicas, stream);
    arrivalDelaySamples = round(sample_range( ...
        cfg.params.common.arrival_delay_s, stream) * fs_Hz);
    arrivalDelay_s = arrivalDelaySamples / fs_Hz;
    samplePeriod_s = (1 + numReplicas) * sampleWidth_s;

    if (numReplicas + 1) * widthSamples > numel(reference.train)
        throw_invalid_physical_parameter( ...
            'SN 请求的基段和全部副本必须完整位于记录长度内。');
    end
    periodSamples = (1 + numReplicas) * widthSamples;
    sampleIndex = 0:numel(reference.train) - 1;
    samplingWindow = mod(sampleIndex, periodSamples) < widthSamples;
    sampledReference = reference.train .* double(samplingWindow);
    signal = complex(zeros(size(reference.train)));
    for replicaIndex = 0:numReplicas
        shiftSamples = replicaIndex * widthSamples;
        % 每份均匀随机初相属于干扰机内部相位扰动，不是背景噪声。
        phaseOffset = 2 * pi * rand(stream);
        delayedReplica = delay_with_zeros( ...
            sampledReference, shiftSamples);
        signal = signal + exp(1i * phaseOffset) * delayedReplica;
    end
    signal = apply_arrival_delay(signal, arrivalDelaySamples);
    signal = complex(real(signal), imag(signal));

    params = struct('JNR_dB', JNR_dB, ...
        'sample_width_s', sampleWidth_s, ...
        'num_replicas', numReplicas, ...
        'sample_period_s', samplePeriod_s, ...
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

function value = sample_integer(range, stream)
%SAMPLE_INTEGER 使用局部随机流从含端点的整数范围内均匀采样。

    value = randi(stream, double(range), 1, 1);
end

function delayed = delay_with_zeros(signal, delaySamples)
%DELAY_WITH_ZEROS 对单份转发片段施加非循环整数样点延迟。

    delayed = complex(zeros(size(signal)));
    if delaySamples < numel(signal)
        delayed(delaySamples + 1:end) = ...
            signal(1:end - delaySamples);
    end
end

function signal = apply_arrival_delay(signal, delaySamples)
%APPLY_ARRIVAL_DELAY 按已量化整数样点对完整波形循环移位一次。

    signal = circshift(signal, [0, delaySamples]);
end
