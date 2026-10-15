function [signal, params] = generate_traction(cfg, stream, reference)
%GENERATE_TRACTION 生成脉间距离连续变化的拖引干扰及运动真值。
%   [SIGNAL, PARAMS] = GENERATE_TRACTION(CFG, STREAM, REFERENCE) 使用标量
%   RandStream STREAM 采样初始距离 R0、径向速度 v 和幅度 A。对脉冲编号
%   p=0...P-1，使用 R(p)=R0+v*p*PRI、delay(p)=2*R(p)/c 计算分数样点
%   时延，通过零边界线性插值延迟单 PRI 参考脉冲并乘 A，最后拼接。CFG 必须
%   已经标准化且 P>=2；延迟不会从 PRI 尾部循环泄漏到开头。
%
%   SIGNAL 为 1×sample_length complex double 干净干扰，不含接收机背景
%   噪声。Traction 不采样 JNR_dB 或公共 arrival_delay_s：amplitude 是
%   唯一的全局幅度监督，传播时延已经由距离轨迹表达。后续组合层禁止另
%   采 JNR 或把波形归一化到独立绝对功率；只允许把 manifest 相对功率
%   偏置换算为线性幅度增益，并将 PARAMS.amplitude 更新为最终 effective
%   amplitude。绝对功率或 JNR 只能由最终波形推导并作为元数据，不能成为
%   独立随机监督量。PARAMS 的距离与脉间距离变化单位为 m，速度单位为
%   m/s，幅度无量纲，脉间时延变化单位为 s。
%
%   示例：
%       reference = build_reference_signal(cfg);
%       stream = RandStream('mt19937ar', 'Seed', 10);
%       [signal, truth] = generate_traction(cfg, stream, reference);

    numPulses = cfg.radar.num_pulses;
    if numPulses < 2
        error('JammingGenerator:TractionRequiresMultiplePulses', ...
            'Traction 至少需要两个脉冲描述脉间距离和时延变化。');
    end
    validate_nonnegative_range(cfg.params.Traction.initial_range_m, ...
        'Traction initial_range_m');
    validate_positive_range(cfg.params.Traction.amplitude, ...
        'Traction amplitude');

    initialRange_m = sample_range( ...
        cfg.params.Traction.initial_range_m, stream);
    velocity_mps = sample_range( ...
        cfg.params.Traction.velocity_mps, stream);
    amplitude = sample_range(cfg.params.Traction.amplitude, stream);
    rangeChangePerPulse_m = velocity_mps * cfg.radar.pri_s;
    delayChangePerPulse_s = 2 * rangeChangePerPulse_m / ...
        cfg.radar.c_mps;

    samplesPerPri = numel(reference.pulse);
    signal = complex(zeros(1, samplesPerPri * numPulses));
    upperTolerance_s = 32 * eps(max([cfg.radar.pri_s, ...
        cfg.radar.pulse_width_s]));
    for pulseIndex = 1:numPulses
        zeroBasedPulseIndex = pulseIndex - 1;
        currentRange_m = initialRange_m + velocity_mps * ...
            zeroBasedPulseIndex * cfg.radar.pri_s;
        currentDelay_s = 2 * currentRange_m / cfg.radar.c_mps;
        if ~isfinite(currentRange_m) || currentRange_m < 0 || ...
                ~isfinite(currentDelay_s) || currentDelay_s < 0 || ...
                currentDelay_s + cfg.radar.pulse_width_s > ...
                cfg.radar.pri_s + upperTolerance_s
            throw_invalid_physical_parameter( ...
                ['Traction 每脉冲必须满足 delay_s>=0 且 ' ...
                'delay_s+pulse_width_s<=pri_s（上界容差 32 eps）。']);
        end
        delaySamples = currentDelay_s * cfg.radar.fs_Hz;
        shiftedPulse = amplitude * fractional_zero_boundary_delay( ...
            reference.pulse, delaySamples);
        blockStart = zeroBasedPulseIndex * samplesPerPri + 1;
        blockStop = pulseIndex * samplesPerPri;
        signal(blockStart:blockStop) = shiftedPulse;
    end
    signal = complex(real(signal), imag(signal));

    params = struct('initial_range_m', initialRange_m, ...
        'velocity_mps', velocity_mps, 'amplitude', amplitude, ...
        'range_change_per_pulse_m', rangeChangePerPulse_m, ...
        'delay_change_per_pulse_s', delayChangePerPulse_s);
end

function shifted = fractional_zero_boundary_delay(signal, delaySamples)
%FRACTIONAL_ZERO_BOUNDARY_DELAY 用线性插值实现正向非循环分数时延。

    sampleLength = numel(signal);
    sampleIndex = 0:sampleLength - 1;
    shifted = interp1(sampleIndex, signal, ...
        sampleIndex - delaySamples, 'linear', 0);
    shifted = complex(real(shifted), imag(shifted));
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
