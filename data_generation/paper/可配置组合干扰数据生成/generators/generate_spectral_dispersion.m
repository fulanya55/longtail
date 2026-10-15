function [signal, params] = generate_spectral_dispersion( ...
        cfg, stream, reference)
%GENERATE_SPECTRAL_DISPERSION 生成标准整数子脉冲 SMSP 及物理真值。
%   [SIGNAL, PARAMS] = GENERATE_SPECTRAL_DISPERSION(CFG, STREAM, REFERENCE)
%   使用标准 DRFM 频谱弥散模型：从 2--8 中抽取整数子脉冲数 M，将原 LFM
%   脉宽 T 划分为约 T/M 的短 LFM，每个短 LFM 仍扫描原带宽 B，因而实际
%   调频斜率约为 M*(B/T)。同一个复子脉冲精确重复 M 次，使输出总时长约
%   等于原始脉宽。模型没有独立连续压缩比，也不把带宽扩大为 M*B。
%
%   子脉冲样点数取 round(samples_per_pulse/M)。离散量化后的宽度、斜率
%   倍数、调频斜率和总时长全部写入 PARAMS；subpulse_bandwidth_Hz 保存
%   参考 LFM 的名义带宽。所有子脉冲共享一个全局初相并精确复用同一复数
%   序列，因此在每个边界从高瞬时频率相干复位到低瞬时频率。
%
%   公共到达时延量化后对单 PRI SMSP 使用零边界平移，再把该 PRI 精确
%   重复到整条记录。输出不含接收机噪声，也不在本函数内按 JNR 缩放。
%   subpulse_start_01_s...08 描述首 PRI、公共时延之前的连续子脉冲起点；
%   未使用槽为 NaN。物理边界错误统一使用
%   JammingGenerator:InvalidPhysicalParameter。

    capacity = 8;
    validate_stream(stream);
    smsp = cfg.params.SpectralDispersion;
    validate_smsp_config(smsp, capacity);
    validate_real_range(cfg.params.common.JNR_dB, ...
        'SMSP JNR_dB');
    validate_delay_range(cfg.params.common.arrival_delay_s, ...
        cfg.radar.pri_s, 'SMSP arrival_delay_s');

    M = sample_integer(smsp.num_subpulses, stream);
    subpulseSamples = round(cfg.derived.samples_per_pulse / M);
    if subpulseSamples < 2
        throw_invalid_physical_parameter( ...
            'SMSP 每个子脉冲至少需要两个样点。');
    end

    centered = build_centered_reference_signal(cfg, reference);
    validate_reference_support(centered.support_Hz, cfg.radar.fs_Hz);

    fs_Hz = cfg.radar.fs_Hz;
    subpulseWidth_s = subpulseSamples / fs_Hz;
    chirpRate_Hz_per_s = ...
        cfg.radar.bandwidth_Hz / subpulseWidth_s;
    chirpRateMultiplier = ...
        cfg.radar.pulse_width_s / subpulseWidth_s;
    outputSamples = M * subpulseSamples;
    outputDuration_s = outputSamples / fs_Hz;
    arrivalDelaySamples = round(sample_range( ...
        cfg.params.common.arrival_delay_s, stream) * fs_Hz);
    if arrivalDelaySamples + outputSamples > ...
            cfg.derived.samples_per_pri
        throw_invalid_physical_parameter( ...
            'SMSP 公共到达时延加输出时长必须完整位于 PRI 内。');
    end

    JNR_dB = sample_range(cfg.params.common.JNR_dB, stream);
    initialPhase_rad = 2 * pi * rand(stream);
    localTime_s = (0:subpulseSamples - 1) / fs_Hz;
    centerTime_s = (subpulseSamples - 1) / (2 * fs_Hz);
    subpulse = exp(1i * (pi * chirpRate_Hz_per_s * ...
        (localTime_s - centerTime_s) .^ 2 + initialPhase_rad));
    cleanPri = complex(zeros(1, cfg.derived.samples_per_pri));
    cleanPri(1:outputSamples) = repmat(subpulse, 1, M);
    delayedPri = delay_with_zeros(cleanPri, arrivalDelaySamples);
    signal = complex(repmat(delayedPri, 1, cfg.radar.num_pulses));
    signal = complex(real(signal), imag(signal));

    params = struct( ...
        'JNR_dB', JNR_dB, ...
        'num_subpulses', M, ...
        'subpulse_width_s', subpulseWidth_s, ...
        'chirp_rate_multiplier', chirpRateMultiplier, ...
        'subpulse_chirp_rate_Hz_per_s', chirpRate_Hz_per_s, ...
        'subpulse_bandwidth_Hz', cfg.radar.bandwidth_Hz, ...
        'output_duration_s', outputDuration_s, ...
        'arrival_delay_s', arrivalDelaySamples / fs_Hz, ...
        'initial_phase_rad', initialPhase_rad);
    for subpulseIndex = 1:capacity
        fieldName = sprintf('subpulse_start_%02d_s', subpulseIndex);
        if subpulseIndex <= M
            params.(fieldName) = ...
                (subpulseIndex - 1) * subpulseWidth_s;
        else
            params.(fieldName) = NaN;
        end
    end
end

function validate_smsp_config(smsp, capacity)
%VALIDATE_SMSP_CONFIG 防御性校验整数子脉冲范围和固定真值容量。

    required = {'num_subpulses', 'max_subpulses'};
    if ~isstruct(smsp) || ~isscalar(smsp) || ...
            ~all(isfield(smsp, required))
        throw_invalid_physical_parameter('SMSP 参数结构缺少必需字段。');
    end
    validCount = is_integer_range(smsp.num_subpulses) && ...
        smsp.num_subpulses(1) >= 2 && ...
        smsp.num_subpulses(2) <= capacity;
    validCapacity = isnumeric(smsp.max_subpulses) && ...
        isreal(smsp.max_subpulses) && isscalar(smsp.max_subpulses) && ...
        isfinite(smsp.max_subpulses) && smsp.max_subpulses == capacity;
    if ~(validCount && validCapacity)
        throw_invalid_physical_parameter( ...
            'SMSP num_subpulses 必须为 2--8，固定容量必须为 8。');
    end
end

function validate_reference_support(support_Hz, fs_Hz)
%VALIDATE_REFERENCE_SUPPORT 保证零中心参考带宽严格位于 Nyquist guard 内。

    guard_Hz = fs_Hz / 2 - max(eps(fs_Hz / 2), fs_Hz * 1e-12);
    if ~isnumeric(support_Hz) || ~isreal(support_Hz) || ...
            ~isequal(size(support_Hz), [1, 2]) || ...
            any(~isfinite(support_Hz)) || ...
            support_Hz(1) <= -guard_Hz || support_Hz(2) >= guard_Hz
        throw_invalid_physical_parameter( ...
            'SMSP 零中心参考带宽必须严格位于 Nyquist guard 内。');
    end
end

function validate_stream(stream)
%VALIDATE_STREAM 要求全部随机量只使用调用者传入的标量随机流。

    if ~(isa(stream, 'RandStream') && isscalar(stream))
        error('JammingGenerator:InvalidStream', ...
            'stream 必须是标量 RandStream。');
    end
end

function validate_delay_range(range, pri_s, name)
%VALIDATE_DELAY_RANGE 校验非负且严格小于 PRI 的时延范围。

    if ~is_valid_range(range) || range(1) < 0 || range(2) >= pri_s
        throw_invalid_physical_parameter( ...
            [name ' 必须满足下界>=0且上界<PRI。']);
    end
end

function validate_real_range(range, name)
%VALIDATE_REAL_RANGE 校验允许任意有限实数的有序范围。

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
%SAMPLE_RANGE 使用局部随机流在连续范围内均匀抽样。

    value = range(1) + (range(2) - range(1)) * rand(stream);
end

function value = sample_integer(range, stream)
%SAMPLE_INTEGER 使用局部随机流从含端点整数范围抽样。

    value = randi(stream, double(range), 1, 1);
end

function delayed = delay_with_zeros(signal, delaySamples)
%DELAY_WITH_ZEROS 对单 PRI 波形执行非循环整数样点时延。

    delayed = complex(zeros(size(signal)));
    if delaySamples < numel(signal)
        delayed(delaySamples + 1:end) = signal(1:end - delaySamples);
    end
end

function throw_invalid_physical_parameter(message)
%THROW_INVALID_PHYSICAL_PARAMETER 抛出统一物理边界错误。

    error('JammingGenerator:InvalidPhysicalParameter', '%s', message);
end
