function reference = build_reference_signal(cfg)
%BUILD_REFERENCE_SIGNAL 构造单个 PRI 参考 LFM 及其展平脉冲串。
%   REFERENCE = BUILD_REFERENCE_SIGNAL(CFG) 返回标量结构体 REFERENCE，字段
%   顺序固定为 pulse、train、time_pri_s、time_train_s。pulse 是
%   1×samples_per_pri 的 complex double 行向量，train 按 PRI 顺序直接
%   展平为 1×sample_length 的 complex double 行向量；两个时间字段均以秒
%   为单位，并与各自信号保持相同的行向量形状。
%
%   有效脉冲使用基带线性调频公式
%       s(t) = exp(1i*pi*K*t^2),  t = (0:Npulse-1)/fs
%   其中 K 的单位为 Hz/s。每个 PRI 的剩余样点严格补零，随后将同一 PRI
%   重复 cfg.radar.num_pulses 次。分配前会校验最大时间、最大二次相位和
%   pulse/train/时间数组的预计字节数；超出有限数值域或
%   cfg.output.max_estimated_bytes 时统一拒绝。在 Windows 上还会只读查询
%   memory 给出的单数组上限和可用总数组内存，并按 80% 可用量保守预算；
%   查询不可用时使用固定保守上限。函数只读取配置，不写文件、不使用
%   随机数，也不修改输入结构体。
%
%   示例：
%       cfg = normalize_and_validate_config( ...
%           create_default_config(), 'balanced');
%       reference = build_reference_signal(cfg);
%       plot(reference.time_pri_s, real(reference.pulse));

    if nargin ~= 1 || ~isstruct(cfg) || ~isscalar(cfg) || ...
            ~isfield(cfg, 'radar') || ~isstruct(cfg.radar) || ...
            ~isscalar(cfg.radar) || ~isfield(cfg, 'derived') || ...
            ~isstruct(cfg.derived) || ~isscalar(cfg.derived) || ...
            ~isfield(cfg, 'output') || ~isstruct(cfg.output) || ...
            ~isscalar(cfg.output)
        throw_invalid_config();
    end

    radarFields = {'fs_Hz', 'num_pulses'};
    derivedFields = {'samples_per_pri', 'samples_per_pulse', ...
        'sample_length', 'chirp_rate_Hz_per_s'};
    outputFields = {'max_estimated_bytes'};
    if ~all(isfield(cfg.radar, radarFields)) || ...
            ~all(isfield(cfg.derived, derivedFields)) || ...
            ~all(isfield(cfg.output, outputFields))
        throw_invalid_config();
    end

    fs_Hz = cfg.radar.fs_Hz;
    numPulses = cfg.radar.num_pulses;
    samplesPerPri = cfg.derived.samples_per_pri;
    samplesPerPulse = cfg.derived.samples_per_pulse;
    sampleLength = cfg.derived.sample_length;
    chirpRate = cfg.derived.chirp_rate_Hz_per_s;
    maxEstimatedBytes = cfg.output.max_estimated_bytes;

    if ~is_positive_finite_scalar(fs_Hz) || ...
            ~is_positive_integer_scalar(numPulses) || ...
            ~is_positive_integer_scalar(samplesPerPri) || ...
            ~is_positive_integer_scalar(samplesPerPulse) || ...
            ~is_positive_integer_scalar(sampleLength) || ...
            ~is_nonzero_finite_scalar(chirpRate) || ...
            ~is_valid_capacity_scalar(maxEstimatedBytes)
        throw_invalid_config();
    end

    fs_Hz = double(fs_Hz);
    numPulses = double(numPulses);
    samplesPerPri = double(samplesPerPri);
    samplesPerPulse = double(samplesPerPulse);
    sampleLength = double(sampleLength);
    chirpRate = double(chirpRate);
    maxEstimatedBytes = double(maxEstimatedBytes);

    expectedLength = samplesPerPri * numPulses;
    if samplesPerPulse > samplesPerPri || ~isfinite(expectedLength) || ...
            expectedLength > flintmax || sampleLength ~= expectedLength
        throw_invalid_config();
    end

    % 以下数值域和容量检查必须先于任何按配置长度进行的数组分配。
    maxTime_s = (sampleLength - 1) / fs_Hz;
    maxPulseTime_s = (samplesPerPulse - 1) / fs_Hz;
    if ~isfinite(maxTime_s) || ~isfinite(maxPulseTime_s)
        throw_invalid_config();
    end
    timeSquared = maxPulseTime_s ^ 2;
    scaledPhase = abs(chirpRate) * timeSquared;
    maxPhase = pi * scaledPhase;
    if ~isfinite(timeSquared) || ~isfinite(scaledPhase) || ...
            ~isfinite(maxPhase)
        throw_invalid_config();
    end

    referenceSampleCount = samplesPerPri + sampleLength;
    pulseBytes = 16 * samplesPerPri;
    trainBytes = 16 * sampleLength;
    timePriBytes = 8 * samplesPerPri;
    timeTrainBytes = 8 * sampleLength;
    outputBytes = pulseBytes + trainBytes + timePriBytes + timeTrainBytes;

    % 构造峰值还计入 LFM、脉内时间、相位及 pulse 实虚部临时数组。
    constructionTemporaryBytes = 32 * samplesPerPulse + ...
        16 * samplesPerPri + 8 * sampleLength;
    peakBytes = outputBytes + constructionTemporaryBytes;
    [maxArrayBytes, effectiveBudgetBytes] = ...
        get_allocation_limits(maxEstimatedBytes);
    maxComplexElements = floor(maxArrayBytes / 16);
    maxDoubleElements = floor(maxArrayBytes / 8);
    if ~isfinite(referenceSampleCount) || ~isfinite(outputBytes) || ...
            ~isfinite(constructionTemporaryBytes) || ...
            ~isfinite(peakBytes) || ...
            samplesPerPri > maxComplexElements || ...
            sampleLength > maxComplexElements || ...
            samplesPerPri > maxDoubleElements || ...
            sampleLength > maxDoubleElements || ...
            peakBytes > effectiveBudgetBytes
        throw_invalid_config();
    end

    timePulse_s = (0:samplesPerPulse - 1) / fs_Hz;
    phase = pi * (chirpRate * timePulse_s .^ 2);
    lfm = exp(1i * phase);
    if ~all_values_are_finite(lfm)
        throw_invalid_config();
    end

    % 最终显式组合实虚部，保证单有效样点且虚部全零时仍保持 complex。
    paddingLength = samplesPerPri - samplesPerPulse;
    pulse = complex([real(lfm), zeros(1, paddingLength)], ...
        [imag(lfm), zeros(1, paddingLength)]);
    % 本机 MATLAB 会把虚部全零的 repmat 结果降为 real；单参数 complex
    % 只做一次复数化，不分别复制完整实部和虚部，并已计入上方峰值预算。
    train = complex(repmat(pulse, 1, numPulses));
    timePri_s = (0:samplesPerPri - 1) / fs_Hz;
    timeTrain_s = (0:sampleLength - 1) / fs_Hz;
    if ~all_values_are_finite(pulse) || ...
            ~all_values_are_finite(train) || ...
            ~all_values_are_finite(timePri_s) || ...
            ~all_values_are_finite(timeTrain_s)
        throw_invalid_config();
    end

    reference = struct('pulse', pulse, 'train', train, ...
        'time_pri_s', timePri_s, 'time_train_s', timeTrain_s);
end

function valid = is_positive_finite_scalar(value)
%IS_POSITIVE_FINITE_SCALAR 判断数值是否为有限正实标量。

    valid = isnumeric(value) && isreal(value) && isscalar(value) && ...
        isfinite(value) && value > 0 && integer_value_fits_double(value);
end

function valid = is_positive_integer_scalar(value)
%IS_POSITIVE_INTEGER_SCALAR 判断数值是否为可安全索引的有限正整数。

    valid = is_positive_finite_scalar(value) && value <= flintmax && ...
        value == fix(value);
end

function valid = is_nonzero_finite_scalar(value)
%IS_NONZERO_FINITE_SCALAR 判断数值是否为有限非零实标量。

    valid = isnumeric(value) && isreal(value) && isscalar(value) && ...
        isfinite(value) && value ~= 0 && integer_value_fits_double(value);
end

function valid = is_valid_capacity_scalar(value)
%IS_VALID_CAPACITY_SCALAR 判断容量是否为可精确转换的有限非负标量。

    valid = isnumeric(value) && isreal(value) && isscalar(value) && ...
        isfinite(value) && value >= 0 && value <= flintmax && ...
        integer_value_fits_double(value);
end

function valid = integer_value_fits_double(value)
%INTEGER_VALUE_FITS_DOUBLE 拒绝超出 double 连续整数域的整数标量。

    if isa(value, 'uint64')
        valid = value <= uint64(flintmax);
    elseif isa(value, 'int64')
        limit = int64(flintmax);
        valid = value >= -limit && value <= limit;
    else
        valid = true;
    end
end

function valid = all_values_are_finite(value)
%ALL_VALUES_ARE_FINITE 同时检查数组实部和虚部均为有限数。

    valid = all(isfinite(real(value))) && all(isfinite(imag(value)));
end

function [maxArrayBytes, effectiveBudgetBytes] = ...
        get_allocation_limits(configuredBudgetBytes)
%GET_ALLOCATION_LIMITS 获取平台单数组上限和保守总峰值预算。
%   Windows memory 查询只读取当前可分配信息；字段缺失、数值非法或平台
%   不支持时，回退到 1 TiB，确保仍在任何数组分配前执行明确上限检查。

    fallbackBytes = 2 ^ 40;
    maxArrayBytes = fallbackBytes;
    availableBudgetBytes = fallbackBytes;
    try
        userView = memory;
        if isstruct(userView) && isscalar(userView)
            if isfield(userView, 'MaxPossibleArrayBytes') && ...
                    is_valid_memory_limit( ...
                    userView.MaxPossibleArrayBytes)
                maxArrayBytes = double(userView.MaxPossibleArrayBytes);
            end
            if isfield(userView, 'MemAvailableAllArrays') && ...
                    is_valid_memory_limit( ...
                    userView.MemAvailableAllArrays)
                conservativeAvailable = 0.8 * double( ...
                    userView.MemAvailableAllArrays);
                if isfinite(conservativeAvailable) && ...
                        conservativeAvailable > 0
                    availableBudgetBytes = conservativeAvailable;
                end
            end
        end
    catch
        % 保留无 memory 平台的固定保守上限，不泄漏平台相关内部错误。
    end
    effectiveBudgetBytes = min(configuredBudgetBytes, ...
        availableBudgetBytes);
end

function valid = is_valid_memory_limit(value)
%IS_VALID_MEMORY_LIMIT 判断 memory 返回值是否为有限正实标量。

    valid = isnumeric(value) && isreal(value) && isscalar(value) && ...
        isfinite(value) && value > 0;
end

function throw_invalid_config()
%THROW_INVALID_CONFIG 抛出参考信号配置的统一错误标识。

    error('JammingReference:InvalidConfig', ...
        ['参考信号配置、时间/相位数值域或输出容量无效；样点长度必须' ...
        '一致且全部数组必须能在配置容量内有限表示。']);
end
