function [signal, params] = generate_single_jamming( ...
        code, cfg, registry, stream, reference)
%GENERATE_SINGLE_JAMMING 按注册表分派一类无背景噪声的干扰生成器。
%   [SIGNAL, PARAMS] = GENERATE_SINGLE_JAMMING(CODE, CFG, REGISTRY,
%   STREAM) 将字符行向量或非缺失 string 标量 CODE 规范化，在 REGISTRY
%   中查找唯一生成器，并用已经标准化的 CFG、标量 RandStream STREAM 和
%   本函数仅构造一次的参考 LFM 信号调用该生成器。SIGNAL 统一返回
%   1×cfg.derived.sample_length complex double 行向量；PARAMS 为包含注册表
%   parameter_names 全部字段的标量结构体，各物理量单位由注册表定义。
%   可选第五参数 REFERENCE 允许上层复用已经构造的参考脉冲/脉冲串；省略
%   时保持四参数兼容并在本函数内构造一次。外部 REFERENCE 必须包含合法的
%   pulse、train、time_pri_s 和 time_train_s 行向量。
%
%   本层不叠加接收机背景噪声，也不缩放原始干扰。组合层对所有声明
%   PARAMS.JNR_dB 的类型做绝对功率标定；Traction 的 amplitude 是唯一全局
%   幅度监督，只允许将相对功率偏置并入最终 amplitude，禁止独立 JNR
%   归一化。生成器自身的噪声调制或随机相位属于干扰物理模型。未知或
%   畸形 CODE 使用
%   JammingGenerator:UnknownType；注册表错误和生成输出错误分别使用
%   JammingGenerator:InvalidRegistry 与
%   JammingGenerator:InvalidGeneratorOutput。
%
%   示例：
%       cfg = normalize_and_validate_config( ...
%           create_default_config(), 'balanced');
%       registry = build_jamming_registry(cfg);
%       stream = RandStream('mt19937ar', 'Seed', 7);
%       [signal, truth] = generate_single_jamming( ...
%           'FMB', cfg, registry, stream);

    code = normalize_code(code);
    validate_registry(registry);
    if ~(isa(stream, 'RandStream') && isscalar(stream))
        error('JammingGenerator:InvalidStream', ...
            'stream 必须是标量 RandStream，生成器不得使用全局随机状态。');
    end

    registryCodes = cellfun(@normalize_registry_code, ...
        {registry.code}, 'UniformOutput', false);
    entryIndex = find(strcmp(code, registryCodes), 1, 'first');
    if isempty(entryIndex)
        throw_unknown_type(code);
    end
    entry = registry(entryIndex);
    if nargin < 5
        reference = build_reference_signal(cfg);
    else
        validate_reference(reference, cfg);
    end
    generator = entry.generator;

    try
        [signal, params] = generator(cfg, stream, reference);
    catch exception
        if startsWith(exception.identifier, 'JammingGenerator:')
            rethrow(exception);
        end
        wrapped = MException('JammingGenerator:InvalidGeneratorOutput', ...
            '类型 %s 的生成器执行失败或未返回两个有效输出。', code);
        wrapped = addCause(wrapped, exception);
        throwAsCaller(wrapped);
    end

    signal = validate_signal(signal, cfg.derived.sample_length, code);
    validate_params(params, entry.parameter_names, code);
end

function validate_reference(reference, cfg)
%VALIDATE_REFERENCE 校验可复用参考信号的字段、形状、有限性及配置长度。

    requiredFields = {'pulse', 'train', 'time_pri_s', 'time_train_s'};
    if ~isstruct(reference) || ~isscalar(reference) || ...
            ~all(isfield(reference, requiredFields))
        throw_invalid_reference();
    end

    pulse = reference.pulse;
    train = reference.train;
    timePri_s = reference.time_pri_s;
    timeTrain_s = reference.time_train_s;
    validPulse = is_valid_signal_row(pulse, cfg.derived.samples_per_pri);
    validTrain = is_valid_signal_row(train, cfg.derived.sample_length);
    validTimePri = is_valid_time_row( ...
        timePri_s, cfg.derived.samples_per_pri);
    validTimeTrain = is_valid_time_row( ...
        timeTrain_s, cfg.derived.sample_length);
    if ~(validPulse && validTrain && validTimePri && validTimeTrain)
        throw_invalid_reference();
    end
    validate_reference_semantics(reference, cfg);
end

function validate_reference_semantics(reference, cfg)
%VALIDATE_REFERENCE_SEMANTICS 校验时间轴、LFM 内容和逐 PRI 重复语义。

    fs_Hz = cfg.radar.fs_Hz;
    samplesPerPri = cfg.derived.samples_per_pri;
    samplesPerPulse = cfg.derived.samples_per_pulse;
    numPulses = cfg.radar.num_pulses;
    timeTolerance_s = 64 * eps(1) * ...
        max(1, cfg.derived.sample_length) / fs_Hz;
    waveformTolerance = 64 * eps(1) * max(1, samplesPerPulse);

    expectedTimePri_s = (0:samplesPerPri - 1) / fs_Hz;
    if max(abs(reference.time_pri_s - expectedTimePri_s)) > ...
            timeTolerance_s
        throw_invalid_reference_semantics();
    end

    pulseTime_s = (0:samplesPerPulse - 1) / fs_Hz;
    expectedLfm = exp(1i * pi * cfg.derived.chirp_rate_Hz_per_s * ...
        pulseTime_s .^ 2);
    if max(abs(reference.pulse(1:samplesPerPulse) - expectedLfm)) > ...
            waveformTolerance
        throw_invalid_reference_semantics();
    end
    if samplesPerPulse < samplesPerPri && ...
            any(abs(reference.pulse(samplesPerPulse + 1:end)) > ...
            waveformTolerance)
        throw_invalid_reference_semantics();
    end

    for pulseIndex = 1:numPulses
        blockStart = (pulseIndex - 1) * samplesPerPri + 1;
        blockStop = pulseIndex * samplesPerPri;
        if max(abs(reference.train(blockStart:blockStop) - ...
                reference.pulse)) > waveformTolerance
            throw_invalid_reference_semantics();
        end
        expectedTimeBlock_s = (blockStart - 1:blockStop - 1) / fs_Hz;
        if max(abs(reference.time_train_s(blockStart:blockStop) - ...
                expectedTimeBlock_s)) > timeTolerance_s
            throw_invalid_reference_semantics();
        end
    end
end

function valid = is_valid_signal_row(value, expectedLength)
%IS_VALID_SIGNAL_ROW 判断参考波形是否为指定长度的有限数值行向量。

    valid = isnumeric(value) && isrow(value) && ...
        numel(value) == expectedLength && ...
        all(isfinite(real(value))) && all(isfinite(imag(value)));
end

function valid = is_valid_time_row(value, expectedLength)
%IS_VALID_TIME_ROW 判断时间轴是否为指定长度的有限实数行向量。

    valid = isnumeric(value) && isreal(value) && isrow(value) && ...
        numel(value) == expectedLength && all(isfinite(value));
end

function code = normalize_code(code)
%NORMALIZE_CODE 将合法标量文本规范为去除首尾空格的字符行向量。

    if isstring(code) && isscalar(code) && ~ismissing(code)
        code = char(code);
    elseif ~(ischar(code) && isrow(code))
        throw_unknown_type('<malformed>');
    end
    code = strtrim(code);
    if isempty(code)
        throw_unknown_type('<empty>');
    end
end

function validate_registry(registry)
%VALIDATE_REGISTRY 校验注册表结构、代码唯一性、句柄和参数名元数据。

    requiredFields = {'code', 'generator', 'parameter_names'};
    if ~isstruct(registry) || isempty(registry) || ...
            ~all(isfield(registry, requiredFields))
        throw_invalid_registry();
    end

    codes = cell(1, numel(registry));
    for entryIndex = 1:numel(registry)
        entry = registry(entryIndex);
        try
            codes{entryIndex} = normalize_registry_code(entry.code);
        catch
            throw_invalid_registry();
        end
        if ~isa(entry.generator, 'function_handle') || ...
                ~isscalar(entry.generator) || ...
                ~iscell(entry.parameter_names) || ...
                ~isrow(entry.parameter_names) || ...
                isempty(entry.parameter_names) || ...
                ~all(cellfun(@is_valid_parameter_name, ...
                entry.parameter_names)) || ...
                numel(unique(entry.parameter_names, 'stable')) ~= ...
                numel(entry.parameter_names)
            throw_invalid_registry();
        end
    end
    if numel(unique(codes, 'stable')) ~= numel(codes)
        throw_invalid_registry();
    end
end

function code = normalize_registry_code(code)
%NORMALIZE_REGISTRY_CODE 规范注册表代码，非法值由上层归为注册表错误。

    if isstring(code) && isscalar(code) && ~ismissing(code)
        code = char(code);
    elseif ~(ischar(code) && isrow(code))
        error('JammingGenerator:InvalidRegistryCode', ...
            '注册表 code 必须是标量文本。');
    end
    code = strtrim(code);
    if isempty(code)
        error('JammingGenerator:InvalidRegistryCode', ...
            '注册表 code 不能为空。');
    end
end

function valid = is_valid_parameter_name(name)
%IS_VALID_PARAMETER_NAME 判断参数名是否为非空字符行向量。

    valid = ischar(name) && isrow(name) && ~isempty(strtrim(name));
end

function signal = validate_signal(signal, expectedLength, code)
%VALIDATE_SIGNAL 校验并规范生成信号的长度、数值域、形状与复数类型。

    if ~isnumeric(signal) || ~isvector(signal) || ...
            numel(signal) ~= expectedLength || ...
            ~all(isfinite(real(signal))) || ...
            ~all(isfinite(imag(signal))) || ...
            ~any(real(signal) ~= 0 | imag(signal) ~= 0)
        throw_invalid_output(code);
    end
    signal = double(signal(:).');
    signal = complex(real(signal), imag(signal));
end

function validate_params(params, parameterNames, code)
%VALIDATE_PARAMS 校验真值字段精确集合及有限标量或合法 NaN 填充值。

    if ~isstruct(params) || ~isscalar(params)
        throw_invalid_output(code);
    end
    actualNames = fieldnames(params);
    expectedNames = parameterNames(:);
    if numel(actualNames) ~= numel(expectedNames) || ...
            ~isempty(setxor(actualNames, expectedNames))
        error('JammingGenerator:InvalidGeneratorOutput', ...
            ['类型 %s 的 params 字段集合必须与 registry.parameter_names ' ...
            '精确一致，不得缺失或包含额外字段；字段顺序不作要求。'], code);
    end
    for parameterIndex = 1:numel(parameterNames)
        parameterName = parameterNames{parameterIndex};
        value = params.(parameterName);
        validFinite = isnumeric(value) && isreal(value) && ...
            isscalar(value) && isfinite(value);
        validPadding = isnumeric(value) && isreal(value) && ...
            isscalar(value) && isnan(value) && ...
            allows_nan_padding(code, parameterName);
        if ~(validFinite || validPadding)
            throw_invalid_output(code);
        end
    end
end

function allowed = allows_nan_padding(code, parameterName)
%ALLOWS_NAN_PADDING 只允许各类型明确定义的固定空槽使用 NaN。

    numberedDelay = ~isempty(regexp(parameterName, ...
        '^delay_[0-9][0-9]_s$', 'once'));
    numberedRange = ~isempty(regexp(parameterName, ...
        '^range_[0-9][0-9]_m$', 'once'));
    numberedAmplitude = ~isempty(regexp(parameterName, ...
        '^amplitude_[0-9][0-9]$', 'once'));
    numberedFrequency = ~isempty(regexp(parameterName, ...
        '^frequency_[0-9][0-9]_Hz$', 'once'));
    numberedPhase = ~isempty(regexp(parameterName, ...
        '^phase_[0-9][0-9]_rad$', 'once'));
    numberedStart = ~isempty(regexp(parameterName, ...
        '^start_[0-9][0-9]_s$', 'once'));
    numberedWidth = ~isempty(regexp(parameterName, ...
        '^width_[0-9][0-9]_s$', 'once'));
    numberedSubpulseStart = ~isempty(regexp(parameterName, ...
        '^subpulse_start_[0-9][0-9]_s$', 'once'));
    if strcmp(code, 'MFT')
        allowed = strcmp(parameterName, 'mean_spacing_s') || ...
            strcmp(parameterName, 'arrival_delay_s') || ...
            numberedDelay || numberedAmplitude;
    elseif strcmp(code, 'RangeFakeTarget')
        allowed = strcmp(parameterName, 'arrival_delay_s') || ...
            numberedDelay || numberedRange || numberedAmplitude;
    elseif strcmp(code, 'CombSpectrum')
        allowed = numberedFrequency || numberedAmplitude || numberedPhase;
    elseif strcmp(code, 'ClutterPulse')
        allowed = numberedStart || numberedWidth || numberedAmplitude || ...
            numberedFrequency || numberedPhase;
    elseif strcmp(code, 'SpectralDispersion')
        allowed = numberedSubpulseStart;
    else
        allowed = false;
    end
end

function throw_unknown_type(code)
%THROW_UNKNOWN_TYPE 抛出类型代码的统一错误标识。

    error('JammingGenerator:UnknownType', ...
        '未知或畸形的干扰类型代码 "%s"。', code);
end

function throw_invalid_registry()
%THROW_INVALID_REGISTRY 抛出注册表结构的统一错误标识。

    error('JammingGenerator:InvalidRegistry', ...
        'registry 必须包含唯一 code、有效生成器句柄和合法参数名列表。');
end

function throw_invalid_output(code)
%THROW_INVALID_OUTPUT 抛出生成器输出契约的统一错误标识。

    error('JammingGenerator:InvalidGeneratorOutput', ...
        '类型 %s 的生成信号或真值参数不满足统一输出契约。', code);
end

function throw_invalid_reference()
%THROW_INVALID_REFERENCE 抛出外部 reference 契约的统一错误标识。

    error('JammingGenerator:InvalidReference', ...
        ['reference 必须包含与 cfg 派生长度一致的有限 pulse、train、' ...
        'time_pri_s 和 time_train_s 行向量。']);
end

function throw_invalid_reference_semantics()
%THROW_INVALID_REFERENCE_SEMANTICS 抛出参考内容与配置不一致的契约错误。

    error('JammingGenerator:InvalidGeneratorOutput', ...
        ['reference 的时间轴、LFM 脉冲或逐 PRI 脉冲串与 cfg ' ...
        '语义不一致。']);
end
