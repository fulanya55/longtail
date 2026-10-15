function cfg = normalize_and_validate_config(cfg, scenario)
%NORMALIZE_AND_VALIDATE_CONFIG 标准化并校验组合干扰数据生成配置。
%   CFG = NORMALIZE_AND_VALIDATE_CONFIG(CFG, SCENARIO) 接收默认配置结构体
%   CFG 和场景 SCENARIO。SCENARIO 仅允许 'balanced'、'imbalanced' 或
%   'openset'。函数返回保留原有字段的标准化 CFG，并将每 PRI 样点数、
%   每脉冲样点数、总样点长度、调频斜率和预计输出字节数统一写入
%   cfg.derived，不向 cfg.radar 写入派生字段。
%
%   单位约定：雷达频率为 Hz，时间为 s，传播速度为 m/s，功率范围为 dB；
%   输出估算按每个 complex double 样点 16 字节、legacy 实数样点 8 字节
%   计算。函数不写文件、不修改输入变量，也不改变随机数状态。
%
%   示例：
%       cfg = create_default_config();
%       cfg = normalize_and_validate_config(cfg, 'balanced');
%       fprintf('预计输出 %.3f GiB\n', ...
%           cfg.derived.estimated_output_bytes / 1024^3);

    scenario = normalize_scenario(scenario);
    cfg = normalize_boolean_fields(cfg);
    cfg.output.max_estimated_bytes = normalize_output_capacity( ...
        cfg.output.max_estimated_bytes);

    registry = build_jamming_registry(cfg);
    validCodes = {registry.code};
    cfg.types.enabled = normalize_type_list(cfg.types.enabled, 'enabled');
    cfg.types.known = normalize_type_list(cfg.types.known, 'known');
    cfg.types.unknown = normalize_type_list(cfg.types.unknown, 'unknown');

    validate_type_list(cfg.types.enabled, 'enabled', validCodes);
    validate_type_list(cfg.types.known, 'known', validCodes);
    validate_type_list(cfg.types.unknown, 'unknown', validCodes);
    validate_type_subsets(cfg.types);
    validate_open_set_type_partition(cfg.types, scenario);

    cfg.radar = normalize_and_validate_radar(cfg.radar);
    cfg.derived.samples_per_pri = round(cfg.radar.pri_s * cfg.radar.fs_Hz);
    cfg.derived.samples_per_pulse = round( ...
        cfg.radar.pulse_width_s * cfg.radar.fs_Hz);
    cfg.derived.sample_length = cfg.derived.samples_per_pri * ...
        cfg.radar.num_pulses;
    cfg.derived.chirp_rate_Hz_per_s = cfg.radar.bandwidth_Hz / ...
        cfg.radar.pulse_width_s;
    if ~is_valid_positive_integer(cfg.derived.samples_per_pri) || ...
            ~is_valid_positive_integer(cfg.derived.samples_per_pulse) || ...
            ~is_valid_positive_integer(cfg.derived.sample_length)
        error('JammingConfig:InvalidRadar', ...
            ['采样率与时间参数舍入后，samples_per_pri、' ...
            'samples_per_pulse 和 sample_length 必须均为正整数。']);
    end

    validate_parameter_ranges(cfg.params, 'cfg.params');
    validate_integer_parameters(cfg.params, 'cfg.params');
    validate_physical_parameters(cfg.params, cfg.radar, ...
        cfg.types.enabled, cfg.derived);
    validate_target_capacities(cfg.params);
    validate_traction(cfg.types.enabled, cfg.radar.num_pulses);
    cfg.dataset = normalize_and_validate_dataset( ...
        cfg.dataset, numel(cfg.types.enabled));
    cfg.imbalance = normalize_and_validate_imbalance(cfg.imbalance, ...
        cfg.dataset.min_components, cfg.dataset.max_components, ...
        cfg.types.enabled);
    cfg.openset = normalize_and_validate_open_set(cfg.openset);

    if strcmp(scenario, 'openset')
        sampleCount = cfg.openset.train_known + ...
            cfg.openset.calibration_known + ...
            cfg.openset.validation_mixed + cfg.openset.test_mixed;
    else
        sampleCount = cfg.dataset.num_samples;
    end

    estimatedBytes = sampleCount * cfg.derived.sample_length * 16;
    if cfg.output.save_legacy
        estimatedBytes = estimatedBytes + ...
            sampleCount * cfg.derived.sample_length * 8;
    end
    if estimatedBytes > cfg.output.max_estimated_bytes
        error('JammingConfig:EstimatedSizeExceeded', ...
            ['预计输出大小为 %.0f 字节，超过 cfg.output.' ...
            'max_estimated_bytes=%.0f 字节。'], estimatedBytes, ...
            cfg.output.max_estimated_bytes);
    end
    cfg.derived.estimated_output_bytes = estimatedBytes;
end

function scenario = normalize_scenario(scenario)
%NORMALIZE_SCENARIO 将场景名称规范为字符行向量。

    if isstring(scenario) && isscalar(scenario) && ~ismissing(scenario)
        scenario = char(scenario);
    elseif ~(ischar(scenario) && isrow(scenario))
        error('JammingConfig:UnknownScenario', ...
            '场景必须是 balanced、imbalanced 或 openset 中的一个名称。');
    end
    scenario = strtrim(scenario);
    if ~ismember(scenario, {'balanced', 'imbalanced', 'openset'})
        error('JammingConfig:UnknownScenario', ...
            '未知场景 "%s"；仅支持 balanced、imbalanced 和 openset。', ...
            scenario);
    end
end

function cfg = normalize_boolean_fields(cfg)
%NORMALIZE_BOOLEAN_FIELDS 校验并统一六个配置开关为 logical 标量。

    cfg.output.save_v73 = normalize_boolean( ...
        cfg.output.save_v73, 'cfg.output.save_v73');
    cfg.output.save_legacy = normalize_boolean( ...
        cfg.output.save_legacy, 'cfg.output.save_legacy');
    cfg.radar.add_target = normalize_boolean( ...
        cfg.radar.add_target, 'cfg.radar.add_target');
    cfg.imbalance.component.enable = normalize_boolean( ...
        cfg.imbalance.component.enable, ...
        'cfg.imbalance.component.enable');
    cfg.imbalance.combination.enable = normalize_boolean( ...
        cfg.imbalance.combination.enable, ...
        'cfg.imbalance.combination.enable');
    cfg.imbalance.power.enable = normalize_boolean( ...
        cfg.imbalance.power.enable, 'cfg.imbalance.power.enable');
end

function value = normalize_boolean(value, path)
%NORMALIZE_BOOLEAN 将 logical 或数值 0/1 标量规范为 logical。

    validLogical = islogical(value) && isscalar(value);
    validNumeric = isnumeric(value) && isreal(value) && isscalar(value) && ...
        isfinite(value) && (value == 0 || value == 1);
    if ~(validLogical || validNumeric)
        error('JammingConfig:InvalidBoolean', ...
            '%s 必须是 logical 标量或数值 0/1。', path);
    end
    value = logical(value);
end

function value = normalize_output_capacity(value)
%NORMALIZE_OUTPUT_CAPACITY 校验并将输出字节上限转换为 double。

    if ~is_valid_nonnegative_scalar(value)
        error('JammingConfig:EstimatedSizeExceeded', ...
            ['cfg.output.max_estimated_bytes 必须是有限非负标量，' ...
            '且不能超过 flintmax。']);
    end
    value = double(value);
end

function types = normalize_type_list(types, fieldName)
%NORMALIZE_TYPE_LIST 将 string 或 cellstr 类型列表规范为行 cellstr。

    if isstring(types)
        if any(ismissing(types), 'all')
            error('JammingConfig:UnknownType', ...
                'cfg.types.%s 含缺失的干扰类型名称。', fieldName);
        end
        types = cellstr(types(:).');
    elseif iscellstr(types)
        types = reshape(types, 1, []);
    elseif iscell(types) && isempty(types)
        types = cell(1, 0);
    else
        error('JammingConfig:UnknownType', ...
            'cfg.types.%s 必须是 string 数组或 cellstr。', fieldName);
    end
    types = cellfun(@strtrim, types, 'UniformOutput', false);
end

function validate_type_list(types, fieldName, validCodes)
%VALIDATE_TYPE_LIST 检查单个类型列表的重复项和注册表归属。

    if numel(unique(types, 'stable')) ~= numel(types)
        error('JammingConfig:DuplicateType', ...
            'cfg.types.%s 含重复的干扰类型。', fieldName);
    end
    unknownTypes = setdiff(types, validCodes, 'stable');
    if ~isempty(unknownTypes)
        error('JammingConfig:UnknownType', ...
            'cfg.types.%s 含注册表未定义的类型：%s。', fieldName, ...
            strjoin(unknownTypes, ', '));
    end
end

function validate_type_subsets(types)
%VALIDATE_TYPE_SUBSETS 检查 known 和 unknown 是否属于 enabled。

    missingKnown = setdiff(types.known, types.enabled, 'stable');
    missingUnknown = setdiff(types.unknown, types.enabled, 'stable');
    missing = [missingKnown, missingUnknown];
    if ~isempty(missing)
        error('JammingConfig:UnknownType', ...
            'known/unknown 中的类型必须同时列入 enabled：%s。', ...
            strjoin(unique(missing, 'stable'), ', '));
    end
end

function validate_open_set_type_partition(types, scenario)
%VALIDATE_OPEN_SET_TYPE_PARTITION 检查开放集类别非空和互斥条件。

    if strcmp(scenario, 'openset') && isempty(types.known)
        error('JammingConfig:OpenSetRequiresKnown', ...
            'openset 场景至少需要一个 known 类型。');
    end
    if strcmp(scenario, 'openset') && isempty(types.unknown)
        error('JammingConfig:OpenSetRequiresUnknown', ...
            'openset 场景至少需要一个 unknown 类型。');
    end
    overlap = intersect(types.known, types.unknown, 'stable');
    if ~isempty(overlap)
        error('JammingConfig:KnownUnknownOverlap', ...
            'known 与 unknown 不能重叠；重复类型：%s。', ...
            strjoin(overlap, ', '));
    end
end

function radar = normalize_and_validate_radar(radar)
%NORMALIZE_AND_VALIDATE_RADAR 校验雷达参数并将数值标量转换为 double。

    positiveFields = {'c_mps', 'fs_Hz', 'pulse_width_s', 'pri_s', ...
        'bandwidth_Hz', 'carrier_Hz', 'num_pulses', 'noise_power'};
    for index = 1:numel(positiveFields)
        fieldName = positiveFields{index};
        if ~is_valid_positive_scalar(radar.(fieldName))
            error('JammingConfig:InvalidRadar', ...
                ['cfg.radar.%s 必须是有限正数，且不能超过 ' ...
                'flintmax。'], fieldName);
        end
    end
    if radar.num_pulses ~= fix(radar.num_pulses)
        error('JammingConfig:InvalidRadar', ...
            'cfg.radar.num_pulses 必须是正整数。');
    end
    if radar.pulse_width_s > radar.pri_s
        error('JammingConfig:InvalidRadar', ...
            '脉宽 pulse_width_s 不能大于脉冲重复间隔 pri_s。');
    end

    if ~is_valid_nonnegative_scalar(radar.target_range_m)
        error('JammingConfig:InvalidRadar', ...
            ['cfg.radar.target_range_m 必须是有限非负数，且不能超过 ' ...
            'flintmax。']);
    end
    if ~is_double_exact_scalar(radar.target_SNR_dB)
        error('JammingConfig:InvalidRadar', ...
            ['cfg.radar.target_SNR_dB 必须是有限实数标量，且绝对值不能' ...
            '超过 flintmax。']);
    end

    numericFields = [positiveFields, {'target_SNR_dB', 'target_range_m'}];
    for index = 1:numel(numericFields)
        fieldName = numericFields{index};
        radar.(fieldName) = double(radar.(fieldName));
    end
end

function validate_parameter_ranges(value, path)
%VALIDATE_PARAMETER_RANGES 递归检查普通参数均为 1×2 数值范围。

    if ~isstruct(value)
        return;
    end
    fields = fieldnames(value);
    for itemIndex = 1:numel(value)
        for fieldIndex = 1:numel(fields)
            fieldName = fields{fieldIndex};
            child = value(itemIndex).(fieldName);
            childPath = sprintf('%s.%s', path, fieldName);
            if isstruct(child)
                validate_parameter_ranges(child, childPath);
            elseif ismember(fieldName, ...
                    {'num_targets', 'num_replicas', 'max_targets'})
                % 整数计数和容量字段由专用整数校验器给出稳定错误标识。
            elseif strcmp(fieldName, 'mode')
                if ~((ischar(child) && isrow(child)) || ...
                        (isstring(child) && isscalar(child) && ...
                        ~ismissing(child)))
                    error('JammingConfig:InvalidParameterRange', ...
                        '%s 必须是字符行向量或非缺失 string 标量。', childPath);
                end
            else
                validate_numeric_range(child, childPath);
            end
        end
    end
end

function validate_numeric_range(value, path)
%VALIDATE_NUMERIC_RANGE 检查一个数值范围有限且下界不大于上界。

    if ~isnumeric(value) || ~isreal(value) || ...
            ~isequal(size(value), [1, 2]) || ...
            ~all(isfinite(value)) || value(1) > value(2)
        error('JammingConfig:InvalidParameterRange', ...
            '%s 必须是有限的 1×2 范围，且下界不能大于上界。', path);
    end
end

function validate_integer_parameters(value, path)
%VALIDATE_INTEGER_PARAMETERS 递归检查目标数、副本数和容量字段。

    if ~isstruct(value)
        return;
    end
    fields = fieldnames(value);
    for itemIndex = 1:numel(value)
        for fieldIndex = 1:numel(fields)
            fieldName = fields{fieldIndex};
            child = value(itemIndex).(fieldName);
            childPath = sprintf('%s.%s', path, fieldName);
            if isstruct(child)
                validate_integer_parameters(child, childPath);
            elseif ismember(fieldName, {'num_targets', 'num_replicas'})
                if ~is_positive_integer_range(child)
                    error('JammingConfig:InvalidIntegerRange', ...
                        '%s 必须是递增的 1×2 正整数范围。', childPath);
                end
            elseif strcmp(fieldName, 'max_targets')
                if ~is_valid_positive_integer(child)
                    error('JammingConfig:InvalidIntegerRange', ...
                        '%s 必须是有限正整数。', childPath);
                end
            end
        end
    end
end

function validate_target_capacities(params)
%VALIDATE_TARGET_CAPACITIES 检查可变长度参数固定容量及目标数上界。

    owners = {'MFT', 'RangeFakeTarget'};
    capacities = [8, 4];
    for index = 1:numel(owners)
        owner = owners{index};
        ownerParams = params.(owner);
        capacity = capacities(index);
        if ownerParams.max_targets ~= capacity || ...
                ownerParams.num_targets(2) > capacity
            error('JammingConfig:InvalidIntegerRange', ...
                ['cfg.params.%s.max_targets 必须固定为 %d，且 ' ...
                'num_targets 上界不能超过该容量。'], owner, capacity);
        end
    end
end

function validate_physical_parameters(params, radar, enabledTypes, derived)
%VALIDATE_PHYSICAL_PARAMETERS 仅校验已启用类型实际使用的物理定义域。

    commonUsers = setdiff(enabledTypes, {'Traction'}, 'stable');
    if ~isempty(commonUsers)
        validate_delay_range(params.common.arrival_delay_s, radar.pri_s, ...
            'cfg.params.common.arrival_delay_s');
    end

    nyquistGuard_Hz = radar.fs_Hz / 2 - max( ...
        eps(radar.fs_Hz / 2), radar.fs_Hz * 1e-12);
    if ismember('FMB', enabledTypes)
        validate_positive_range(params.FMB.bandwidth_factor, ...
            'cfg.params.FMB.bandwidth_factor');
        validate_strict_frequency_range(params.FMB.center_frequency_Hz, ...
            nyquistGuard_Hz, 'cfg.params.FMB.center_frequency_Hz');
        validate_discrete_noise_bandwidth(params.FMB, radar, derived, ...
            nyquistGuard_Hz, 'cfg.params.FMB');
    end
    if ismember('AMA', enabledTypes)
        validate_positive_range(params.AMA.bandwidth_factor, ...
            'cfg.params.AMA.bandwidth_factor');
        validate_strict_frequency_range(params.AMA.center_frequency_Hz, ...
            nyquistGuard_Hz, 'cfg.params.AMA.center_frequency_Hz');
        validate_discrete_noise_bandwidth(params.AMA, radar, derived, ...
            nyquistGuard_Hz, 'cfg.params.AMA');
    end
    if ismember('SawFM', enabledTypes)
        validate_positive_range(params.SawFM.sweep_bandwidth_Hz, ...
            'cfg.params.SawFM.sweep_bandwidth_Hz');
        validate_positive_range(params.SawFM.sweep_period_s, ...
            'cfg.params.SawFM.sweep_period_s');
    end
    if ismember('SinFM', enabledTypes)
        validate_nonnegative_range(params.SinFM.frequency_deviation_Hz, ...
            'cfg.params.SinFM.frequency_deviation_Hz');
        validate_positive_range(params.SinFM.modulation_frequency_Hz, ...
            'cfg.params.SinFM.modulation_frequency_Hz');
        centerGuard_Hz = nyquistGuard_Hz - ...
            params.SinFM.frequency_deviation_Hz(1);
        if centerGuard_Hz < 0
            error('JammingConfig:InvalidPhysicalParameter', ...
                ['cfg.params.SinFM.frequency_deviation_Hz 下界没有严格 ' ...
                'Nyquist 支持。']);
        end
        validate_frequency_intersection( ...
            params.SinFM.center_frequency_Hz, centerGuard_Hz, ...
            'cfg.params.SinFM.center_frequency_Hz');
    end
    if ismember('SN', enabledTypes)
        validate_positive_range(params.SN.sample_width_s, ...
            'cfg.params.SN.sample_width_s');
        validate_replica_capacity(params.SN, radar.fs_Hz, ...
            derived.sample_length, 'cfg.params.SN');
    end
    if ismember('MFT', enabledTypes)
        validate_delay_range(params.MFT.target_delay_s, radar.pri_s, ...
            'cfg.params.MFT.target_delay_s');
        validate_positive_range(params.MFT.target_relative_amplitude, ...
            'cfg.params.MFT.target_relative_amplitude');
        validate_target_delay_identifiability(params.MFT, ...
            params.common.arrival_delay_s, radar, derived, ...
            'cfg.params.MFT');
    end
    if ismember('VFT', enabledTypes)
        [referenceMinimum_Hz, referenceMaximum_Hz] = ...
            reference_lfm_frequency_support( ...
            derived.samples_per_pulse, derived.chirp_rate_Hz_per_s, ...
            radar.fs_Hz);
        vftSupportedRange_Hz = [ ...
            -nyquistGuard_Hz - referenceMinimum_Hz, ...
            nyquistGuard_Hz - referenceMaximum_Hz];
        validate_range_intersection(params.VFT.doppler_Hz, ...
            vftSupportedRange_Hz, 'cfg.params.VFT.doppler_Hz');
    end
    if ismember('ISRJ', enabledTypes)
        validate_positive_range(params.ISRJ.sample_width_s, ...
            'cfg.params.ISRJ.sample_width_s');
        validate_replica_capacity(params.ISRJ, radar.fs_Hz, ...
            derived.sample_length, 'cfg.params.ISRJ');
    end
    if ismember('RangeFakeTarget', enabledTypes)
        validate_delay_range(params.RangeFakeTarget.target_delay_s, ...
            radar.pri_s, 'cfg.params.RangeFakeTarget.target_delay_s');
        validate_positive_range( ...
            params.RangeFakeTarget.target_relative_amplitude, ...
            'cfg.params.RangeFakeTarget.target_relative_amplitude');
        validate_target_delay_identifiability(params.RangeFakeTarget, ...
            params.common.arrival_delay_s, radar, derived, ...
            'cfg.params.RangeFakeTarget');
    end
    if ismember('Traction', enabledTypes)
        validate_nonnegative_range(params.Traction.initial_range_m, ...
            'cfg.params.Traction.initial_range_m');
        validate_positive_range(params.Traction.amplitude, ...
            'cfg.params.Traction.amplitude');
        validate_traction_trajectory(params.Traction, radar);
    end
end

function validate_discrete_noise_bandwidth(params, radar, derived, ...
        nyquistGuard_Hz, path)
%VALIDATE_DISCRETE_NOISE_BANDWIDTH 预检最坏情况下的非零共轭 FFT 支持。
%   生成器以完整记录长度执行 FFT 低通。对最小请求带宽和中心频率范围中
%   绝对值最大的端点同时取最坏情况，裁剪后的双边带宽必须至少为两个
%   FFT 频率间隔，才能选中 +df 与 -df，而不是只保留随后会被去除的 DC。

    frequencyStep_Hz = radar.fs_Hz / derived.sample_length;
    minimumRequestedBandwidth_Hz = params.bandwidth_factor(1) * ...
        radar.bandwidth_Hz;
    maximumCenterMagnitude_Hz = max(abs(params.center_frequency_Hz));
    worstAvailableBandwidth_Hz = 2 * (nyquistGuard_Hz - ...
        maximumCenterMagnitude_Hz);
    worstRealizableBandwidth_Hz = min( ...
        minimumRequestedBandwidth_Hz, worstAvailableBandwidth_Hz);
    if ~isfinite(worstRealizableBandwidth_Hz) || ...
            worstRealizableBandwidth_Hz < 2 * frequencyStep_Hz
        error('JammingConfig:InvalidPhysicalParameter', ...
            ['%s 在最小带宽和最不利中心频率下必须至少覆盖一对非零 ' ...
            '共轭 FFT 频点（离散双边带宽 >= 2*fs/sample_length）。'], path);
    end
end

function validate_target_delay_identifiability(params, arrivalRange_s, ...
        radar, derived, path)
%VALIDATE_TARGET_DELAY_IDENTIFIABILITY 预检假目标时延槽和 PRI 内完整支持。
%   假目标相对时延在整数样点域无放回选择，因此量化后可用槽数至少应
%   覆盖 num_targets 上界；最坏相对时延与公共到达时延相加后，完整脉冲
%   仍须位于同一 PRI 内，避免循环移位造成回绕或截断。

    minimumDelaySamples = round(params.target_delay_s(1) * radar.fs_Hz);
    maximumDelaySamples = round(params.target_delay_s(2) * radar.fs_Hz);
    availableSlotCount = maximumDelaySamples - minimumDelaySamples + 1;
    if availableSlotCount < params.num_targets(2)
        error('JammingConfig:InvalidPhysicalParameter', ...
            ['%s 的量化 target_delay_s 仅有 %d 个唯一样点槽，少于 ' ...
            'num_targets 上界 %d。'], path, availableSlotCount, ...
            params.num_targets(2));
    end

    worstEnd_s = params.target_delay_s(2) + arrivalRange_s(2) + ...
        radar.pulse_width_s;
    maximumArrivalSamples = round(arrivalRange_s(2) * radar.fs_Hz);
    worstEndSamples = maximumDelaySamples + maximumArrivalSamples + ...
        derived.samples_per_pulse;
    tolerance_s = 32 * eps(max([1, radar.pri_s, worstEnd_s]));
    if worstEnd_s > radar.pri_s + tolerance_s || ...
            worstEndSamples > derived.samples_per_pri
        error('JammingConfig:InvalidPhysicalParameter', ...
            ['%s 的最大相对时延、最大公共到达时延与脉宽之和不得超过 ' ...
            'PRI，且量化后完整脉冲也必须位于当前 PRI 内。'], path);
    end
end

function validate_strict_frequency_range(range, guard_Hz, path)
%VALIDATE_STRICT_FREQUENCY_RANGE 要求整个范围严格位于对称 guard 内。

    if guard_Hz <= 0 || range(1) <= -guard_Hz || range(2) >= guard_Hz
        error('JammingConfig:InvalidPhysicalParameter', ...
            '%s 的所有可能值必须严格满足 abs(value)<Nyquist guard。', path);
    end
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

function validate_range_intersection(range, supportedRange, path)
%VALIDATE_RANGE_INTERSECTION 要求配置范围与给定物理支持存在交集。

    if supportedRange(1) > supportedRange(2) || ...
            range(1) > supportedRange(2) || range(2) < supportedRange(1)
        error('JammingConfig:InvalidPhysicalParameter', ...
            '%s 与参考波形的严格无混叠支持域没有交集。', path);
    end
end

function validate_frequency_intersection(range, guard_Hz, path)
%VALIDATE_FREQUENCY_INTERSECTION 要求配置频率范围存在严格域内交集。

    if guard_Hz < 0 || range(1) > guard_Hz || range(2) < -guard_Hz
        error('JammingConfig:InvalidPhysicalParameter', ...
            '%s 必须与严格 Nyquist 频率域相交。', path);
    end
end

function validate_replica_capacity(params, fs_Hz, sampleLength, path)
%VALIDATE_REPLICA_CAPACITY 保证最坏基段和全部副本完整位于记录内。

    maxWidthSamples = max(1, round(params.sample_width_s(2) * fs_Hz));
    maxReplica = params.num_replicas(2);
    if (maxReplica + 1) * maxWidthSamples > sampleLength
        error('JammingConfig:InvalidPhysicalParameter', ...
            ['%s 的最大基段加全部副本必须完整位于 cfg.derived.' ...
            'sample_length 内。'], path);
    end
end

function validate_traction_trajectory(params, radar)
%VALIDATE_TRACTION_TRAJECTORY 预检距离/速度范围的最坏脉冲轨迹。

    finalPulseTime_s = (radar.num_pulses - 1) * radar.pri_s;
    minimumRange_m = params.initial_range_m(1) + min( ...
        0, params.velocity_mps(1) * finalPulseTime_s);
    maximumRange_m = params.initial_range_m(2) + max( ...
        0, params.velocity_mps(2) * finalPulseTime_s);
    minimumDelay_s = 2 * minimumRange_m / radar.c_mps;
    maximumDelay_s = 2 * maximumRange_m / radar.c_mps;
    upperTolerance_s = 32 * eps(max([radar.pri_s, ...
        radar.pulse_width_s, abs(maximumDelay_s)]));
    if minimumDelay_s < 0 || maximumDelay_s + ...
            radar.pulse_width_s > radar.pri_s + upperTolerance_s
        error('JammingConfig:InvalidPhysicalParameter', ...
            ['cfg.params.Traction 的最坏轨迹必须满足 delay_s>=0 且 ' ...
            'delay_s+pulse_width_s<=pri_s（上界容差 32 eps）。']);
    end
end

function validate_delay_range(range, pri_s, path)
%VALIDATE_DELAY_RANGE 要求时延范围非负且严格小于单个 PRI。

    if range(1) < 0 || range(2) >= pri_s
        error('JammingConfig:InvalidPhysicalParameter', ...
            '%s 必须满足下界 >= 0 且上界 < cfg.radar.pri_s。', path);
    end
end

function validate_positive_range(range, path)
%VALIDATE_POSITIVE_RANGE 要求整个连续参数范围严格为正。

    if range(1) <= 0
        error('JammingConfig:InvalidPhysicalParameter', ...
            '%s 的下界必须严格大于 0。', path);
    end
end

function validate_nonnegative_range(range, path)
%VALIDATE_NONNEGATIVE_RANGE 要求整个连续参数范围非负。

    if range(1) < 0
        error('JammingConfig:InvalidPhysicalParameter', ...
            '%s 的下界必须大于或等于 0。', path);
    end
end

function validate_traction(enabledTypes, numPulses)
%VALIDATE_TRACTION 检查拖引干扰所需的最小脉冲数。

    if ismember('Traction', enabledTypes) && numPulses < 2
        error('JammingConfig:TractionRequiresMultiplePulses', ...
            '启用 Traction 时 cfg.radar.num_pulses 必须至少为 2。');
    end
end

function dataset = normalize_and_validate_dataset(dataset, enabledCount)
%NORMALIZE_AND_VALIDATE_DATASET 校验数据集计数并转换为 double。

    countFields = {'num_samples', 'samples_per_pattern'};
    for index = 1:numel(countFields)
        fieldName = countFields{index};
        if ~is_valid_positive_integer(dataset.(fieldName))
            error('JammingConfig:InvalidDatasetCount', ...
                ['cfg.dataset.%s 必须是有限正整数，且不能超过 ' ...
                'flintmax。'], fieldName);
        end
    end

    minComponents = dataset.min_components;
    maxComponents = dataset.max_components;
    if ~is_valid_positive_integer(minComponents) || ...
            ~is_valid_positive_integer(maxComponents) || ...
            minComponents > maxComponents || maxComponents > enabledCount
        error('JammingConfig:InvalidComponentCount', ...
            ['min_components/max_components 必须是正整数，满足 min<=max，' ...
            '且 max 不能超过 enabled 类型数量 %d。'], enabledCount);
    end

    numericFields = [countFields, {'min_components', 'max_components'}];
    for index = 1:numel(numericFields)
        fieldName = numericFields{index};
        dataset.(fieldName) = double(dataset.(fieldName));
    end
end

function imbalance = normalize_and_validate_imbalance(imbalance, ...
        minComponents, maxComponents, enabledTypes)
%NORMALIZE_AND_VALIDATE_IMBALANCE 按启用开关标准化或校验不平衡参数。

    enabledCount = numel(enabledTypes);
    if imbalance.component.enable
        componentWeights = imbalance.component.weights;
        if ~isnumeric(componentWeights) || ~isreal(componentWeights) || ...
                ~isvector(componentWeights)
            error('JammingConfig:InvalidImbalanceWeights', ...
                'component.weights 必须是实数数值向量。');
        end
        componentWeights = reshape(double(componentWeights), 1, []);
        imbalance.component.weights = componentWeights;
        if numel(componentWeights) ~= enabledCount || ...
                any(~isfinite(componentWeights)) || ...
                any(componentWeights < 0) || ~any(componentWeights > 0)
            error('JammingConfig:InvalidImbalanceWeights', ...
                ['启用 component imbalance 时，weights 必须含 %d 个有限' ...
                '非负权重，且至少一个权重大于 0。'], enabledCount);
        end
    else
        % 关闭抽样偏置时，使用与 enabled 一一对应的等权行向量。
        imbalance.component.weights = ones(1, enabledCount);
    end

    if imbalance.combination.enable
        cardinalityWeights = imbalance.combination.cardinality_weights;
        if ~isnumeric(cardinalityWeights) || ...
                ~isreal(cardinalityWeights) || ...
                ~isvector(cardinalityWeights)
            error('JammingConfig:InvalidImbalanceWeights', ...
                'cardinality_weights 必须是实数数值向量。');
        end
        cardinalityWeights = reshape(double(cardinalityWeights), 1, []);
        imbalance.combination.cardinality_weights = cardinalityWeights;
        if numel(cardinalityWeights) < maxComponents
            error('JammingConfig:InvalidImbalanceWeights', ...
                ['启用 combination imbalance 时，cardinality_weights ' ...
                '长度必须至少覆盖 max_components=%d。'], maxComponents);
        end
        usedCardinalityWeights = cardinalityWeights(1:maxComponents);
        allowedCardinalityWeights = ...
            cardinalityWeights(minComponents:maxComponents);
        if any(~isfinite(usedCardinalityWeights)) || ...
                any(usedCardinalityWeights < 0) || ...
                ~any(allowedCardinalityWeights > 0)
            error('JammingConfig:InvalidImbalanceWeights', ...
                ['使用的 cardinality_weights 必须有限、非负，且允许阶数' ...
                ' min_components:max_components 内至少一个权重大于 0。']);
        end
        validate_pattern_weights( ...
            imbalance.combination.pattern_weights, enabledTypes, ...
            minComponents, maxComponents);
    else
        cardinalityWeights = imbalance.combination.cardinality_weights;
        if isnumeric(cardinalityWeights) && isreal(cardinalityWeights) && ...
                isvector(cardinalityWeights)
            imbalance.combination.cardinality_weights = reshape( ...
                double(cardinalityWeights), 1, []);
        else
            imbalance.combination.cardinality_weights = ...
                ones(1, maxComponents);
        end
    end

    if imbalance.power.enable
        validate_numeric_range(imbalance.power.relative_power_dB, ...
            'cfg.imbalance.power.relative_power_dB');
    end
end

function validate_pattern_weights(patternWeights, enabledTypes, ...
        minComponents, maxComponents)
%VALIDATE_PATTERN_WEIGHTS 检查模式权重表的列、权重和组合文本。

    if ~istable(patternWeights)
        error('JammingConfig:InvalidImbalanceWeights', ...
            'pattern_weights 必须是 table。');
    end
    if isempty(patternWeights)
        return;
    end
    requiredColumns = {'pattern_codes', 'weight'};
    if ~all(ismember(requiredColumns, patternWeights.Properties.VariableNames))
        error('JammingConfig:InvalidImbalanceWeights', ...
            '非空 pattern_weights 必须包含 pattern_codes 和 weight 两列。');
    end
    weights = patternWeights.weight;
    if ~isnumeric(weights) || ~isreal(weights) || ~isvector(weights) || ...
            numel(weights) ~= height(patternWeights) || ...
            any(~isfinite(weights)) || any(weights < 0) || ~any(weights > 0)
        error('JammingConfig:InvalidImbalanceWeights', ...
            'pattern_weights.weight 必须有限、非负且至少一个大于 0。');
    end

    patternCodes = normalize_pattern_codes(patternWeights.pattern_codes, ...
        height(patternWeights));
    canonicalPatterns = cell(height(patternWeights), 1);
    for rowIndex = 1:height(patternWeights)
        components = strsplit(patternCodes{rowIndex}, '+', ...
            'CollapseDelimiters', false);
        components = cellfun(@strtrim, components, ...
            'UniformOutput', false);
        if any(cellfun(@isempty, components))
            error('JammingConfig:InvalidImbalanceWeights', ...
                '第 %d 行 pattern_codes 含空的干扰类型。', rowIndex);
        end
        if numel(unique(components, 'stable')) ~= numel(components)
            error('JammingConfig:InvalidImbalanceWeights', ...
                '第 %d 行 pattern_codes 含重复干扰成分。', rowIndex);
        end
        unknownTypes = setdiff(components, enabledTypes, 'stable');
        if ~isempty(unknownTypes)
            error('JammingConfig:InvalidImbalanceWeights', ...
                '第 %d 行含未启用的干扰类型：%s。', rowIndex, ...
                strjoin(unknownTypes, ', '));
        end
        componentCount = numel(components);
        if componentCount < minComponents || componentCount > maxComponents
            error('JammingConfig:InvalidImbalanceWeights', ...
                ['第 %d 行组合阶数为 %d，必须位于 dataset 分量范围 ' ...
                '[%d, %d] 内。'], rowIndex, componentCount, ...
                minComponents, maxComponents);
        end
        canonicalPatterns{rowIndex} = strjoin(sort(components), '+');
    end
    if numel(unique(canonicalPatterns, 'stable')) ~= ...
            numel(canonicalPatterns)
        error('JammingConfig:InvalidImbalanceWeights', ...
            'pattern_weights 含重复的组合模式。');
    end
end

function patternCodes = normalize_pattern_codes(rawCodes, expectedRows)
%NORMALIZE_PATTERN_CODES 将模式文本列规范为 cellstr 列。

    if isstring(rawCodes)
        if any(ismissing(rawCodes), 'all')
            error('JammingConfig:InvalidImbalanceWeights', ...
                'pattern_codes 不能含缺失 string。');
        end
        patternCodes = cellstr(rawCodes(:));
    elseif ischar(rawCodes)
        patternCodes = cellstr(rawCodes);
    elseif iscell(rawCodes)
        patternCodes = cell(numel(rawCodes), 1);
        for index = 1:numel(rawCodes)
            code = rawCodes{index};
            if isstring(code) && isscalar(code) && ~ismissing(code)
                code = char(code);
            end
            if ~(ischar(code) && isrow(code))
                error('JammingConfig:InvalidImbalanceWeights', ...
                    'pattern_codes 的每个元素必须是文本标量。');
            end
            patternCodes{index} = code;
        end
    else
        error('JammingConfig:InvalidImbalanceWeights', ...
            'pattern_codes 必须是逐行文本组合。');
    end
    if numel(patternCodes) ~= expectedRows
        error('JammingConfig:InvalidImbalanceWeights', ...
            'pattern_codes 的文本数量必须等于表行数。');
    end
    patternCodes = reshape(patternCodes, [], 1);
end

function openset = normalize_and_validate_open_set(openset)
%NORMALIZE_AND_VALIDATE_OPEN_SET 校验开放集参数并转换数值形状与类型。

    ratios = openset.mixed_ratios;
    if ~isnumeric(ratios) || ~isreal(ratios) || ...
            ~isvector(ratios) || numel(ratios) ~= 4
        error('JammingConfig:InvalidOpenSetRatios', ...
            'mixed_ratios 必须是含四个元素的实数数值向量。');
    end
    ratios = reshape(double(ratios), 1, []);
    if any(~isfinite(ratios)) || any(ratios < 0) || ...
            abs(sum(ratios) - 1) > 1e-12
        error('JammingConfig:InvalidOpenSetRatios', ...
            'mixed_ratios 必须是四个有限非负数，且总和在 1e-12 内等于 1。');
    end
    openset.mixed_ratios = ratios;

    positiveFields = {'train_known', 'calibration_known'};
    nonnegativeFields = {'validation_mixed', 'test_mixed'};
    for index = 1:numel(positiveFields)
        fieldName = positiveFields{index};
        if ~is_valid_positive_integer(openset.(fieldName))
            error('JammingConfig:InvalidOpenSetSplit', ...
                ['cfg.openset.%s 必须是有限正整数，且不能超过 ' ...
                'flintmax。'], fieldName);
        end
        openset.(fieldName) = double(openset.(fieldName));
    end
    for index = 1:numel(nonnegativeFields)
        fieldName = nonnegativeFields{index};
        if ~is_valid_nonnegative_integer(openset.(fieldName))
            error('JammingConfig:InvalidOpenSetSplit', ...
                ['cfg.openset.%s 必须是有限非负整数，且不能超过 ' ...
                'flintmax。'], fieldName);
        end
        openset.(fieldName) = double(openset.(fieldName));
    end
end

function valid = is_positive_integer_range(value)
%IS_POSITIVE_INTEGER_RANGE 判断是否为有序 1×2 正整数范围。

    valid = isnumeric(value) && isreal(value) && ...
        isequal(size(value), [1, 2]) && ...
        all(isfinite(value)) && all(value > 0) && ...
        all(value == fix(value)) && value(1) <= value(2) && ...
        all(arrayfun(@is_double_exact_scalar, value));
end

function valid = is_valid_positive_scalar(value)
%IS_VALID_POSITIVE_SCALAR 判断是否为可精确转 double 的有限正数标量。

    valid = is_double_exact_scalar(value) && value > 0;
end

function valid = is_valid_nonnegative_scalar(value)
%IS_VALID_NONNEGATIVE_SCALAR 判断是否为可精确转 double 的非负标量。

    valid = is_double_exact_scalar(value) && value >= 0;
end

function valid = is_valid_positive_integer(value)
%IS_VALID_POSITIVE_INTEGER 判断是否为有限正整数标量。

    valid = is_valid_positive_scalar(value) && value == fix(value);
end

function valid = is_valid_nonnegative_integer(value)
%IS_VALID_NONNEGATIVE_INTEGER 判断是否为有限非负整数标量。

    valid = is_valid_nonnegative_scalar(value) && value == fix(value);
end

function valid = is_double_exact_scalar(value)
%IS_DOUBLE_EXACT_SCALAR 判断数值标量能否在 flintmax 内安全转为 double。

    valid = isnumeric(value) && isreal(value) && isscalar(value) && ...
        isfinite(value);
    if ~valid
        return;
    end
    if isa(value, 'uint64')
        valid = value <= uint64(flintmax);
    elseif isa(value, 'int64')
        limit = int64(flintmax);
        valid = value >= -limit && value <= limit;
    else
        valid = abs(double(value)) <= flintmax;
    end
end
