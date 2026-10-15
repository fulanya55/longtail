function selection = legacy_sample_unique_delays(delayRange_s, ...
        arrivalRange_s, targetCount, fs_Hz, sampleLength)
%LEGACY_SAMPLE_UNIQUE_DELAYS 抽取无碰撞且能完整落入记录的目标总时延。
%   SELECTION = LEGACY_SAMPLE_UNIQUE_DELAYS(DELAYRANGE_S,
%   ARRIVALRANGE_S, TARGETCOUNT, FS_HZ, SAMPLELENGTH) 先在公共到达时延
%   范围内抽取一个整数采样点，再从目标相对时延的离散槽中无放回抽取
%   TARGETCOUNT 个值。只有满足 total_delay < SAMPLELENGTH 的槽才可使用。
%
%   返回字段：
%     relative_samples  按最终总时延排序后的目标相对时延采样点；
%     arrival_samples   本次公共到达时延采样点；
%     total_samples     已排序的最终总时延采样点；
%     sort_order        从抽取顺序到最终时延升序的置换，用于同步重排幅度。
%
%   可用槽不足时抛出 LegacyJamming:InsufficientDelaySlots，禁止静默生成
%   重复目标时延或越界波形。

    validate_nonnegative_range(delayRange_s, 'delayRange_s');
    validate_nonnegative_range(arrivalRange_s, 'arrivalRange_s');
    validCount = isnumeric(targetCount) && isreal(targetCount) && ...
        isscalar(targetCount) && isfinite(targetCount) && ...
        targetCount >= 1 && targetCount == fix(targetCount);
    validRate = isnumeric(fs_Hz) && isreal(fs_Hz) && ...
        isscalar(fs_Hz) && isfinite(fs_Hz) && fs_Hz > 0;
    validLength = isnumeric(sampleLength) && isreal(sampleLength) && ...
        isscalar(sampleLength) && isfinite(sampleLength) && ...
        sampleLength >= 1 && sampleLength == fix(sampleLength);
    if ~(validCount && validRate && validLength)
        error('LegacyJamming:InvalidPhysicalParameter', ...
            '目标数、采样率和记录长度必须为有效标量。');
    end

    arrivalMinimum = round(arrivalRange_s(1) * fs_Hz);
    arrivalMaximum = round(arrivalRange_s(2) * fs_Hz);
    arrivalSamples = arrivalMinimum + floor( ...
        (arrivalMaximum - arrivalMinimum + 1) * rand());
    relativeMinimum = round(delayRange_s(1) * fs_Hz);
    relativeMaximum = round(delayRange_s(2) * fs_Hz);
    candidates = relativeMinimum:relativeMaximum;
    candidates = candidates(candidates >= 0 & ...
        candidates + arrivalSamples < sampleLength);
    if numel(candidates) < targetCount
        error('LegacyJamming:InsufficientDelaySlots', ...
            '可用离散时延槽少于目标数量。');
    end

    chosen = candidates(randperm(numel(candidates), targetCount));
    totalUnsorted = chosen + arrivalSamples;
    [totalSamples, sortOrder] = sort(totalUnsorted);
    selection = struct( ...
        'relative_samples', chosen(sortOrder), ...
        'arrival_samples', arrivalSamples, ...
        'total_samples', totalSamples, ...
        'sort_order', sortOrder);
end

function validate_nonnegative_range(value, name)
%VALIDATE_NONNEGATIVE_RANGE 校验有限非负有序 1×2 范围。

    if ~isnumeric(value) || ~isreal(value) || ...
            ~isequal(size(value), [1, 2]) || any(~isfinite(value)) || ...
            value(1) < 0 || value(1) > value(2)
        error('LegacyJamming:InvalidPhysicalParameter', ...
            '%s 必须是有限非负有序 1×2 范围。', name);
    end
end
