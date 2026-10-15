function levels = normalize_power_sweep_levels(value)
%NORMALIZE_POWER_SWEEP_LEVELS 规范九挡集合的非空无重复子集。
%   LEVELS = NORMALIZE_POWER_SWEEP_LEVELS(VALUE) 只接受10:5:50中的有限
%   实数向量，保留调用者给定顺序并拒绝重复。公共四类、五类入口共用
%   本函数，避免两份参数校验发生漂移。

    valid = isnumeric(value) && isreal(value) && isvector(value) && ...
        ~isempty(value) && all(isfinite(value)) && ...
        all(ismember(double(value), 10:5:50));
    if valid
        levels = reshape(double(value), 1, []);
        valid = numel(unique(levels, 'stable')) == numel(levels);
    end
    if ~valid
        error('JammingPowerSweep:InvalidLevels', ...
            'levels_dB必须是10:5:50的非空无重复数值子集。');
    end
end
