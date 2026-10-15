function [y, measuredPower_dB] = calibrate_component_power( ...
        x, targetPower_dB)
%CALIBRATE_COMPONENT_POWER 将一个波形标定到指定平均功率。
%   [Y, MEASUREDPOWER_DB] = CALIBRATE_COMPONENT_POWER(X, TARGETPOWER_DB)
%   接收非空实数或复数数值向量 X，以及有限实数标量 TARGETPOWER_DB。
%   输出 Y 为 double，并保持 X 的行/列形状；MEASUREDPOWER_DB 为标定后
%   通过最大幅度归一化后的稳定 RMS 实测得到的 double 标量。
%
%   TARGETPOWER_DB 表示功率分贝，对应目标 RMS 幅度为 10^(dB/20)。函数
%   先取实部和虚部分量绝对值的最大值归一化，不会预先计算可能溢出的
%   原始复数模；归一化后再计算 RMS，因而可避免普通
%   mean(abs(X).^2) 的上溢和下溢。最终只施加一个正实缩放因子，不添加
%   噪声，也不改变任何非零样点的相位或波形方向。
%
%   示例：
%       x = exp(1i * 2 * pi * (0:127) / 128);
%       [y, measuredPower_dB] = calibrate_component_power(x, -12);

    if nargin ~= 2 || ~isnumeric(x) || isempty(x) || ~isvector(x) || ...
            ~all(isfinite(real(x(:)))) || ...
            ~all(isfinite(imag(x(:)))) || ...
            ~integer_values_fit_double(x)
        throw_invalid_component_power();
    end
    if ~isnumeric(targetPower_dB) || ~isreal(targetPower_dB) || ...
            ~isscalar(targetPower_dB) || ~isfinite(targetPower_dB) || ...
            ~integer_values_fit_double(targetPower_dB)
        throw_invalid_component_power();
    end

    % double 转换保留原始行/列形状，并统一后续功率运算的数值类型。
    x = double(x);
    targetPower_dB = double(targetPower_dB);
    [componentScale, rmsNorm, currentPower_dB, validInputPower] = ...
        stable_rms_summary(x);
    if ~validInputPower || ~isfinite(currentPower_dB)
        throw_invalid_component_power();
    end

    % 分步除法避免 componentScale*rmsNorm 本身发生上溢或下溢。
    unitSignal = (x / componentScale) / rmsNorm;
    targetRms = 10 ^ (targetPower_dB / 20);
    if ~isfinite(targetRms) || targetRms <= 0
        throw_invalid_component_power();
    end

    y = unitSignal * targetRms;
    if ~all(isfinite(real(y(:)))) || ~all(isfinite(imag(y(:))))
        throw_invalid_component_power();
    end

    [~, ~, measuredPower_dB, validMeasuredPower] = stable_rms_summary(y);
    if ~validMeasuredPower || ~isfinite(measuredPower_dB)
        throw_invalid_component_power();
    end
end

function [componentScale, rmsNorm, power_dB, valid] = ...
        stable_rms_summary(signal)
%STABLE_RMS_SUMMARY 用实虚分量尺度归一化稳定计算 RMS 及功率分贝。

    realScale = max(abs(real(signal)));
    imaginaryScale = max(abs(imag(signal)));
    componentScale = max(realScale, imaginaryScale);
    if ~isfinite(componentScale) || componentScale <= 0
        rmsNorm = NaN;
        power_dB = NaN;
        valid = false;
        return;
    end

    scaledSignal = signal / componentScale;
    rmsNorm = sqrt(mean(abs(scaledSignal) .^ 2));
    if ~isfinite(rmsNorm) || rmsNorm <= 0
        power_dB = NaN;
        valid = false;
        return;
    end

    power_dB = 20 * log10(componentScale) + 20 * log10(rmsNorm);
    valid = isfinite(power_dB);
end

function valid = integer_values_fit_double(value)
%INTEGER_VALUES_FIT_DOUBLE 拒绝超出 double 连续整数域的整数样点。

    if isa(value, 'uint64')
        valid = all(value(:) <= uint64(flintmax));
    elseif isa(value, 'int64')
        limit = int64(flintmax);
        valid = all(value(:) >= -limit & value(:) <= limit);
    else
        valid = true;
    end
end

function throw_invalid_component_power()
%THROW_INVALID_COMPONENT_POWER 抛出分量功率标定的统一错误标识。

    error('JammingGenerator:InvalidComponentPower', ...
        ['分量必须是可精确转换的非空有限数值向量且具有正有限 RMS；' ...
        '目标功率必须对应有限非零 RMS，且输出不得包含非有限值。']);
end
