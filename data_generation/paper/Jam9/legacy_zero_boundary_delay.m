function shifted = legacy_zero_boundary_delay(signal, delaySamples)
%LEGACY_ZERO_BOUNDARY_DELAY 执行整数采样点的正向非循环延迟。
%   SHIFTED = LEGACY_ZERO_BOUNDARY_DELAY(SIGNAL, DELAYSAMPLES) 沿 SIGNAL
%   的第二维移动数据。左侧空出的采样点补零，越过记录末端的数据被截断，
%   因而不会像 circshift 一样把尾部样本回绕到记录开头。
%
%   SIGNAL 可以是行向量或二维矩阵，所有实部和虚部必须有限。
%   DELAYSAMPLES 必须是有限非负整数标量。延迟大于或等于记录列数时返回
%   同尺寸全零数组，并保持输入数值类型和复数属性。

    validSignal = isnumeric(signal) && ~isempty(signal) && ...
        ismatrix(signal) && all(isfinite(real(signal)), 'all') && ...
        all(isfinite(imag(signal)), 'all');
    validDelay = isnumeric(delaySamples) && isreal(delaySamples) && ...
        isscalar(delaySamples) && isfinite(delaySamples) && ...
        delaySamples >= 0 && delaySamples == fix(delaySamples);
    if ~(validSignal && validDelay)
        error('LegacyJamming:InvalidPhysicalParameter', ...
            ['signal 必须是有限数值向量或矩阵，delaySamples 必须是' ...
            '有限非负整数标量。']);
    end

    shifted = zeros(size(signal), 'like', signal);
    if delaySamples < size(signal, 2)
        shifted(:, delaySamples + 1:end) = ...
            signal(:, 1:end - delaySamples);
    end
end
