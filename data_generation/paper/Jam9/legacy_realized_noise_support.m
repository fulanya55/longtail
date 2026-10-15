function [mask, realizedBandwidth_Hz, info] = ...
        legacy_realized_noise_support(sampleLength, fs_Hz, ...
        requestedBandwidth_Hz)
%LEGACY_REALIZED_NOISE_SUPPORT 构造实噪声的共轭对称离散低通支撑。
%   [MASK, REALIZEDBANDWIDTH_HZ, INFO] =
%   LEGACY_REALIZED_NOISE_SUPPORT(SAMPLELENGTH, FS_HZ,
%   REQUESTEDBANDWIDTH_HZ) 使用未移位 FFT 的索引顺序生成 1×N logical
%   掩膜。掩膜包含 DC，并在正、负频率侧选择数量完全相同的非零频点，
%   因而与实值时域噪声所需的共轭对称性一致。
%
%   连续请求带宽会向下量化到采样网格上能够成对表示的实际带宽。
%   REALIZEDBANDWIDTH_HZ 是最小负频率到最大正频率之间的实际跨度，
%   即 2*K*(FS_HZ/SAMPLELENGTH)。无配对的偶数长度 Nyquist 单点不会被
%   选中。若请求只能留下 DC，则抛出
%   LegacyJamming:InvalidPhysicalParameter，避免产生零带宽伪真值。

    validLength = isnumeric(sampleLength) && isreal(sampleLength) && ...
        isscalar(sampleLength) && isfinite(sampleLength) && ...
        sampleLength >= 3 && sampleLength == fix(sampleLength);
    validRate = isnumeric(fs_Hz) && isreal(fs_Hz) && ...
        isscalar(fs_Hz) && isfinite(fs_Hz) && fs_Hz > 0;
    validBandwidth = isnumeric(requestedBandwidth_Hz) && ...
        isreal(requestedBandwidth_Hz) && ...
        isscalar(requestedBandwidth_Hz) && ...
        isfinite(requestedBandwidth_Hz) && requestedBandwidth_Hz > 0;
    if ~(validLength && validRate && validBandwidth)
        error('LegacyJamming:InvalidPhysicalParameter', ...
            '采样点数、采样率和请求带宽必须为有限有效值。');
    end

    sampleLength = double(sampleLength);
    fs_Hz = double(fs_Hz);
    requestedBandwidth_Hz = double(requestedBandwidth_Hz);
    binSpacing_Hz = fs_Hz / sampleLength;
    maxPositiveBins = floor((sampleLength - 1) / 2);
    positiveBinCount = min(floor(requestedBandwidth_Hz / ...
        (2 * binSpacing_Hz)), maxPositiveBins);
    if positiveBinCount < 1
        error('LegacyJamming:InvalidPhysicalParameter', ...
            '离散带宽必须至少包含一对非零共轭 FFT 频点。');
    end

    mask = false(1, sampleLength);
    mask(1) = true;
    mask(2:positiveBinCount + 1) = true;
    mask(end - positiveBinCount + 1:end) = true;
    realizedBandwidth_Hz = 2 * positiveBinCount * binSpacing_Hz;
    info = struct( ...
        'bin_spacing_Hz', binSpacing_Hz, ...
        'positive_bin_count', positiveBinCount, ...
        'support_bin_count', 2 * positiveBinCount + 1, ...
        'requested_bandwidth_Hz', requestedBandwidth_Hz);
end
