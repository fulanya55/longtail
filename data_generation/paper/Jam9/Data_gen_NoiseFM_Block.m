function [out_sig, truth] = Data_gen_NoiseFM_Block(sig_LOS, nPulse, ...
        JNR_min, JNR_max, L_min, L_max, sig_bandwidth, freq0, fs, nPP, ...
        nBatch, nRepetition, flag_addTar, jam_delay_min, jam_delay_max)
%DATA_GEN_NOISEFM_BLOCK 生成阻塞式噪声调频干扰及可选可观测真值。
%   OUT_SIG = DATA_GEN_NOISEFM_BLOCK(...) 保持历史单输出接口，返回
%   (nBatch*nPulse)×nPP complex double 波形。背景噪声和可选目标回波的
%   加入顺序与历史函数一致。
%
%   [OUT_SIG, TRUTH] = DATA_GEN_NOISEFM_BLOCK(...) 额外返回逐波形行真值。
%   请求 TRUTH 时 nRepetition 必须等于 1，确保一个输出波形只对应一组
%   参数。TRUTH 包含 JNR_dB、离散实现 bandwidth_factor、
%   interference_bandwidth_Hz、center_frequency_Hz、arrival_delay_s，
%   以及仅供诊断、不进入监督参数模式的 requested_bandwidth_factor。
%
%   FMB 先将请求干扰带宽换算为调制噪声带宽，再通过共轭对称 FFT 掩膜
%   离散化。监督带宽由实际掩膜反算，不能直接复制连续请求值。到达时延
%   使用零边界非循环延迟，尾部样本不会回绕到记录开头。

    if nargout > 1 && nRepetition ~= 1
        error('LegacyJamming:TruthRequiresSingleRepetition', ...
            '请求逐波形真值时 nRepetition 必须等于 1。');
    end

    nSample = nPP;
    sampleCount = nBatch * nPulse;
    noiseAmplitude = 1 / sqrt(2);
    phase0 = 0;
    effectiveFmIndex = 10;
    time_s = (0:nSample - 1) / fs;
    out_sig = complex(zeros(sampleCount, nSample));
    truth = struct();
    if nargout > 1
        truth = legacy_initialize_truth(sampleCount, {'JNR_dB', ...
            'bandwidth_factor', 'interference_bandwidth_Hz', ...
            'center_frequency_Hz', 'arrival_delay_s', ...
            'requested_bandwidth_factor'});
    end

    for sampleIndex = 1:sampleCount
        for repetitionIndex = 1:nRepetition
            requestedFactor = L_min + (L_max - L_min) * rand();
            requestedInterferenceBandwidth_Hz = ...
                requestedFactor * sig_bandwidth;
            requestedDeviation_Hz = requestedInterferenceBandwidth_Hz / ...
                (2 * sqrt(2 * log(2)));
            requestedNoiseBandwidth_Hz = ...
                requestedDeviation_Hz / effectiveFmIndex;
            [filterMask, realizedNoiseBandwidth_Hz] = ...
                legacy_realized_noise_support(nSample, fs, ...
                requestedNoiseBandwidth_Hz);
            realizedInterferenceBandwidth_Hz = ...
                realizedNoiseBandwidth_Hz * effectiveFmIndex * ...
                2 * sqrt(2 * log(2));
            realizedFactor = realizedInterferenceBandwidth_Hz / ...
                sig_bandwidth;

            JNR_dB = JNR_min + (JNR_max - JNR_min) * rand();
            jammerAmplitude = 10 ^ (JNR_dB / 20);
            whiteNoise = randn(1, nSample);
            coloredNoise = real(ifft(fft(whiteNoise) .* filterMask));
            coloredNoise = coloredNoise - mean(coloredNoise);
            noiseStd = std(coloredNoise);
            if ~isfinite(noiseStd) || noiseStd <= 0
                error('LegacyJamming:InvalidPhysicalParameter', ...
                    'FMB 离散带限噪声必须具有有限正标准差。');
            end
            realizedDeviation_Hz = realizedInterferenceBandwidth_Hz / ...
                (2 * sqrt(2 * log(2)));
            fmSlope = realizedDeviation_Hz / noiseStd;
            integratedNoise = cumsum(coloredNoise / fs);
            jammer = jammerAmplitude * exp(1i * ( ...
                2 * pi * freq0 * time_s + ...
                2 * pi * fmSlope * integratedNoise + phase0));
            out_sig(sampleIndex, :) = ...
                out_sig(sampleIndex, :) + jammer;

            if nargout > 1
                truth.JNR_dB(sampleIndex) = JNR_dB;
                truth.bandwidth_factor(sampleIndex) = realizedFactor;
                truth.interference_bandwidth_Hz(sampleIndex) = ...
                    realizedInterferenceBandwidth_Hz;
                truth.center_frequency_Hz(sampleIndex) = freq0;
                truth.requested_bandwidth_factor(sampleIndex) = ...
                    requestedFactor;
            end
        end

        delaySamples = round((jam_delay_min + ...
            (jam_delay_max - jam_delay_min) * rand()) * fs);
        out_sig(sampleIndex, :) = legacy_zero_boundary_delay( ...
            out_sig(sampleIndex, :), delaySamples);
        out_sig(sampleIndex, :) = out_sig(sampleIndex, :) + ...
            noiseAmplitude * (randn(1, nSample) + 1i * randn(1, nSample));
        if flag_addTar == 1
            out_sig(sampleIndex, :) = Add1tar( ...
                out_sig(sampleIndex, :), sig_LOS, nPulse, fs);
        end
        if nargout > 1
            truth.arrival_delay_s(sampleIndex) = delaySamples / fs;
        end
    end
end
