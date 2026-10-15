function [out_sig, truth] = Data_gen_NoiseAM_Spot(sig_LOS, nPulse, ...
        JNR_min, JNR_max, L_min, L_max, sig_bandwidth, freq0, fs, nPP, ...
        nBatch, nRepetition, flag_addTar, jam_delay_min, jam_delay_max)
%DATA_GEN_NOISEAM_SPOT 生成瞄准式噪声调幅干扰及可选可观测真值。
%   OUT_SIG = DATA_GEN_NOISEAM_SPOT(...) 保持历史单输出接口，返回
%   (nBatch*nPulse)×nPP complex double 波形。
%
%   [OUT_SIG, TRUTH] = DATA_GEN_NOISEAM_SPOT(...) 返回逐波形行真值。
%   请求 TRUTH 时 nRepetition 必须为 1。TRUTH 包含 JNR_dB、离散实现
%   bandwidth_factor、interference_bandwidth_Hz、center_frequency_Hz、
%   modulation_index、arrival_delay_s，以及只用于诊断的连续请求带宽因子。
%
%   调制噪声由共轭对称 FFT 掩膜生成，实际带宽由选中的正负频率对计算。
%   历史常数 U0=3 对应归一化包络 1+noise/U0，因此可观测调制指数记为
%   1/U0。到达时延使用零边界非循环延迟。

    if nargout > 1 && nRepetition ~= 1
        error('LegacyJamming:TruthRequiresSingleRepetition', ...
            '请求逐波形真值时 nRepetition 必须等于 1。');
    end

    nSample = nPP;
    sampleCount = nBatch * nPulse;
    noiseAmplitude = 1 / sqrt(2);
    carrierEnvelope = 3;
    modulationIndex = 1 / carrierEnvelope;
    phase0 = 0;
    time_s = (0:nSample - 1) / fs;
    out_sig = complex(zeros(sampleCount, nSample));
    truth = struct();
    if nargout > 1
        truth = legacy_initialize_truth(sampleCount, {'JNR_dB', ...
            'bandwidth_factor', 'interference_bandwidth_Hz', ...
            'center_frequency_Hz', 'modulation_index', ...
            'arrival_delay_s', 'requested_bandwidth_factor'});
    end

    for sampleIndex = 1:sampleCount
        for repetitionIndex = 1:nRepetition
            requestedFactor = L_min + (L_max - L_min) * rand();
            requestedBandwidth_Hz = requestedFactor * sig_bandwidth;
            [filterMask, realizedBandwidth_Hz] = ...
                legacy_realized_noise_support(nSample, fs, ...
                requestedBandwidth_Hz);
            realizedFactor = realizedBandwidth_Hz / sig_bandwidth;
            JNR_dB = JNR_min + (JNR_max - JNR_min) * rand();

            whiteNoise = randn(1, nSample);
            coloredNoise = real(ifft(fft(whiteNoise) .* filterMask));
            coloredNoise = real(sqrt(fs / realizedBandwidth_Hz) * ...
                coloredNoise);
            jammerEnvelope = 10 ^ (JNR_dB / 20) / ...
                sqrt(carrierEnvelope ^ 2 + 1) * ...
                (carrierEnvelope + coloredNoise);
            jammer = jammerEnvelope .* exp(1i * ( ...
                2 * pi * freq0 * time_s + phase0));
            out_sig(sampleIndex, :) = ...
                out_sig(sampleIndex, :) + jammer;

            if nargout > 1
                truth.JNR_dB(sampleIndex) = JNR_dB;
                truth.bandwidth_factor(sampleIndex) = realizedFactor;
                truth.interference_bandwidth_Hz(sampleIndex) = ...
                    realizedBandwidth_Hz;
                truth.center_frequency_Hz(sampleIndex) = freq0;
                truth.modulation_index(sampleIndex) = modulationIndex;
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
