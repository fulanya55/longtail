function [out_sig, truth] = Data_gen_Range_fake_target(sig_LOS, nPulse, ...
        fs, delay_min, delay_max, amp_min, amp_max, nBatch, ...
        nRepetition, flag_addTar, jam_delay_min, jam_delay_max)
%DATA_GEN_RANGE_FAKE_TARGET 生成距离假目标干扰及可选最终总时延真值。
%   OUT_SIG = DATA_GEN_RANGE_FAKE_TARGET(...) 保持历史单输出接口。历史
%   模型每次重复生成一个距离假目标。
%
%   [OUT_SIG, TRUTH] = DATA_GEN_RANGE_FAKE_TARGET(...) 返回逐波形行真值。
%   请求 TRUTH 时 nRepetition 必须为 1。TRUTH 使用 v1 固定容量 4：
%   delay_s、range_m、amplitude 的第一个槽为活动目标，其余槽保持 NaN。
%   range_m 由最终总时延按 c*delay/2 计算。
%
%   公共到达时延已吸收到最终总时延，因此 arrival_delay_s 保持 NaN；
%   历史函数没有独立 JNR 控制量，因此 JNR_dB 也保持 NaN。延迟使用零
%   边界非循环操作，禁止记录尾部回绕到开头。

    if nargout > 1 && nRepetition ~= 1
        error('LegacyJamming:TruthRequiresSingleRepetition', ...
            '请求逐波形真值时 nRepetition 必须等于 1。');
    end

    c_mps = 3e8;
    reference = sig_LOS(:).';
    nSample = numel(reference);
    sampleCount = nBatch * nPulse;
    noiseAmplitude = 1 / sqrt(2);
    out_sig = complex(zeros(sampleCount, nSample));
    truth = struct();
    if nargout > 1
        truth = legacy_initialize_truth(sampleCount, {'JNR_dB', ...
            'num_targets', 'mean_delay_s', 'arrival_delay_s'});
        truth.delay_s = nan(sampleCount, 4);
        truth.range_m = nan(sampleCount, 4);
        truth.amplitude = nan(sampleCount, 4);
    end

    for sampleIndex = 1:sampleCount
        for repetitionIndex = 1:nRepetition
            selection = legacy_sample_unique_delays( ...
                [delay_min, delay_max], [jam_delay_min, jam_delay_max], ...
                1, fs, nSample);
            amplitude = amp_min + (amp_max - amp_min) * rand();
            jammer = amplitude * legacy_zero_boundary_delay( ...
                reference, selection.total_samples);
            out_sig(sampleIndex, :) = out_sig(sampleIndex, :) + jammer;

            if nargout > 1
                delay_s = selection.total_samples / fs;
                truth.num_targets(sampleIndex) = 1;
                truth.mean_delay_s(sampleIndex) = delay_s;
                truth.delay_s(sampleIndex, 1) = delay_s;
                truth.range_m(sampleIndex, 1) = c_mps * delay_s / 2;
                truth.amplitude(sampleIndex, 1) = amplitude;
            end
        end

        out_sig(sampleIndex, :) = out_sig(sampleIndex, :) + ...
            noiseAmplitude * (randn(1, nSample) + 1i * randn(1, nSample));
        if flag_addTar == 1
            out_sig(sampleIndex, :) = Add1tar( ...
                out_sig(sampleIndex, :), reference, nPulse, fs);
        end
    end
end
