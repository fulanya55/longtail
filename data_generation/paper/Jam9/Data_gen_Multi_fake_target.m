function [out_sig, truth] = Data_gen_Multi_fake_target(sig_LOS, nPulse, ...
        fs, n_fake_tar_min, n_fake_tar_max, delay_min, delay_max, ...
        amp_min, amp_max, nBatch, nRepetition, flag_addTar, ...
        jam_delay_min, jam_delay_max)
%DATA_GEN_MULTI_FAKE_TARGET 生成多假目标干扰及可选最终总时延真值。
%   OUT_SIG = DATA_GEN_MULTI_FAKE_TARGET(...) 保持历史单输出接口，返回
%   (nBatch*nPulse)×length(sig_LOS) complex double 波形。
%
%   [OUT_SIG, TRUTH] = DATA_GEN_MULTI_FAKE_TARGET(...) 返回逐波形行真值。
%   请求 TRUTH 时 nRepetition 必须为 1。TRUTH.num_targets 在配置的闭区间
%   内均匀抽取；delay_s 和 amplitude 为固定容量矩阵，活动槽按最终总时延
%   升序成对排列，未使用槽保持 NaN。
%
%   公共到达时延与目标相对时延先在整数采样网格上相加，再生成波形和
%   标签。arrival_delay_s 保持 NaN，因为该公共量已吸收到逐目标最终总
%   时延，无法从单个最终波形中与相对时延唯一分离。历史 MFT 没有独立
%   JNR 控制量，因此 JNR_dB 同样保持 NaN。

    if nargout > 1 && nRepetition ~= 1
        error('LegacyJamming:TruthRequiresSingleRepetition', ...
            '请求逐波形真值时 nRepetition 必须等于 1。');
    end
    if n_fake_tar_min < 1 || n_fake_tar_max < n_fake_tar_min || ...
            n_fake_tar_min ~= fix(n_fake_tar_min) || ...
            n_fake_tar_max ~= fix(n_fake_tar_max)
        error('LegacyJamming:InvalidPhysicalParameter', ...
            'MFT 目标数量范围必须是有限正整数闭区间。');
    end

    reference = sig_LOS(:).';
    nSample = numel(reference);
    sampleCount = nBatch * nPulse;
    noiseAmplitude = 1 / sqrt(2);
    out_sig = complex(zeros(sampleCount, nSample));
    truth = struct();
    if nargout > 1
        truth = legacy_initialize_truth(sampleCount, {'JNR_dB', ...
            'num_targets', 'mean_delay_s', 'mean_spacing_s', ...
            'arrival_delay_s'});
        truth.delay_s = nan(sampleCount, n_fake_tar_max);
        truth.amplitude = nan(sampleCount, n_fake_tar_max);
    end

    for sampleIndex = 1:sampleCount
        for repetitionIndex = 1:nRepetition
            targetCount = n_fake_tar_min + floor( ...
                (n_fake_tar_max - n_fake_tar_min + 1) * rand());
            selection = legacy_sample_unique_delays( ...
                [delay_min, delay_max], [jam_delay_min, jam_delay_max], ...
                targetCount, fs, nSample);
            amplitudes = amp_min + (amp_max - amp_min) * ...
                rand(1, targetCount);
            amplitudes = amplitudes(selection.sort_order);
            jammer = complex(zeros(1, nSample));
            for targetIndex = 1:targetCount
                jammer = jammer + amplitudes(targetIndex) * ...
                    legacy_zero_boundary_delay(reference, ...
                    selection.total_samples(targetIndex));
            end
            out_sig(sampleIndex, :) = out_sig(sampleIndex, :) + jammer;

            if nargout > 1
                delays_s = selection.total_samples / fs;
                truth.num_targets(sampleIndex) = targetCount;
                truth.mean_delay_s(sampleIndex) = mean(delays_s);
                if targetCount > 1
                    truth.mean_spacing_s(sampleIndex) = mean(diff(delays_s));
                end
                truth.delay_s(sampleIndex, 1:targetCount) = delays_s;
                truth.amplitude(sampleIndex, 1:targetCount) = amplitudes;
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
