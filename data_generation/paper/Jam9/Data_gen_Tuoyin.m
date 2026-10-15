function [out_sig, out_sig_pc, truth] = Data_gen_Tuoyin(B, Tp, freq0, ...
        fs, Tr, R_min, R_max, A_min, A_max, v_min, v_max, ...
        nBatch, nPulse)
%DATA_GEN_TUOYIN 生成四阶段拖引干扰、脉压结果及可选轨迹真值。
%   [OUT_SIG, OUT_SIG_PC] = DATA_GEN_TUOYIN(...) 保持历史前两个输出：
%   OUT_SIG 为时域拖引干扰，OUT_SIG_PC 为对应匹配滤波结果。输出行数为
%   nBatch*nPulse，每四行属于同一条拖引轨迹的四个阶段。
%
%   [OUT_SIG, OUT_SIG_PC, TRUTH] = DATA_GEN_TUOYIN(...) 额外返回逐行
%   轨迹真值。四个阶段相对初始时刻的时间固定为 [0,1,4,9] s。每组只
%   抽取一次初始距离、速度和幅度，并按
%       range(t) = initial_range + velocity*t
%       delay(t) = 2*range(t)/c
%   计算实际传播时延。初始距离因此直接作用于第一阶段波形，不再只是
%   被采样但未进入仿真。
%
%   nominal_amplitude 与 effective_amplitude 在本原子函数中相等；若组合
%   场景需要功率偏置，只允许在组合层把偏置换算为线性增益并更新有效
%   幅度，不能再引入一套独立 JNR 缩放。所有延迟均为零边界非循环延迟。

    c_mps = 3e8;
    totalRows = nBatch * nPulse;
    if ~isnumeric(totalRows) || ~isreal(totalRows) || ...
            ~isscalar(totalRows) || ~isfinite(totalRows) || ...
            totalRows < 4 || totalRows ~= fix(totalRows) || ...
            mod(totalRows, 4) ~= 0
        error('LegacyJamming:InvalidPhysicalParameter', ...
            'Traction 输出行数 nBatch*nPulse 必须是 4 的正整数倍。');
    end

    pulseSamples = round(Tp * fs);
    samplesPerPri = round(Tr * fs);
    validSampling = isfinite(pulseSamples) && isfinite(samplesPerPri) && ...
        pulseSamples >= 1 && samplesPerPri >= pulseSamples && ...
        abs(pulseSamples - Tp * fs) <= 16 * eps(max(1, Tp * fs)) && ...
        abs(samplesPerPri - Tr * fs) <= 16 * eps(max(1, Tr * fs));
    if ~validSampling
        error('LegacyJamming:InvalidPhysicalParameter', ...
            'Tp*fs 和 Tr*fs 必须是正整数，且 PRI 采样点数不少于脉宽。');
    end
    if any(~isfinite([B, Tp, freq0, fs, Tr, R_min, R_max, ...
            A_min, A_max, v_min, v_max])) || fs <= 0 || Tp <= 0 || ...
            Tr <= 0 || B <= 0 || R_min < 0 || R_min > R_max || ...
            A_min <= 0 || A_min > A_max || v_min > v_max
        error('LegacyJamming:InvalidPhysicalParameter', ...
            'Traction 物理参数必须为有限且有序的有效范围。');
    end

    chirpSlope_Hz_per_s = B / Tp;
    pulseTime_s = (0:pulseSamples - 1) / fs - Tp / 2;
    basePulse = exp(1i * (2 * pi * freq0 * pulseTime_s + ...
        pi * chirpSlope_Hz_per_s * pulseTime_s .^ 2));
    referencePulse = [basePulse, ...
        complex(zeros(1, samplesPerPri - pulseSamples))];
    stageTime_s = [0, 1, 4, 9];
    groupCount = totalRows / 4;
    out_sig = complex(zeros(totalRows, samplesPerPri));
    out_sig_pc = complex(zeros(totalRows, samplesPerPri));
    truth = struct();
    if nargout > 2
        truth = legacy_initialize_truth(totalRows, {'initial_range_m', ...
            'velocity_mps', 'nominal_amplitude', 'effective_amplitude', ...
            'range_change_per_stage_m', 'delay_change_per_stage_s', ...
            'stage_time_s', 'current_range_m', 'current_delay_s'});
    end

    for groupIndex = 1:groupCount
        initialRange_m = R_min + (R_max - R_min) * rand();
        amplitude = A_min + (A_max - A_min) * rand();
        velocity_mps = v_min + (v_max - v_min) * rand();
        for stageIndex = 1:4
            outputRow = (groupIndex - 1) * 4 + stageIndex;
            elapsed_s = stageTime_s(stageIndex);
            rangeChange_m = velocity_mps * elapsed_s;
            currentRange_m = initialRange_m + rangeChange_m;
            currentDelay_s = 2 * currentRange_m / c_mps;
            delaySamples = round(currentDelay_s * fs);
            if ~isfinite(currentRange_m) || currentRange_m < 0 || ...
                    ~isfinite(currentDelay_s) || currentDelay_s < 0 || ...
                    delaySamples + pulseSamples > samplesPerPri
                error('LegacyJamming:InvalidPhysicalParameter', ...
                    ['Traction 每阶段必须满足 range>=0 且传播时延后的' ...
                    '完整脉冲仍位于一个 PRI 记录内。']);
            end

            delayedPulse = amplitude * legacy_zero_boundary_delay( ...
                referencePulse, delaySamples);
            out_sig(outputRow, :) = delayedPulse;
            out_sig_pc(outputRow, :) = ifftshift( ...
                range_process_mf(referencePulse, delayedPulse, ...
                samplesPerPri));

            if nargout > 2
                truth.initial_range_m(outputRow) = initialRange_m;
                truth.velocity_mps(outputRow) = velocity_mps;
                truth.nominal_amplitude(outputRow) = amplitude;
                truth.effective_amplitude(outputRow) = amplitude;
                truth.range_change_per_stage_m(outputRow) = rangeChange_m;
                truth.delay_change_per_stage_s(outputRow) = ...
                    2 * rangeChange_m / c_mps;
                truth.stage_time_s(outputRow) = elapsed_s;
                truth.current_range_m(outputRow) = currentRange_m;
                truth.current_delay_s(outputRow) = delaySamples / fs;
            end
        end
    end
end
