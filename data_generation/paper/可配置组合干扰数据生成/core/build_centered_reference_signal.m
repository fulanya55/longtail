function centered = build_centered_reference_signal(cfg, reference)
%BUILD_CENTERED_REFERENCE_SIGNAL 将现有离散LFM搬移到零中心复基带。
%   CENTERED = BUILD_CENTERED_REFERENCE_SIGNAL(CFG, REFERENCE) 接收已规范化
%   配置 CFG 和 BUILD_REFERENCE_SIGNAL 生成的 REFERENCE，并返回标量结构体
%   CENTERED。pulse 为 1×samples_per_pri 的复数单 PRI 信号，train 为该
%   pulse 按 PRI 精确重复 num_pulses 次的复数行向量；support_Hz 给出搬移
%   后的离散有效频率支撑 [下限, 上限]，frequency_shift_Hz 给出实际施加
%   的频移，二者单位均为 Hz。
%
%   离散频率支撑由相邻有效样点的共轭乘积相位估计：相位差先解缠，再乘
%   fs/(2*pi) 转换为 Hz。搬移量使用该离散支撑的实际中点，而不是名义
%   带宽 B/2，因此能够补偿有限样点造成的端点偏差，并使返回支撑严格
%   关于零对称。调制仅改变相位，不改变原有效脉冲的包络；PRI 余下样点
%   保持复零填充。
%
%   本函数不读取或推进随机数流，不写文件，也不修改 CFG 或 REFERENCE。
%   少于两个有效样点时无法形成相邻相位差，将稳定抛出
%   JammingGenerator:InvalidPhysicalParameter。

    fs_Hz = cfg.radar.fs_Hz;
    samplesPerPulse = cfg.derived.samples_per_pulse;
    samplesPerPri = cfg.derived.samples_per_pri;
    active = complex(reference.pulse(1:samplesPerPulse));
    if samplesPerPulse < 2
        error('JammingGenerator:InvalidPhysicalParameter', ...
            '零中心参考LFM至少需要两个有效样点。');
    end

    % 相邻相位差对应离散样点区间上的实现频率；解缠后换算为 Hz。
    adjacentPhase = unwrap(angle(active(2:end) .* ...
        conj(active(1:end - 1))));
    frequency_Hz = adjacentPhase * fs_Hz / (2 * pi);
    originalSupport_Hz = [min(frequency_Hz), max(frequency_Hz)];
    midpoint_Hz = mean(originalSupport_Hz);
    timePulse_s = (0:samplesPerPulse - 1) / fs_Hz;
    centeredActive = active .* ...
        exp(-1i * 2 * pi * midpoint_Hz * timePulse_s);

    centeredPulse = complex(zeros(1, samplesPerPri));
    centeredPulse(1:samplesPerPulse) = centeredActive;
    centeredTrain = complex(repmat( ...
        centeredPulse, 1, cfg.radar.num_pulses));
    centered = struct( ...
        'pulse', centeredPulse, ...
        'train', centeredTrain, ...
        'support_Hz', originalSupport_Hz - midpoint_Hz, ...
        'frequency_shift_Hz', -midpoint_Hz);
end
