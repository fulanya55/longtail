function cfg = create_power_sweep_6class_config(sweepJnr_dB)
%CREATE_POWER_SWEEP_6CLASS_CONFIG 创建六类九挡配对功率扫描配置。
%   CFG = CREATE_POWER_SWEEP_6CLASS_CONFIG(SWEEPJNR_DB) 在已经验证的
%   CREATE_CLOSEDSET_6CLASS_CONFIG 基础上构造一个功率扫描挡位。允许的
%   SWEEPJNR_DB 为 10:5:50 中的一个有限实数标量。
%
%   FMB、SawFM 和 SinFM 的最终 JNR 固定为 30 dB；MFT、VFT 和 ISRJ
%   的最终 JNR 同时设置为 SWEEPJNR_DB。所有六类生成器的公共标称 JNR
%   先固定为 30 dB，随后由 BUILD_CLOSEDSET_MANIFEST 根据本函数保存的
%   cfg.experiment.power_sweep 配置，为后三类写入 sweepJnr_dB-30 dB 的
%   manifest 相对功率偏置。这样无需改动物理生成器，并且组合层回写的
%   JNR 参数真值就是最终有效 JNR。
%
%   九个挡位使用相同 cfg.seed，因此 train/test 内相同 sample_id 在各挡
%   具有相同的组合、物理参数子流和接收机噪声子流。输出目录按挡位独立，
%   防止覆盖原始 0716 闭集数据以及其他功率挡位。
%
%   示例：
%       cfg = create_power_sweep_6class_config(10);
%       disp(cfg.output_dir);
%       disp(cfg.experiment.power_sweep.sweep_JNR_dB);

    standardLevels_dB = 10:5:50;
    if ~(isnumeric(sweepJnr_dB) && isreal(sweepJnr_dB) && ...
            isscalar(sweepJnr_dB) && isfinite(sweepJnr_dB) && ...
            ismember(double(sweepJnr_dB), standardLevels_dB))
        error('JammingPowerSweep:InvalidLevel', ...
            'sweepJnr_dB 必须是九挡集合 10:5:50 中的一个有限实数标量。');
    end
    sweepJnr_dB = double(sweepJnr_dB);
    levelIndex = find(standardLevels_dB == sweepJnr_dB, 1, 'first');

    % 继承六类、63种组合、每组合400条以及全部已批准的物理参数。
    cfg = create_closedset_6class_config();

    % 公共生成器只采样标称JNR；固定为30后，所有挡位的随机调用路径完全
    % 一致。实际类型功率差由 manifest 偏置确定性实现。
    cfg.params.common.JNR_dB = [30, 30];

    % 每挡位使用独立子目录，目录名同时编码序号和真实JNR，便于人工审计。
    sweepRoot = fullfile( ...
        'G:\论文相关\干扰数据\复合干扰数据0716', ...
        'power_sweep_9level_v1');
    levelDirectory = sprintf( ...
        'level_%02d__jnr_%ddB', levelIndex, sweepJnr_dB);
    cfg.output_dir = fullfile(sweepRoot, levelDirectory);

    % experiment 字段随 config 一起写入 MAT；回载验证据此重建每个活动
    % 分量的期望有效JNR、相对偏置和参数真值。
    cfg.experiment = struct();
    cfg.experiment.name = 'power_sweep_9level_v1';
    cfg.experiment.power_sweep = struct();
    cfg.experiment.power_sweep.fixed_codes = ...
        {'FMB', 'SawFM', 'SinFM'};
    cfg.experiment.power_sweep.swept_codes = ...
        {'MFT', 'VFT', 'ISRJ'};
    cfg.experiment.power_sweep.fixed_JNR_dB = 30;
    cfg.experiment.power_sweep.sweep_JNR_dB = sweepJnr_dB;
    cfg.experiment.power_sweep.standard_levels_dB = standardLevels_dB;
    cfg.experiment.power_sweep.level_index = levelIndex;
    cfg.experiment.power_sweep.paired_seeds = true;
end
