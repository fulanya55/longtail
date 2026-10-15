function cfg = create_default_config()
%CREATE_DEFAULT_CONFIG 创建可配置组合干扰数据生成器的默认配置。
%   CFG = CREATE_DEFAULT_CONFIG() 不接收输入，返回一个标量结构体 CFG。
%   CFG 汇总随机种子、输出约束、雷达参数、数据集规模、十类干扰参数、
%   不平衡采样策略和开放集划分。字段名中的 _Hz、_s、_m、_mps 和 _dB
%   分别表示赫兹、秒、米、米每秒和分贝；无量纲比例或计数不带单位后缀。
%
%   修改示例：
%       cfg = create_default_config();
%       cfg.radar.num_pulses = 8;                 % 将脉冲数改为 8。
%       cfg.params.FMB.bandwidth_factor = [3, 6]; % 修改 FMB 带宽因子范围。
%       cfg.output.save_v73 = false;              % 关闭 MAT v7.3 保存格式。

    allTypes = {'FMB', 'AMA', 'SawFM', 'SinFM', 'SN', ...
        'MFT', 'VFT', 'ISRJ', 'RangeFakeTarget', 'Traction'};

    cfg = struct();

    % 基础字段控制可复现实验和默认输出位置。
    cfg.seed = 20260714;
    % 输出目录默认相对于调用函数时的当前工作目录；正式运行入口应允许用户
    % 根据任务位置和存储空间显式覆盖 cfg.output_dir。
    cfg.output_dir = fullfile(pwd, 'generated_data');
    cfg.output.save_v73 = true;
    cfg.output.save_legacy = true;
    cfg.output.max_estimated_bytes = 20 * 1024^3;
    cfg.output.power_tolerance_dB = 0.20;

    % 雷达字段定义采样、发射波形、目标和噪声条件。
    cfg.radar.c_mps = 3e8;
    cfg.radar.fs_Hz = 10e6;
    cfg.radar.pulse_width_s = 100e-6;
    cfg.radar.pri_s = 300e-6;
    cfg.radar.bandwidth_Hz = 5e6;
    cfg.radar.carrier_Hz = 10e9;
    cfg.radar.num_pulses = 4;
    cfg.radar.add_target = false;
    cfg.radar.target_SNR_dB = 20;
    cfg.radar.target_range_m = 10000;
    cfg.radar.noise_power = 1;

    % 类型字段固定十类顺序，并用稳定差集得到未知类顺序。
    cfg.types.enabled = allTypes;
    cfg.types.known = {'FMB', 'AMA', 'SawFM', 'MFT', 'ISRJ'};
    cfg.types.unknown = setdiff(allTypes, cfg.types.known, 'stable');

    % 数据集字段控制样本量、复合分量数量和组合采样方式。
    cfg.dataset.num_samples = 120;
    cfg.dataset.min_components = 1;
    cfg.dataset.max_components = 3;
    cfg.dataset.combination_mode = 'random';
    cfg.dataset.samples_per_pattern = 8;

    % 公共参数范围：到达时延单位为秒，干噪比单位为分贝。
    cfg.params.common.arrival_delay_s = [0, 50e-6];
    cfg.params.common.JNR_dB = [20, 40];

    % FMB 参数：带宽因子无量纲，中心频率单位为赫兹。
    cfg.params.FMB.bandwidth_factor = [2, 5];
    cfg.params.FMB.center_frequency_Hz = [0, 0];

    % AMA 参数：带宽因子和调制指数无量纲，中心频率单位为赫兹。
    cfg.params.AMA.bandwidth_factor = [1, 2];
    cfg.params.AMA.center_frequency_Hz = [0, 0];
    cfg.params.AMA.modulation_index = [1, 3];

    % SawFM 参数：扫频带宽单位为赫兹，扫频周期单位为秒。
    cfg.params.SawFM.sweep_bandwidth_Hz = [2e6, 5e6];
    cfg.params.SawFM.sweep_period_s = [20e-6, 80e-6];

    % SinFM 参数：三个频率字段单位均为赫兹。
    cfg.params.SinFM.center_frequency_Hz = [0.5e6, 1.5e6];
    cfg.params.SinFM.frequency_deviation_Hz = [1e6, 5e6];
    cfg.params.SinFM.modulation_frequency_Hz = [10e3, 20e3];

    % SN 参数：采样宽度单位为秒，转发副本数为整数计数。
    cfg.params.SN.sample_width_s = [5e-6, 20e-6];
    cfg.params.SN.num_replicas = [1, 4];

    % MFT 参数：时延单位为秒，相对幅度无量纲，最大目标数为计数。
    cfg.params.MFT.num_targets = [3, 8];
    cfg.params.MFT.target_delay_s = [30e-6, 50e-6];
    cfg.params.MFT.target_relative_amplitude = [0.5, 1.5];
    cfg.params.MFT.max_targets = 8;

    % VFT 参数：多普勒频率单位为赫兹。
    cfg.params.VFT.doppler_Hz = [1e3, 30e3];

    % ISRJ 参数：采样宽度单位为秒，副本数为计数，模式使用字符代码。
    cfg.params.ISRJ.sample_width_s = [5e-6, 20e-6];
    cfg.params.ISRJ.num_replicas = [1, 5];
    cfg.params.ISRJ.mode = 'direct';

    % 距离假目标参数：时延单位为秒，相对幅度无量纲。
    cfg.params.RangeFakeTarget.num_targets = [1, 4];
    cfg.params.RangeFakeTarget.target_delay_s = [10e-6, 80e-6];
    cfg.params.RangeFakeTarget.target_relative_amplitude = [0.5, 1.5];
    cfg.params.RangeFakeTarget.max_targets = 4;

    % 拖引参数：初始距离单位为米，速度单位为米每秒，幅度无量纲。
    cfg.params.Traction.initial_range_m = [1000, 5000];
    cfg.params.Traction.velocity_mps = [20, 200];
    cfg.params.Traction.amplitude = [0.5, 2.0];

    % 分量不平衡字段控制十类分量的相对抽样权重。
    cfg.imbalance.component.enable = false;
    cfg.imbalance.component.weights = ones(1, numel(allTypes));

    % 组合不平衡字段控制组合基数和具体组合模式的抽样权重。
    cfg.imbalance.combination.enable = false;
    cfg.imbalance.combination.cardinality_weights = [1, 1, 1];
    cfg.imbalance.combination.pattern_weights = table();
    cfg.imbalance.combination.min_samples_per_pattern = 1;

    % 功率不平衡字段控制弱分量轮换策略及相对功率范围，单位为分贝。
    cfg.imbalance.power.enable = false;
    cfg.imbalance.power.weak_component_policy = 'rotate';
    cfg.imbalance.power.relative_power_dB = [-20, -6];

    % 开放集字段定义已知类训练、校准及混合验证/测试样本数量。
    cfg.openset.train_known = 80;
    cfg.openset.calibration_known = 20;
    cfg.openset.validation_mixed = 40;
    cfg.openset.test_mixed = 60;
    cfg.openset.mixed_ratios = [0.25, 0.25, 0.35, 0.15];
end
