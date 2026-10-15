function cfg = create_default_config(varargin)
%CREATE_DEFAULT_CONFIG 创建带版本选择的可配置组合干扰默认配置。
%   CFG = CREATE_DEFAULT_CONFIG() 返回十三类 v2 默认配置。
%   CFG = CREATE_DEFAULT_CONFIG(VERSION) 接受数值标量 1 或 2。VERSION=1
%   返回与历史程序完全相同的十类结构，不包含 cfg.registry 或三类新增参数；
%   VERSION=2 返回十三类结构，并显式保存 cfg.registry.version=2。
%   字段名中的 _Hz、_s、_m、_mps 和 _dB 分别表示赫兹、秒、米、米每秒
%   和分贝；无量纲比例或计数不带单位后缀。
%
%   修改示例：
%       cfg = create_default_config();
%       cfg.radar.num_pulses = 8;                 % 将脉冲数改为 8。
%       cfg.params.FMB.bandwidth_factor = [3, 6]; % 修改 FMB 带宽因子范围。
%       cfg.output.save_v73 = false;              % 关闭 MAT v7.3 保存格式。

    registryVersion = resolve_registry_version(varargin{:});
    legacyTypes = {'FMB', 'AMA', 'SawFM', 'SinFM', 'SN', ...
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

    % 先构造 v1 的十类稳定顺序；v2 在函数末尾只追加 ID 11--13。
    cfg.types.enabled = legacyTypes;
    cfg.types.known = {'FMB', 'AMA', 'SawFM', 'MFT', 'ISRJ'};
    cfg.types.unknown = setdiff(legacyTypes, cfg.types.known, 'stable');

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

    % v1 分量不平衡字段先按十类注册表给出默认相对抽样权重。
    cfg.imbalance.component.enable = false;
    cfg.imbalance.component.weights = ones(1, numel(legacyTypes));

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

    if registryVersion == 1
        return;
    end

    % v2 只在完整 v1 模板构造完成后追加字段，避免历史配置意外出现新键。
    newTypes = {'CombSpectrum', 'ClutterPulse', 'SpectralDispersion'};
    cfg.registry.version = 2;
    cfg.types.enabled = [legacyTypes, newTypes];
    cfg.types.known = {'FMB', 'AMA', 'SawFM', 'MFT', 'ISRJ', ...
        'CombSpectrum', 'ClutterPulse', 'SpectralDispersion'};
    cfg.types.unknown = setdiff( ...
        cfg.types.enabled, cfg.types.known, 'stable');

    % 梳状谱：齿数和固定容量为整数，频率单位为 Hz，幅度为无量纲比例。
    cfg.params.CombSpectrum.num_teeth = [3, 8];
    cfg.params.CombSpectrum.spacing_Hz = [0.2e6, 0.6e6];
    cfg.params.CombSpectrum.center_offset_Hz = [-0.3e6, 0.3e6];
    cfg.params.CombSpectrum.amplitude = [0.5, 1.5];
    cfg.params.CombSpectrum.amplitude_mode = 'equal';
    cfg.params.CombSpectrum.max_teeth = 8;

    % 杂乱脉冲：默认独立逐 PRI 抽样，固定容量为每 PRI 8 个、总计 32 个。
    cfg.params.ClutterPulse.pulses_per_pri = [3, 8];
    cfg.params.ClutterPulse.pulse_width_s = [2e-6, 20e-6];
    cfg.params.ClutterPulse.relative_amplitude = [0.5, 1.5];
    cfg.params.ClutterPulse.frequency_offset_Hz = [-1e6, 1e6];
    cfg.params.ClutterPulse.mode = 'independent';
    cfg.params.ClutterPulse.max_pulses_per_pri = 8;
    cfg.params.ClutterPulse.max_total_pulses = 32;

    % 标准 SMSP：整数 M 同时定义子脉冲数、理想压缩倍数和斜率倍数。
    cfg.params.SpectralDispersion.num_subpulses = [2, 8];
    cfg.params.SpectralDispersion.max_subpulses = 8;
    cfg.imbalance.component.weights = ones(1, numel(cfg.types.enabled));
end

function version = resolve_registry_version(varargin)
%RESOLVE_REGISTRY_VERSION 规范无参 v2 或显式 1/2 版本选择。

    if isempty(varargin)
        version = 2;
        return;
    end
    if numel(varargin) ~= 1
        throw_invalid_registry_version();
    end
    value = varargin{1};
    valid = isnumeric(value) && isreal(value) && isscalar(value) && ...
        isfinite(value) && value == fix(value) && ismember(double(value), [1, 2]);
    if ~valid
        throw_invalid_registry_version();
    end
    version = double(value);
end

function throw_invalid_registry_version()
%THROW_INVALID_REGISTRY_VERSION 抛出稳定的配置版本错误。

    error('JammingConfig:InvalidRegistryVersion', ...
        'registry version 必须是数值标量 1 或 2。');
end
