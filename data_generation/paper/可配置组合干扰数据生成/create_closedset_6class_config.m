function cfg = create_closedset_6class_config()
%CREATE_CLOSEDSET_6CLASS_CONFIG 创建 0716 六类闭集实验专用配置。
%   CFG = CREATE_CLOSEDSET_6CLASS_CONFIG() 以 CREATE_DEFAULT_CONFIG 为
%   完整字段模板，只覆盖本次实验已经确认的雷达参数、六类干扰范围、
%   组合方式、样本数量和输出位置。返回值仍可由用户在正式生成前继续
%   修改；公共入口会在使用配置前统一执行物理边界和容量校验。
%
%   本实验启用 FMB、SawFM、SinFM、MFT、VFT、ISRJ 六类，并把六类
%   全部定义为已知类，因此训练集和测试集均为闭集数据。组合模式为
%   all，枚举六类的全部 2^6-1=63 个非空组合，每种组合生成 400 条，
%   所以每个划分共有 63*400=25200 条样本。
%
%   示例：
%       cfg = create_closedset_6class_config();
%       disp(cfg.params.FMB.bandwidth_factor); % [1.5, 3]

    % 正式四/五/六类数据必须保持旧 MAT 的十类标签和 79 维参数顺序。
    cfg = create_default_config(1);

    % 使用与数据集日期一致的基础随机种子。train 和 test manifest 还会
    % 分别叠加固定划分偏移，因此两份数据的样本种子互不重叠。
    cfg.seed = 20260716;

    % 两个 MAT 文件最终写入同一目标目录。v7.3 支持超过 2 GB 的变量；
    % legacy 格式把实部和虚部拼接，便于直接兼容历史实值网络输入。
    cfg.output_dir = 'G:\论文相关\干扰数据\复合干扰数据0716';
    cfg.output.save_v73 = true;
    cfg.output.save_legacy = true;

    % 雷达体制沿用已有仿真程序：10 MHz 采样、100 us 脉宽、300 us PRI、
    % 5 MHz LFM 带宽和 10 GHz 载频。每条样本只保存一个 PRI，因此波形
    % 长度为 10e6*300e-6=3000 点。本实验不叠加真实目标，只在全部干扰
    % 分量合成后加入一次单位功率复高斯接收机噪声。
    cfg.radar.fs_Hz = 10e6;
    cfg.radar.pulse_width_s = 100e-6;
    cfg.radar.pri_s = 300e-6;
    cfg.radar.bandwidth_Hz = 5e6;
    cfg.radar.carrier_Hz = 10e9;
    cfg.radar.num_pulses = 1;
    cfg.radar.add_target = false;
    cfg.radar.noise_power = 1;

    % enabled 决定实际参与组合的类型；known 与 enabled 相同、unknown 为空
    % 表示严格闭集。后续标签 label_selected/known_label 均按注册表稳定顺序
    % 输出，即 FMB、SawFM、SinFM、MFT、VFT、ISRJ。
    selectedTypes = {'FMB', 'SawFM', 'SinFM', 'MFT', 'VFT', 'ISRJ'};
    cfg.types.enabled = selectedTypes;
    cfg.types.known = selectedTypes;
    cfg.types.unknown = {};

    % all 模式覆盖一至六阶全部组合；samples_per_pattern=400 表示每个
    % 组合模式各生成 400 条。num_samples 仅供 random 模式使用，在本次
    % all 模式下不会决定最终样本量，仍显式写成单个划分总量便于阅读。
    cfg.dataset.num_samples = 25200;
    cfg.dataset.min_components = 1;
    cfg.dataset.max_components = 6;
    cfg.dataset.combination_mode = 'all';
    cfg.dataset.samples_per_pattern = 400;

    % 六类干扰共有的到达时延和干噪比范围。JNR 是各分量相对于单位噪声
    % 功率的标称功率比；普通闭集 manifest 不再额外施加弱分量功率偏置。
    cfg.params.common.arrival_delay_s = [10e-6, 50e-6];
    cfg.params.common.JNR_dB = [10, 30];

    % FMB：带宽因子按用户最终确认设置为 1.5--3，中心频率固定为零。
    % 生成器会按照采样率和离散 FFT 频点裁剪到可实现的带宽，并把实际
    % 实现值写入参数真值，而不是把超出 Nyquist 的请求值伪装成真值。
    cfg.params.FMB.bandwidth_factor = [1.5, 3];
    cfg.params.FMB.center_frequency_Hz = [0, 0];

    % SawFM：锯齿扫频带宽 2--5 MHz，扫频周期 20--80 us；扫频斜率由
    % 每条样本的实际带宽和周期计算并保存。
    cfg.params.SawFM.sweep_bandwidth_Hz = [2e6, 5e6];
    cfg.params.SawFM.sweep_period_s = [20e-6, 80e-6];

    % SinFM：中心频率、频偏和调制频率分别在以下范围内随机抽取。
    cfg.params.SinFM.center_frequency_Hz = [0.5e6, 1.5e6];
    cfg.params.SinFM.frequency_deviation_Hz = [1e6, 3e6];
    cfg.params.SinFM.modulation_frequency_Hz = [10e3, 20e3];

    % MFT：每条分量包含 3--8 个假目标；target_delay_s 是相对于公共
    % 到达时延的附加延时，幅度是各假目标复制波形的相对幅度。
    cfg.params.MFT.num_targets = [3, 8];
    cfg.params.MFT.target_delay_s = [30e-6, 50e-6];
    cfg.params.MFT.target_relative_amplitude = [0.5, 1.5];
    cfg.params.MFT.max_targets = 8;

    % VFT：采用 1--7 kHz 多普勒频移，保证与当前离散 LFM 频率支持相交
    % 且不发生频谱折叠；对应速度由载频和多普勒频移自动换算。
    cfg.params.VFT.doppler_Hz = [1e3, 7e3];

    % ISRJ：抽取 5--20 us 采样片段并直接转发 1--5 次。direct 模式下
    % 生成器会检查原始片段和全部副本均完整位于 3000 点记录内。
    cfg.params.ISRJ.sample_width_s = [5e-6, 20e-6];
    cfg.params.ISRJ.num_replicas = [1, 5];
    cfg.params.ISRJ.mode = 'direct';
end
