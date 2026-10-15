function cfg = create_closedset_5class_config()
%CREATE_CLOSEDSET_5CLASS_CONFIG 创建五类全功率闭集正式配置。
%   类型固定为FMB、SinFM、MFT、VFT、ISRJ，覆盖31种非空组合，每组合
%   400条，每个split共12400条。独立种子域300260722不与其他三个实验
%   复用。函数只返回配置，不创建目录或生成数据。

    outputRoot = fullfile( ...
        'G:\论文相关\干扰数据\复合干扰数据0716', '5类');
    cfg = create_closedset_subset_config( ...
        {'FMB', 'SinFM', 'MFT', 'VFT', 'ISRJ'}, ...
        300260722, fullfile(outputRoot, 'full_power_10_30dB_v1'));
end
