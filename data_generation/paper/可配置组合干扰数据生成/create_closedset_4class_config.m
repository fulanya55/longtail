function cfg = create_closedset_4class_config()
%CREATE_CLOSEDSET_4CLASS_CONFIG 创建四类全功率闭集正式配置。
%   类型固定为FMB、SawFM、MFT、ISRJ，覆盖15种非空组合，每组合400
%   条，每个split共6000条。独立种子域100260722不与四类九挡、五类
%   全功率或五类九挡复用。函数只返回配置，不创建目录或生成数据。

    outputRoot = fullfile( ...
        'G:\论文相关\干扰数据\复合干扰数据0716', '4类');
    cfg = create_closedset_subset_config( ...
        {'FMB', 'SawFM', 'MFT', 'ISRJ'}, ...
        100260722, fullfile(outputRoot, 'full_power_10_30dB_v1'));
end
