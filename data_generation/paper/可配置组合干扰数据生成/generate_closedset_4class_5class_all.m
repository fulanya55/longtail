function result = generate_closedset_4class_5class_all( ...
        executeGeneration, datasetSelection, levels_dB, ...
        splitSelection, existingPolicy)
%GENERATE_CLOSEDSET_4CLASS_5CLASS_ALL 串行生成四类和五类全部数据。
%   默认顺序为四类全功率、四类九挡、五类全功率、五类九挡；全程由
%   调用者启动的单一MATLAB进程执行，不进行并行正式生成。结束前重新
%   构建四个代表实验的train/test manifest，确认8个种子域两两无交集。

    add_local_paths();
    if nargin < 1, executeGeneration = true; end
    if nargin < 2, datasetSelection = 'all'; end
    if nargin < 3, levels_dB = 10:5:50; end
    if nargin < 4, splitSelection = 'both'; end
    if nargin < 5, existingPolicy = 'resume'; end

    levels_dB = normalize_power_sweep_levels(levels_dB);
    fourResult = generate_closedset_4class_datasets( ...
        executeGeneration, datasetSelection, levels_dB, ...
        splitSelection, existingPolicy);
    fiveResult = generate_closedset_5class_datasets( ...
        executeGeneration, datasetSelection, levels_dB, ...
        splitSelection, existingPolicy);

    configs = {create_closedset_4class_config(), ...
        create_power_sweep_4class_config(10), ...
        create_closedset_5class_config(), ...
        create_power_sweep_5class_config(10)};
    names = {'4class_full', '4class_sweep', ...
        '5class_full', '5class_sweep'};
    seedAudit = audit_closedset_seed_domains(configs, names);
    result = struct('four_class', fourResult, ...
        'five_class', fiveResult, 'seed_audit', seedAudit);
end

function add_local_paths()
%ADD_LOCAL_PATHS 保证从任意当前目录均可调用总入口。

    root = fileparts(mfilename('fullpath'));
    addpath(root, fullfile(root, 'core'), fullfile(root, 'generators'));
end
