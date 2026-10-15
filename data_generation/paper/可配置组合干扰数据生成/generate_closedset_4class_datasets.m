function result = generate_closedset_4class_datasets( ...
        executeGeneration, datasetSelection, levels_dB, ...
        splitSelection, existingPolicy)
%GENERATE_CLOSEDSET_4CLASS_DATASETS 生成四类全功率和九挡闭集数据。
%   默认正式串行执行全功率train/test，再执行10:5:50九挡train/test。
%   DATASETSELECTION允许all、full_power、power_sweep；dry-run只返回
%   manifest摘要和配对证据，不创建4类目录。正式运行复用底层原子保存、
%   增强回载和resume机制。

    add_local_paths();
    if nargin < 1, executeGeneration = true; end
    if nargin < 2, datasetSelection = 'all'; end
    if nargin < 3, levels_dB = 10:5:50; end
    if nargin < 4, splitSelection = 'both'; end
    if nargin < 5, existingPolicy = 'resume'; end

    levels_dB = normalize_power_sweep_levels(levels_dB);
    sweepConfigs = cell(numel(levels_dB), 1);
    for index = 1:numel(levels_dB)
        sweepConfigs{index} = ...
            create_power_sweep_4class_config(levels_dB(index));
    end
    result = run_subset_dataset_family( ...
        create_closedset_4class_config(), sweepConfigs, ...
        executeGeneration, datasetSelection, splitSelection, ...
        existingPolicy);
end

function add_local_paths()
%ADD_LOCAL_PATHS 保证从任意当前目录均可调用公共入口。

    root = fileparts(mfilename('fullpath'));
    addpath(root, fullfile(root, 'core'), fullfile(root, 'generators'));
end
